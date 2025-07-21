from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from typing import List, Literal
from graph.AgentState import AgentState
from vector_db.build_vector_db import load_public_db
from langchain_openai import OpenAIEmbeddings
import os

# --- Tool Definitions for Function Calling ---
class RetrievalStrategy(BaseModel):
    """Determines the best data source to query based on the user's question."""
    source: Literal["user_document", "support_programs", "financial_products", "general"] = Field(
        description="The optimal data source to use for answering the user's query. 'general' is for conversational or off-topic questions."
    )
    query: str = Field(
        description="A rewritten, optimized query for the selected data source."
    )

# --- Node 1: Determine Retrieval Strategy ---
def determine_retrieval_strategy(state: AgentState, llm: ChatOpenAI):
    """Uses the LLM to decide which information source is most relevant."""
    print("---DETERMINING RETRIEVAL STRATEGY---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are an expert at analyzing user queries and determining the best data source to find the answer.
        Based on the user's question, decide which of the following sources is most appropriate:

        - **user_document**: For questions about a specific file or document the user has provided.
        - **support_programs**: For questions about government or public support programs for small and medium-sized enterprises (SMEs).
        - **financial_products**: For questions about loans, investments, or other financial products for businesses.
        - **general**: For conversational greetings, off-topic questions, or when no specific data source is needed.
        
        You must also refine the user's query to be more effective for information retrieval.
        """),
        ("user", "Chat History:\n{chat_history}\n\nUser Question: {input}")
    ])
    
    strategy_chain = prompt | llm.with_structured_output(RetrievalStrategy)
    strategy = strategy_chain.invoke({
        "input": state['input'],
        "chat_history": state['chat_history']
    })
    
    print(f"Strategy: {strategy.source}, Query: {strategy.query}")
    # Note: We are returning a dictionary that will update the state.
    return {"retrieval_strategy": strategy}


# --- Node 2: Retrieve Documents ---
def retrieve_documents(state: AgentState, embeddings: OpenAIEmbeddings):
    """Retrieves documents from the appropriate source based on the strategy."""
    print("---RETRIEVING DOCUMENTS---")
    strategy = state["retrieval_strategy"]
    source = strategy.source
    query = strategy.query
    
    retrieved_docs = []
    if source == "user_document" and state.get('document_db'):
        print(f"Retrieving from user document with query: {query}")
        retriever = state['document_db'].as_retriever(k=5)
        retrieved_docs = retriever.invoke(query)
    elif source == "support_programs":
        print(f"Retrieving from support programs DB with query: {query}")
        db = state.get('supporting_db')
        if db:
            retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}, k=8)
            retrieved_docs = retriever.invoke(query)
    elif source == "financial_products":
        print(f"Retrieving from financial products DB with query: {query}")
        # Lazily load the financial DB to save resources if not always needed
        financial_db = state.get('financial_db')
        if not financial_db:
            financial_db = load_public_db(embeddings, db_type='financial')
            state['financial_db'] = financial_db
        
        if financial_db:
            retriever = financial_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}, k=8)
            retrieved_docs = retriever.invoke(query)
    else:
        print("No specific data source required or available.")

    print(f"Retrieved {len(retrieved_docs)} documents.")
    return {"retrieved_docs": retrieved_docs}


# --- Node 3: Generate Answer ---
def generate_answer(state: AgentState, llm: ChatOpenAI):
    """Generates a final answer based on the retrieved context and query."""
    print("---GENERATING ANSWER---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are an expert AI assistant for small and medium-sized enterprises (SMEs).
        Your task is to provide clear, concise, and accurate answers based on the provided context.
        Always respond in Korean.
        If the context is empty, state that you could not find relevant information.
        When referencing specific programs or products, include any available links.
        """),
        ("user", "Based on the following information:\n\nContext:\n{context}\n\nUser Question: {input}\nPlease provide a comprehensive answer.")
    ])
    
    generation_chain = prompt | llm
    
    context = "\n".join([doc.page_content for doc in state['retrieved_docs']])
    
    response = generation_chain.invoke({
        "input": state['input'],
        "context": context,
        "chat_history": state['chat_history']
    })
    
    print(f"Generated Answer: {response.content}")
    return {"generated_answer": response.content}