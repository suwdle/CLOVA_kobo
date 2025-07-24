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
    """Retrieves documents and adds source metadata for the context protocol."""
    print("---RETRIEVING DOCUMENTS---")
    strategy = state["retrieval_strategy"]
    source_type = strategy.source
    query = strategy.query
    
    documents = []
    if source_type == "user_document" and state.get('document_db'):
        print(f"Retrieving from user document with query: {query}")
        retriever = state['document_db'].as_retriever(k=5)
        documents = retriever.invoke(query)
        for doc in documents:
            doc.metadata['source'] = 'user_document'
            doc.metadata['name'] = 'User-Uploaded Document'

    elif source_type == "support_programs":
        print(f"Retrieving from support programs DB with query: {query}")
        db = state.get('supporting_db')
        if db:
            retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}, k=8)
            documents = retriever.invoke(query)
            for doc in documents:
                doc.metadata['source'] = 'support_programs_db'
                # Assuming the CSV has a '사업명' column or similar
                doc.metadata['name'] = doc.page_content.split(',')[0] # Basic name extraction

    elif source_type == "financial_products":
        print(f"Retrieving from financial products DB with query: {query}")
        financial_db = state.get('financial_db')
        if not financial_db:
            financial_db = load_public_db(embeddings, db_type='financial')
            state['financial_db'] = financial_db
        
        if financial_db:
            retriever = financial_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}, k=8)
            documents = retriever.invoke(query)
            for doc in documents:
                doc.metadata['source'] = 'financial_products_db'
                # Assuming the CSV has a '상품명' column
                doc.metadata['name'] = doc.page_content.split(',')[0] # Basic name extraction
    else:
        print("No specific data source required or available.")

    print(f"Retrieved {len(documents)} documents.")
    return {"retrieved_docs": documents}


# --- Node 3: Generate Answer ---
def generate_answer(state: AgentState, llm: ChatOpenAI):
    """Generates a final answer based on the structured context."""
    print("---GENERATING ANSWER---")

    # 1. Build the structured context string
    context_str = "<CONTEXT>\n"
    for doc in state['retrieved_docs']:
        source = doc.metadata.get('source', 'unknown')
        name = doc.metadata.get('name', 'Unnamed Document')
        priority = 'high' if source == 'user_document' else 'medium'
        
        context_str += f'  <SOURCE type="{source}" priority="{priority}">\n'
        context_str += f'    <DOCUMENT name="{name}">\n'
        context_str += f'      {doc.page_content}\n'
        context_str += f'    </DOCUMENT>\n'
        context_str += f'  </SOURCE>\n'
    context_str += "</CONTEXT>"

    # 2. Define the system prompt with instructions on how to use the context
    system_prompt = """
    You are an expert AI assistant for small and medium-sized enterprises (SMEs).
    Your task is to provide clear, concise, and accurate answers in Korean based on the structured information provided in the <CONTEXT> block.

    **Instructions for Interpreting the Context:**
    - The <CONTEXT> block contains all the information you should use.
    - Each piece of information is wrapped in a <SOURCE> tag, which has a `type` attribute indicating where it came from (e.g., `user_document`, `financial_products_db`).
    - Pay close attention to the `priority` attribute. Information with `priority="high"` is the most important.
    - When generating your answer, it is helpful to cite the source of your information to build trust (e.g., "사용자께서 제공해주신 문서에 따르면..." or "금융상품 데이터베이스에 따르면...").
    - If the context is empty or does not contain relevant information to answer the question, clearly state that you could not find the necessary information.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Please provide a comprehensive answer to my question based on the provided context.\n\n{context}\n\nUser Question: {input}")
    ])
    
    generation_chain = prompt | llm
    
    response = generation_chain.invoke({
        "input": state['input'],
        "context": context_str,
        "chat_history": state['chat_history']
    })
    
    print(f"Generated Answer: {response.content}")
    return {"generated_answer": response.content}
