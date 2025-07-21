from langgraph.graph import StateGraph, END
from graph.AgentState import AgentState
from graph.nodes import determine_retrieval_strategy, retrieve_documents, generate_answer
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def run_workflow(query: str, document_db, supporting_db, llm: ChatOpenAI, embeddings: OpenAIEmbeddings, chat_history: list):
    """Defines and runs the main agentic workflow."""

    # Define the state graph
    workflow = StateGraph(AgentState)

    # Add nodes
    # We bind the LLM and embeddings instances to the node functions where they are needed.
    workflow.add_node("determine_strategy", lambda state: determine_retrieval_strategy(state, llm))
    workflow.add_node("retrieve_documents", lambda state: retrieve_documents(state, embeddings))
    workflow.add_node("generate_answer", lambda state: generate_answer(state, llm))

    # Define the edges
    workflow.set_entry_point("determine_strategy")
    workflow.add_edge("determine_strategy", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "generate_answer")
    workflow.add_edge("generate_answer", END)

    # Compile the graph
    app = workflow.compile()

    # Prepare the initial state
    initial_state = {
        "input": query,
        "document_db": document_db,
        "supporting_db": supporting_db,
        "financial_db": None, # Lazily loaded in the retrieval node
        "chat_history": chat_history,
        "retrieved_docs": [],
    }

    # Run the workflow
    result = app.invoke(initial_state)
    return result