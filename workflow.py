from langgraph.graph import StateGraph, END
from AgentState import AgentState
from nodes import agent, naver_retrieve, input_retrieve, db_retrieve, combiner, generate, rewrite
from edges import which_retrieved, grade_documents, should_continue

# workflow function for LangGraph
# we need these variables to run this function.
def run_workflow(input_query, doc_path, openai_api_key, document_db, supporting_db, llm, agent_components, chat_history):
    
    # define StateGraph
    workflow = StateGraph(AgentState)
    
    # add nodes
    workflow.add_node("agent", agent)
    workflow.add_node("naver_retrieve", naver_retrieve)
    workflow.add_node("input_retrieve", input_retrieve)
    workflow.add_node("db_retrieve", db_retrieve)
    workflow.add_node("combiner", combiner)
    workflow.add_node("generate", generate)
    workflow.add_node("rewrite", rewrite)
    
    # start of the my LangGraph
    workflow.set_entry_point("agent")
    # 어떤 파일을 검색할지 정하는 edge 
    workflow.add_conditional_edges(
        "agent",
        which_retrieved,
        {
            "user_file": "input_retrieve",
            "db": "db_retrieve",
            "search engine": "naver_retrieve"
        }
    )
    
    # combiner에 들어가야 양쪽 정보 모두 활용 가능
    workflow.add_edge("input_retrieve", "combiner")
    workflow.add_edge("db_retrieve", "combiner")
    workflow.add_edge("naver_retrieve", "combiner")
    workflow.add_edge("combiner","generate")
    
    # evaluate generated_answer
    workflow.add_conditional_edges(
        "generate",
        grade_documents,
        {
            "yes": END,
            "no": "rewrite",
        },
    )
    # rewrite question for better answer
    workflow.add_conditional_edges(
        "rewrite",
        should_continue,
        {
            "continue": "agent",
            "end": END
        }
    )

    app = workflow.compile()

    # 기본 agentstate
    initial_state = {
        "input": input_query, 
        "doc_path": doc_path,
        "openai_api_key": openai_api_key,
        "agent_scratchpad": [],
        "agent_response":"",
        "document_db": document_db,
        "supporting_db": supporting_db,
        "llm" : llm,
        "agent_components": agent_components,
        "chat_history": chat_history
    }
    result = app.invoke(initial_state)
    return result

# extract final answer from final agent state
def extract_final_response(result):
    
    final_response = result.get("generated_answer", "")
    
    
    return final_response

