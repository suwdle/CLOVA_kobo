# import library
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.tools import Tool
from langchain_community.vectorstores import FAISS
from typing import List, Dict
from AgentState import AgentState
from typing import List, Dict, Any
import urllib.request
from tools import extract_content
from dotenv import load_dotenv
import os

# 1. agent node
def agent(state):

    chat_history = state.get('chat_history', [])
    agent_components = state.get('agent_components', None)
    if not agent_components:
        raise ValueError("agent_components not found in state")

    base_chain = agent_components["base_chain"]
    response = base_chain.invoke({"input":state['input'], "chat_history": chat_history})
    
    return {"agent_response": response.content}

# 네이버 뉴스 검색 api를 이용한 뉴스 검색 후 관련된 기사 내용 반환
def naver_retrieve(state: AgentState) -> Dict[str, List[Dict[str, Any]]]:
    load_dotenv()
    # API 키 가져오기
    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")
    print("enter the naver engine")

    agent_response = state.get('agent_response', '')
    words_for_searching = extract_content(agent_response)
    print(words_for_searching)
    encText = urllib.parse.quote(words_for_searching)

    url = 'https://openapi.naver.com/v1/search/news?query='+ encText
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-id",client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)
    response = urllib.request.urlopen(request)
    rescode = response.getcode()
    if(rescode==200):
        response_body = response.read()
        search_response = response_body.decode('utf-8')
        return {"naver_docs": search_response}
    else:
        print("Error Code:" + rescode)

# 2. user's input retrieve node
def input_retrieve(state: AgentState) -> Dict[str, List[Dict[str, Any]]]:
    # 입력이 dict 타입이 아닌 경우 예외 처리
    if not isinstance(state, dict) and not hasattr(state, 'get'):
        raise TypeError(f"Expected state to be a dict or have a 'get' method, but got {type(state)}")  

    # vectorstore 가져오기
    vectorstore = state.get('document_db', None)

    # vectorstore가 None인 경우 작업 스킵
    if vectorstore is None:
        print("vectorstore is None, skipping retrieval.")
        return {"retrieved_docs": []}  # 빈 리스트 반환

    # agent_response 가져오기
    agent_response = state.get('agent_response', '').lower()
    words_for_searching = extract_content(agent_response)
    # retriever 설정 및 문서 검색
    retriever = vectorstore.as_retriever(k=7)
    docs = retriever.invoke(words_for_searching)
    print("PDF retrieved and ready")
    
    return {"retrieved_docs": docs}

# 4. DB retrieve node
def db_retrieve(state: AgentState) -> Dict[str, List[Dict[str, Any]]] :
    if not isinstance(state, dict) and not hasattr(state, 'get'):
        raise TypeError(f"Expected state to be a dict or have a 'get' method, but got {type(state)}")
    supporting_db = state.get('supporting_db', FAISS)
    agent_response = state.get('agent_response', '').lower()
    words_for_searching = extract_content(agent_response)
    retriever = supporting_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold":0.5}, k=8)

    docs = retriever.invoke(words_for_searching)
    return {"db_docs": docs}

# 5. combiner node
# 검색 결과를 합쳐주는 노드. 이원화된 db에서 정보를 가져와 답변할 수 있도록 함
def combiner(state: AgentState) -> Dict[str, List[Dict[str, Any]]]:
    retrieved_docs = state.get('retrieved_docs', []) or []
    db_docs = state.get('db_docs', []) or []
    combined_result = retrieved_docs + db_docs
    print('combiner complete')
    return {"combined_result": combined_result}

# 6 . rewrite node
# 그래프를 한번 더 순환하게 될 때, 질문을 검색한 텍스트를 이용해 증강해주는 agent node
def rewrite(state):
    combined_result = state.get('combined_result', [])
    combined_text = " ".join([doc.page_content for doc in combined_result])

    agent_components = state.get('agent_components', None)
    if not agent_components:
        raise ValueError("agent_components not found in state")
    rewrite_chain = agent_components['rewrite_chain']
    
    rewritten_info = rewrite_chain.invoke({
        "input": state['input'],
        "context": combined_text,
        "answer": state["generated_answer"]
    })
    
    return {"input": rewritten_info.content}



# 7. generate node
# 사용자를 위한 답변을 생성하는 agent node
def generate(state):
    combined_result = state.get('combined_result', [])
    if combined_result is None :
        combined_text = ''
    else:
        combined_text = ' '.join(doc.page_content for doc in combined_result)
    
    naver_docs = state.get('naver_docs','')
    if naver_docs is None:
        naver_text = ''
    else:
        naver_text = naver_docs
    agent_components = state.get('agent_components', None)
    if not agent_components:
        raise ValueError("agent_components not found in state")
    generate_chain = agent_components['generate_chain']
    
    generated_info = generate_chain.invoke({
        "context": combined_text + naver_text,
        "query": state.get('input'),
        "agent_response": state.get('agent_response', [])
    })
    
    return {"generated_answer": generated_info.content}

