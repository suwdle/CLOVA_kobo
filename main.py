from graph.workflow import run_workflow, extract_final_response
from dotenv import load_dotenv
import os
from vector_db.build_vector_db import public_to_vector_db, document_to_vector_db
from graph.agent_components import initialize_agent_components
from langchain_openai import ChatOpenAI
from tools.ExtractLink import ExtractLink
import joblib

if __name__ == "__main__":
    load_dotenv()
    # API 키 가져오기
    openai_api_key = os.getenv("OPENAI_API_KEY")
    # 에러 처리
    if not openai_api_key:
        raise ValueError("OpenAI API key not found in environment variables")
    # load naive_bayesian classifier and vectorizer for query classification
    model = joblib.load('naive_bayes_model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
    supporting_db = public_to_vector_db()
    llm = ChatOpenAI(temperature=0.4, model='gpt-4o-mini',openai_api_key= openai_api_key)
    agent_components = initialize_agent_components(llm)
    chat_history = []
    doc_path = '/home/seokjun/Downloads/어텐션 알고리즘과 트랜스포머.pptx'
    document_db = document_to_vector_db(doc_path)
    while True:        
        query = input("질문을 입력하세요 (종료하려면 'exit' 입력): ")
        # 텍스트 데이터 벡터화
        query_vec = vectorizer.transform([query])
        # 모델을 사용하여 예측
        prediction = model.predict(query_vec)
        print("Prediction:", prediction)    

        # exit 입력시 반복문 탈출
        if query.lower() == 'exit':
            break
        result = run_workflow(query, doc_path, openai_api_key, document_db, supporting_db, llm, agent_components, chat_history)
        # LangGraph의 마지막 답변을 추출
        response = extract_final_response(result)
        print("답변:", response)
        chat_history.append(query)
        chat_history.append(response)
        extracted_data = ExtractLink(response, llm)
        