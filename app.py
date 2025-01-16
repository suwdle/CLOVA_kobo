from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from agent_components import initialize_agent_components
import requests
from build_vector_db import document_to_vector_db, public_to_vector_db
from ExtractLink import ExtractLink
from workflow import run_workflow, extract_final_response
from flask_cors import CORS

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)

# 환경 변수 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables")

BACKEND_URL = "http://localhost:5050"

# 벡터 데이터베이스 초기화
supporting_db = public_to_vector_db()

# LLM 초기화
llm = ChatOpenAI(temperature=0.5, model='gpt-4o', openai_api_key=openai_api_key)
# agent_components 초기화
agent_components = initialize_agent_components(llm)
# 전역 변수라 오류 생길수도? -> 빌드 후 확인
chat_history = []
@app.route('/', methods=['POST'])
def process_request():
    try:
        # 클라이언트로부터 JSON 데이터 가져오기 (Java에서 전달됨)
        data = request.get_json()
        documentId = data.get('documentId')
        query = data.get('content')

        if not query:
            return jsonify({"error": "Invalid request: 'content' is required"}), 400

        print(f"질문 받아오기 성공: {query}")

        doc_path = None
        document_db = None

        # 사용자 문서가 있을 때
        if documentId:
            document_response = requests.get(f'{BACKEND_URL}/api/documents/{documentId}', timeout=10)
            document_response.raise_for_status()
            document = document_response.json()
            doc_path = document.get('document')
            print(f"사용자 문서 받아오기 성공: {doc_path}")

            # PDF를 벡터 데이터베이스로 변환
            document_db = document_to_vector_db(doc_path)
            print("document converted to vector database")
        else:
            print("사용자 문서가 없는 질문입니다.")


        # 워크플로우 실행
        result = run_workflow(query, doc_path, openai_api_key, document_db, supporting_db, llm, agent_components, chat_history)
        print("Workflow executed")
        # 최종 응답 추출
        answer = extract_final_response(result)
        print(f"Extracted answer: {answer}")
        # chat history for llm
        chat_history.append(query)
        chat_history.append(answer)
        # 데이터 추출
        extracted_data = ExtractLink(answer, llm)
        print(f"Extracted data: {extracted_data}")

        # 분석 결과를 클라이언트로 반환 (Java로)
        #  query, answer, label까지 반환으로 추가 (backend와 상의)
        analysis_result = {
            "documentId": documentId,
            "content": query,
            "result": answer
        }

        return jsonify(analysis_result), 200

    except requests.RequestException as e:
        print(f"Error in API request: {e}")
        return jsonify({"error": "Error in API request", "details": str(e)}), 500
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
