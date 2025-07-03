from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from graph.agent_components import initialize_agent_components
import requests
from vector_db.build_vector_db import document_to_vector_db, public_to_vector_db
from tools.ExtractLink import ExtractLink
from graph.workflow import run_workflow, extract_final_response
from flask_cors import CORS
import joblib
import boto3
from botocore.exceptions import NoCredentialsError
# Flask 앱 초기화
app = Flask(__name__)
CORS(app)

# 환경 변수 로드
load_dotenv()
clova_api_key = os.getenv("CLOVA_API_KEY")
if not clova_api_key:
    raise ValueError("Clova API key not found in environment variables")

aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_REGION')
s3_bucket_name = os.getenv('S3_BUCKET_NAME')

s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=aws_region
)
BACKEND_URL = "https://co-worker.store" # 배포 URL로 대체

# Initialize vector db
embeddings = OpenAIEmbeddings(api_key=clova_api_key, base_url="https://clovastudio.stream.ntruss.com/v1/openai")
supporting_db = public_to_vector_db(embeddings)

# Initialize OpenAI llm
llm = ChatOpenAI(temperature=0.5, model='gpt-3.5-turbo', api_key=clova_api_key, base_url="https://clovastudio.stream.ntruss.com/v1/openai")
# agent_components
agent_components = initialize_agent_components(llm)
model = joblib.load('naive_bayes_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

chat_history = []
@app.route('/', methods=['POST'])
def process_request():
    try:
        # data and documentID from client
        data = request.get_json()
        document_data = data.get('document')
        file_name = document_data.get('originalFileName')
        file_url = document_data.get('fileUrl')
        query = data.get('content')

        if not query:
            return jsonify({"error": "Invalid request: 'content' is required"}), 400

        print(f"질문 받아오기 성공: {query}")

        doc_path = None
        document_db = None

        # When document from client exists
        if file_name and file_url:
            try:
                # S3에서 파일 다운로드 (file_name이 S3 key라고 가정)
                s3_local_path = f"/tmp/{file_name}" 
                s3.download_file(s3_bucket_name, file_name, s3_local_path)
                doc_path = s3_local_path
                print(f"S3에서 파일 다운로드 성공: {doc_path}")

                # Upload PDF to vector db
                document_db = document_to_vector_db(doc_path, embeddings)
                print("document converted to vector database")
            except NoCredentialsError:
                print("S3 인증 정보가 잘못되었습니다.")
                return jsonify({"error": "AWS credentials error"}), 500
            except Exception as e:
                print(f"S3 파일 다운로드 중 오류 발생: {e}")
                return jsonify({"error": "Failed to download file from S3", "details": str(e)}), 500

        # 워크플로우 실행
        result = run_workflow(query, doc_path, clova_api_key, document_db, supporting_db, llm, agent_components, chat_history)
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
        
        # 텍스트 데이터 벡터화
        query_vec = vectorizer.transform([query])
        # 모델을 사용하여 예측
        prediction = model.predict(query_vec)
        # 분석 결과를 클라이언트로 반환 (Java로)
        analysis_result = {
            "document":{
                "originalFileName": file_name,
                "fileUrl": file_url
            },
            "content": query,
            "result": answer,
            "label" : int(prediction[0])
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
