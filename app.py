from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from graph.workflow import run_workflow
from utils.tools import extract_text
from vector_db.build_vector_db import build_document_db, load_public_db
from flask_cors import CORS
import boto3
from botocore.exceptions import NoCredentialsError
import joblib

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# --- Configuration and Initialization ---
def load_environment_variables():
    """Load environment variables from .env file."""
    load_dotenv()
    config = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "aws_access_key_id": os.getenv('AWS_ACCESS_KEY_ID'),
        "aws_secret_access_key": os.getenv('AWS_SECRET_ACCESS_KEY'),
        "aws_region": os.getenv('AWS_REGION'),
        "s3_bucket_name": os.getenv('S3_BUCKET_NAME')
    }
    if not config["openai_api_key"]:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    return config

config = load_environment_variables()

# --- LLM and Embeddings ---
llm = ChatOpenAI(temperature=0.5, model='gpt-4o-mini', api_key=config["openai_api_key"])
embeddings = OpenAIEmbeddings(api_key=config["openai_api_key"])

# --- Vector DB ---
supporting_db = load_public_db(embeddings)
if supporting_db is None:
    print("Supporting DB could not be loaded. Please check the configuration.")
    # Exit or handle gracefully
    exit()


# --- Classifier ---
try:
    model = joblib.load('classifier/naive_bayes_model.joblib')
    vectorizer = joblib.load('classifier/vectorizer.joblib')
except FileNotFoundError:
    print("Classifier model or vectorizer not found. Please ensure the files exist.")
    model, vectorizer = None, None


# --- S3 Client ---
s3_client = boto3.client(
    's3',
    aws_access_key_id=config["aws_access_key_id"],
    aws_secret_access_key=config["aws_secret_access_key"],
    region_name=config["aws_region"]
)

def download_from_s3(bucket_name, file_key, local_path):
    """Download a file from S3."""
    try:
        s3_client.download_file(bucket_name, file_key, local_path)
        print(f"Successfully downloaded {file_key} from S3 to {local_path}")
        return local_path
    except NoCredentialsError:
        print("S3 credentials not available.")
        raise
    except Exception as e:
        print(f"Error downloading file from S3: {e}")
        raise

@app.route('/', methods=['POST'])
def process_request():
    """Main endpoint to handle user queries."""
    try:
        data = request.get_json()
        if not data or 'content' not in data:
            return jsonify({"error": "Invalid request: 'content' is required"}), 400

        query = data['content']
        document_info = data.get('document')
        print(f"Received query: {query}")

        document_db = None
        file_name = None
        if document_info and document_info.get('originalFileName') and document_info.get('fileUrl'):
            file_name = document_info['originalFileName']
            s3_key = file_name # Assuming the file name is the key in S3
            local_path = f"/tmp/{s3_key}"
            
            try:
                download_from_s3(config["s3_bucket_name"], s3_key, local_path)
                document_db = build_document_db(local_path, embeddings)
                if document_db:
                    print("Successfully created vector database from the document.")
            except Exception as e:
                return jsonify({"error": "Failed to process document from S3", "details": str(e)}), 500

        # Each request has its own chat history, making it stateless.
        chat_history = []

        # Execute the workflow
        result = run_workflow(query, document_db, supporting_db, llm, chat_history)
        answer = result.get("generated_answer", "Sorry, I couldn't generate an answer.")
        print(f"Final Answer: {answer}")

        # Classify the query
        label = -1 # Default label
        if model and vectorizer:
            try:
                query_vec = vectorizer.transform([query])
                prediction = model.predict(query_vec)
                label = int(prediction[0])
            except Exception as e:
                print(f"Error during query classification: {e}")


        response_data = {
            "document": document_info,
            "content": query,
            "result": answer,
            "label": label
        }

        return jsonify(response_data), 200

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)