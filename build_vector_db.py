from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

from dotenv import load_dotenv
import os

import os
from dotenv import load_dotenv
from llama_index import SimpleDirectoryReader
from llama_index.llm_predictor.llama_parse import LlamaParse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def document_to_vector_db(file_path):
    try:
        load_dotenv()
        llamaparse_key = os.getenv('LLAMA_PARSE_KEY')

        print("Starting document processing")

        # LlamaParse 설정
        parser = LlamaParse(
            api_key=llamaparse_key,
            result_type="markdown",  # "markdown" 또는 "text" 사용 가능
            num_workers=4,
            verbose=True,
            language="ko"
        )
        file_extractor = {
            ".pdf": parser,
            ".pptx": parser,
            ".docx": parser,
            ".xlsx": parser
        }

        # 파일 파싱
        documents = SimpleDirectoryReader(
            input_files=[file_path],
            file_extractor=file_extractor,
        ).load_data()

        # 텍스트 추출
        text = ""
        for doc in documents:
            text += doc.get_text()

        # 텍스트 정규화
        text = text.replace('\n', ' ').replace('\r', '')
        text = ' '.join(text.split())  # 연속된 공백 제거
        text = text.encode('utf-8', errors='ignore').decode('utf-8')

        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        texts = text_splitter.split_text(text)

        # FAISS 벡터 저장소 생성
        embeddings = OpenAIEmbeddings()
        vector_db = FAISS.from_texts(texts, embeddings)

        print("Document processing completed")
        return vector_db

    except Exception as e:
        print(f"Error in document processing: {e}")
        return None

    
def public_to_vector_db():

    loader = CSVLoader(file_path='app/test_data/중소기업지원사업목록_20240331.csv', encoding='cp949')
    try:
        data = loader.load()
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        raise
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=40)
    texts = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings()
    supporting_db = FAISS.from_documents(texts, embeddings)
    print('...db build complete...')
    return supporting_db
