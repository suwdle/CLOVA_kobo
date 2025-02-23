from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from tools import extract_text

def public_to_vector_db():

    loader = CSVLoader(file_path='./test_data/중소기업지원사업목록_20240331.csv', encoding='cp949')
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


def document_to_vector_db(file_path):
    try:
        print("문서 처리 시작")
        # 파일에서 텍스트 추출
        text = extract_text(file_path)
        
        # 텍스트 정규화
        text = text.replace('\n', ' ').replace('\r', '')
        text = ' '.join(text.split())  # 중복 공백 제거
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
        
        print("문서 처리 완료")
        return vector_db
    except Exception as e:
        print(f"문서 처리 중 오류 발생: {e}")
        return None

    

