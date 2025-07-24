from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from utils.tools import extract_text
import os

# --- Constants ---
FAISS_INDEX_DIR = "faiss_index"
SUPPORT_PROGRAM_INDEX = os.path.join(FAISS_INDEX_DIR, "support_programs")
FINANCIAL_PRODUCT_INDEX = os.path.join(FAISS_INDEX_DIR, "financial_products")

# --- Private Functions ---
def _build_and_save_db_from_csv(csv_path, index_path, embeddings, encoding='cp949'):
    """Builds and saves a FAISS database from a CSV file."""
    try:
        loader = CSVLoader(file_path=csv_path, encoding=encoding)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=40)
        texts = text_splitter.split_documents(data)
        db = FAISS.from_documents(texts, embeddings)
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        db.save_local(index_path)
        print(f"Successfully built and saved DB to {index_path}")
        return db
    except Exception as e:
        print(f"Error building DB from {csv_path}: {e}")
        return None

# --- Public API ---
def load_public_db(embeddings, db_type='support'):
    """Loads a pre-built public FAISS database (support programs or financial products)."""
    if db_type == 'support':
        index_path = SUPPORT_PROGRAM_INDEX
        csv_path = './test_data/중소벤처기업부_중소기업지원사업목록_20250331.csv'
    elif db_type == 'financial':
        index_path = FINANCIAL_PRODUCT_INDEX
        csv_path = './test_data/금융상품목록_예시.csv' 
    else:
        raise ValueError("Invalid db_type specified. Choose 'support' or 'financial'.")

    if os.path.exists(index_path):
        print(f"Loading existing FAISS DB from {index_path}")
        try:
            return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Error loading DB from {index_path}: {e}")
            # If loading fails, try rebuilding from source
            print("Attempting to rebuild the database...")
    
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file not found at {csv_path}. Cannot build the database.")
        if db_type == 'financial':
            print("A placeholder for the financial products CSV is expected. Please create it.")
        return None

    print(f"Building new FAISS DB for {db_type}...")
    return _build_and_save_db_from_csv(csv_path, index_path, embeddings)

def build_document_db(file_path, embeddings):
    """Creates a FAISS vector database from a user-provided document."""
    try:
        print(f"Processing document: {file_path}")
        text = extract_text(file_path)
        if not text:
            print("No text could be extracted from the document.")
            return None

        # Normalize and clean text
        text = text.replace('\n', ' ').replace('\r', '')
        text = ' '.join(text.split())
        text = text.encode('utf-8', errors='ignore').decode('utf-8')

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        texts = text_splitter.split_text(text)
        
        vector_db = FAISS.from_texts(texts, embeddings)
        print("Successfully created in-memory vector database for the document.")
        return vector_db
    except Exception as e:
        print(f"Error processing document {file_path}: {e}")
        return None