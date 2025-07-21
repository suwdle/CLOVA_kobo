from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from graph.workflow import run_workflow
from vector_db.build_vector_db import load_public_db, build_document_db
from utils.tools import extract_structured_data

def main():
    """Main function to run the chatbot from the command line."""
    # --- Load Environment and Configuration ---
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # --- Initialize Models and Databases ---
    llm = ChatOpenAI(temperature=0.7, model='gpt-4o-mini', api_key=openai_api_key)
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)

    print("Loading databases...")
    support_db = load_public_db(embeddings, db_type='support')
    # financial_db is loaded lazily within the workflow if needed

    # --- Optional: Load a user document for testing ---
    document_db = None
    # Example: Uncomment the following lines to test with a local document
    # doc_path = './test_data/your_document.pdf' # <--- CHANGE THIS PATH
    # if os.path.exists(doc_path):
    #     print(f"Building vector DB for document: {doc_path}")
    #     document_db = build_document_db(doc_path, embeddings)
    # else:
    #     print(f"Document path not found: {doc_path}")

    # --- Main Chat Loop ---
    chat_history = []
    print("\nChatbot is ready. Type 'exit' to end the conversation.")

    while True:
        query = input("\nUser: ")
        if query.lower() == 'exit':
            break

        # Run the workflow
        result = run_workflow(query, document_db, support_db, llm, embeddings, chat_history)
        final_answer = result.get("generated_answer", "Sorry, I encountered an error.")
        
        print(f"AI: {final_answer}")

        # Extract structured data from the answer for potential downstream use
        extracted_info = extract_structured_data(final_answer, llm)
        if extracted_info:
            print(f"Extracted Information: {extracted_info}")

        # Update chat history
        chat_history.append(("human", query))
        chat_history.append(("ai", final_answer))

if __name__ == "__main__":
    main()