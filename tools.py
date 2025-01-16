import re
from typing import List, Dict
import json
from langchain.schema import Document

def extract_content(text):
    # Use regular expression to find content between [[ and ]]
    pattern = r'\[\[(.*?)\]\]'
    result = re.findall(pattern, text)
    return str(result)


def process_json_to_documents(data: List[Dict]) -> List[Dict]:
    if isinstance(data, bytes):
        data = data.decode('utf-8')
    
    if isinstance(data, str):
        data = json.loads(data)
    
    documents = []
    for item in data:
        # JSON 구조에 따라 적절히 수정
        text = f"{item[1]}"
        doc = Document(page_content=text, metadata={})
        documents.append(doc)
    return documents