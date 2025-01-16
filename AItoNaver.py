# 테스트용 코드, 추후 삭제 요망

import os
import urllib.request
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables")

test_prompt = ChatPromptTemplate.from_messages([
    ("system", '''너의 임무는 사용자의 질문으로부터 답변에 필요한 정보를 찾는데 유용한 검색어를 만들어 검색 API에 전달하는거야.
     검색 API가 시간, 날짜 관련은 최신 정렬로 제어하고 있으니, 특정하는 질문이 없는 한 검색어로 넣지마
     검색어 본문만 출력해주면 돼'''),
    ("human", "내 질문:{input}")
])

llm = ChatOpenAI(temperature=0.5, model='gpt-4o', openai_api_key=openai_api_key)

test_chain = test_prompt | llm.bind(temperature=0.5)
client_id = os.getenv("CLIENT_ID")
client_secret= os.getenv("CLIENT_SECRET")
response = test_chain.invoke({"input":'성공회대학교에 대해 검색해줘'})
test_input = str(response.content)
print(test_input)

encText = urllib.parse.quote(test_input)

url = 'https://openapi.naver.com/v1/search/news?query='+ encText
request = urllib.request.Request(url)
request.add_header("X-Naver-Client-id",client_id)
request.add_header("X-Naver-Client-Secret", client_secret)
response = urllib.request.urlopen(request)
rescode = response.getcode()
if(rescode==200):
    response_body = response.read()
    search_response = response_body.decode('utf-8')
    print(response_body.decode('utf-8'))
else:
    print("Error Code:" + rescode)
