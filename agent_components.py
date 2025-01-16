from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.prompts import MessagesPlaceholder
from datetime import datetime


def initialize_agent_components(llm):
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    # Prompt 템플릿을 한 번만 초기화
    base_prompt = ChatPromptTemplate.from_messages([
        ("system", f'''You are an expert AI assistant specializing in providing financial information to small businesses. Your role is to accurately retrieve and generate information that meets the user's queries.
        ### Instructions:
        1. Always respond in Korean.
        2. Ensure that your answers are correct and based on the given information.
        3. Choose the appropriate method for information retrieval:
        - **유저 파일**: Use this when the user inquires about their company or personal files.
        - **데이터베이스**: Use this for questions regarding SME (Small and Medium Enterprises) support programs.
        - **검색엔진**: Use this to obtain real-time information or data that is unlikely to be found in the first two methods.
        4. Analyze user input and clearly describe the data you need to obtain.
        5. For efficient searching, format your search content with [[ at the beginning and ]] at the end. Ensure to provide only one suitable keyword when utilizing the search engine, as it must be included.
        6. Keep your responses concise and suitable for retrieval use, ensuring all communication is in Korean.
        7. Current time: {current_time}'''),
        ("human", '''지금까지의 채팅 내역입니다. 사용자가 이전 질문에 대해 추가적인 정보를 원할 때는 참고하세요. Chat History: {chat_history},
                        현재 질문: {input}'''),
        ("ai", "I understand. I'll determine the best course of action."),
        ("human", "Great, what do you think we should do next?")
    ])
    
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", '''
        You are an expert AI assistant specializing in enhancing user inquiries for AI agents to ensure responses are precisely tailored and highly accurate. 

        ### Instructions: 
        1. Rewrite the user's question using subtlety and clarity based on the provided context.
        2. Create a descriptive analysis of the data the AI agent is required to gather in order to formulate its answer.

        ### Context & Query: 
        - Context: {context}
        - Original Question: {input}
        - AI Previously Provided Answer: {answer}
        '''),
        ("human", "Please provide the necessary context and data to improve the AI's response.")
    ])
    
    generate_prompt = ChatPromptTemplate.from_messages([
        ("system", '''You are an AI assistant that generates the answer based on given context. 
                      Please write in Korean. Answer logically and avoid writing false information'''),
        ("human", '''Using the following information, generate a comprehensive response on the question.
                    When you refer to news, please indicate source link.
                    구체적인 수치를 인용하며 서술해라. 지원 사업 데이터에 링크가 있다면 정확히 참조해줘.
                    기록번호나 날짜는 제외해서 서술해도 돼.
                information:{context}, question: {query}, agent_response : {agent_response}''')
    ])

    grade_prompt = PromptTemplate.from_template("""
        As an AI evaluation expert, your task is to assess if an AI-generated response meaningfully answers a human question.

        ### Instructions:
        1. Review the question and agent response carefully
        2. Evaluate if the response provides useful, relevant information that addresses the core of the question
        3. Respond with "yes" if:
        - The response contains information that helps answer the question
        - The response is on-topic and related to what was asked
        - The response makes a genuine attempt to be helpful
        
        4. Respond with "no" only if:
        - The response is completely unrelated to the question
        - The response contains demonstrably false information
        - The response is harmful or inappropriate

        ### Context:
        - Question: {question}
        - Agent Response: {agent_response}
        - Answer: {answer}

        Provide only "yes" or "no" as your response.
    """)

    
    # LLM 바인딩을 미리 한 번만 실행
    base_chain = base_prompt | llm.bind(temperature=0.4)
    rewrite_chain = rewrite_prompt | llm.bind(temperature=0.3)
    generate_chain = generate_prompt | llm.bind(temperature=0.6)
    grade_chain = grade_prompt | llm.bind(temperature = 0.3)
    return {
        "base_chain": base_chain,
        "rewrite_chain": rewrite_chain,
        "generate_chain": generate_chain,
        "grade_chain": grade_chain
    }