from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts import PromptTemplate

class CrawlingKeyWord(BaseModel): # 추후 백엔드와 json 파일 형식 맞춰야 함
    query: str = Field(description="name of supporting program")
    Keyword: str = Field(description="link of supporting program")

def ExtractKeyWord(response, model):
    output_parser = JsonOutputParser(pydantic_object=CrawlingKeyWord)

    format_instructions = output_parser.get_format_instructions()
    prompt = PromptTemplate(
        template='''\n{format_instructions}\n query:{query}\n
                    You should extract keywords from the query.
                    You should extract one keyword at least. 
                ''',
        input_variables=["query"],
        partial_variables={"format_instructions":format_instructions}
    )
    input = response
    chain = prompt | model.bind(temperature=0.2) | output_parser
    return chain.invoke({"query":input})