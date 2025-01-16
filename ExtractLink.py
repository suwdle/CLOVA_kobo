
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts import PromptTemplate

class CrawlingLink(BaseModel):
    name: str = Field(description="name of supporting program")
    link: str = Field(description="link of supporting program")

def ExtractLink(response, model):
    output_parser = JsonOutputParser(pydantic_object=CrawlingLink)

    format_instructions = output_parser.get_format_instructions()
    prompt = PromptTemplate(
        template='''Answer the query. \n{format_instructions}\n{query}\n
                    You should extract name and link of the supporting programs from the query.
                    But, don't extract name and link 전자공시시스템 (https://dart.fss.or.kr).
                    추출할 것이 없으면 빈 문자열을 출력해.
                ''',
        input_variables=["query"],
        partial_variables={"format_instructions":format_instructions}
    )
    input = response
    chain = prompt | model.bind(temperature=0.2) | output_parser
    return chain.invoke({"query":input})
