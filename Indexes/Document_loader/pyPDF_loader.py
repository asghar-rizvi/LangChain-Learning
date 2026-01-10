from typing import List
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import re
from dotenv import load_dotenv

load_dotenv()

class PageSummary(BaseModel):
    main_topic: str = Field(description="The primary subject of this page")
    key_points: List[str] = Field(description="List of 3 important facts from this page")

def load_model():
    llm = HuggingFaceEndpoint(
        repo_id = "Qwen/Qwen2.5-7B-Instruct",
        task = "text-generation",
        temperature=0.1
    )
    model = ChatHuggingFace(llm=llm)
    return model

def load_pdf(path,page_number):
    loader = PyPDFLoader(path)
    docs= loader.load()
    return docs[page_number].page_content
    
def clean_data(text : str) ->str:
    text = re.sub(r'[^\w\s]', '', text)
    text = " ".join(text.split())
    return text.lower()

def pdf_chain(model):
    parser = PydanticOutputParser(pydantic_object=PageSummary)
    preprocessing = RunnableLambda(lambda x:clean_data(x))
    
    prompt = ChatPromptTemplate.from_template(
        "You are a research assistant. Analyze the following page text.\n"
        "{format_instructions}\n"
        "Page Content: {content}"
    ).partial(format_instructions=parser.get_format_instructions())
    
    chain = {"content":preprocessing} | prompt | model | parser
    return chain

if __name__ == '__main__':
    model = load_model()
    # print(load_pdf('pdc_2.pdf',1))
    text = load_pdf('pdc_2.pdf',1)
    chain = pdf_chain(model)
    result = chain.invoke(text)
    print('result: ',result)