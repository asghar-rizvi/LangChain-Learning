from typing import List
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import re
from dotenv import load_dotenv

load_dotenv()

class Poem_Explaination(BaseModel):
    keypoints: List[str] = Field(description="List of specific cricket terms used in the poem")
    explaination: str = Field(description="Explaination of the poem")

def load_model():
    llm = HuggingFaceEndpoint(
        repo_id = "Qwen/Qwen2.5-7B-Instruct",
        task = "text-generation",
        temperature=0.1
    )
    model = ChatHuggingFace(llm=llm)
    return model

def load_text(path):
    loader = TextLoader('text_loader_example.txt')
    
    docs = loader.load()
    
    text = docs[0].page_content
    return " ".join(text.split())

def clean_data(text : str) ->str:
    text = re.sub(r'[^\w\s]', '', text)
    text = " ".join(text.split())
    return text.lower()

def summary_chain():
    model = load_model()
    preprocessing = RunnableLambda(lambda x : clean_data(x))
    
    parser = PydanticOutputParser(pydantic_object=Poem_Explaination)
    
    prompt = ChatPromptTemplate.from_template(
        "Your task is to process the poem written and find keywords used in the poem related to the topic of the poem, and also generate explaination of the poem\n{format_instructions}.\nPoem: {poem}"
        ).partial(format_instructions=parser.get_format_instructions())
    
    chain = {"poem" : preprocessing} | prompt | model | parser
    
    return chain

if __name__ == '__main__':
    text = load_text("text_loader_example.txt")
    try:
        chain = summary_chain()
        result = chain.invoke(text)
        print(result)
    except Exception as e:
        print('Stopped due to following exception: ',e)
        