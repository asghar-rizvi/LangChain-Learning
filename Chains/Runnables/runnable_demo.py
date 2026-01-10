from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import List
import datetime

load_dotenv()

class TechnicalSpecs(BaseModel):
    product_name: str = Field(description="Name of the product")
    key_features: List[str] = Field(description="List of top 3 technical features")
    target_audience: str = Field(description="Who is this product for?")
    
def load_model():
    llm = HuggingFaceEndpoint(
        repo_id = "Qwen/Qwen2.5-7B-Instruct",
        task = "text-generation",
        temperature=0.1
    )
    model = ChatHuggingFace(llm=llm)
    return model

def get_extraction_model(model):
    parser = PydanticOutputParser(pydantic_object=TechnicalSpecs)
    prompt = ChatPromptTemplate.from_template(
        "Extract technical details from this description.\n{format_instructions}\nDescription: {input}"
    ).partial(format_instructions=parser.get_format_instructions())
    
    return prompt | model | parser

def get_swot_chain(model):
    parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_template(
        "Provide a brief SWOT analysis (Strengths, Weaknesses, Opportunities, Threats) for the following product in bullet points:\n{input}"
    )
    return prompt | model | parser

def get_marketing_chain(model):
    parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_template(
        'Using these Technical Specs: {specs}\nAnd this SWOT Analysis: {swot}\nWrite a 2-paragraph persuasive marketing pitch for this product.'
    )
    return prompt | model | parser

def format_metadata(chain_output):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"--- ANALYSIS REPORT (Generated: {timestamp}) ---\n\n{chain_output}"

def build_master_chain():
    model = load_model()
    
    parallel_analysis = RunnableParallel({
        "specs": get_extraction_model(model),
        "swot": get_swot_chain(model)
    })
    
    full_chain = (
        parallel_analysis | get_marketing_chain(model) | RunnableLambda(format_metadata)
    )

    return full_chain

if __name__ == '__main__':
    product_description = """
    The 'SolarCharge Pro' is a foldable 50W solar panel designed for backpackers. 
    It features dual USB-C ports, weighs only 1.2kg, and has an integrated 
    10,000mAh battery. While it charges fast in direct sunlight, it is 
    not fully waterproof and costs $199, which is higher than basic panels.
    """
    
    master_chain = build_master_chain()
    
    try:
        final_report = master_chain.invoke({"input": product_description})
        print(final_report)
    except Exception as e:
        print(f"Chain Failed due to: {e}")