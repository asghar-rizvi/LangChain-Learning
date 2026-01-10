from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint  
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from dotenv import load_dotenv
load_dotenv()


def load_model():
    llm = HuggingFaceEndpoint(
        repo_id = "Qwen/Qwen2.5-7B-Instruct",
        task = "text-generation",
        temperature=0.1
    )
    model = ChatHuggingFace(llm=llm)
    return model

if __name__ == '__main__':
    model = load_model()
    
    project_update = """
            Yesterday we finished the migration to Python 3.12. It was hard because of 
            dependency conflicts in our legacy code. We also reduced our AWS bill by 15% 
            by optimizing our Docker containers and improved API latency by 200ms using Redis.
        """
    
    string_parser = StrOutputParser()
    
    template1 = PromptTemplate(
        template = "Extract the 3 most technical achivements from this information of the user: {project_update}",
        input_variables=[project_update]
    )
    template2 = PromptTemplate(
        template = "Rewrite these achivements as a proper small linkedin post, with a strong hook and emojis.\n Achievments: {achivements}",
    )
    
    chain1 = template1 | model | string_parser
    
    chain2 =  template2 | model | string_parser
    
    full_chain = ( {"achivements":chain1} | chain2 )
    
    # full_chain.get_graph().print_ascii()
    
    
    try:
        result = full_chain.invoke({"project_update":project_update})
        print(result)
    except Exception as e:
        print(f"Parsing Error: {e}")