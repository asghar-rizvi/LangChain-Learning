from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()


def load_model():
    llm = HuggingFaceEndpoint(
        repo_id = "Qwen/Qwen2.5-7B-Instruct",
        task = "text-generation"
    )
    model = ChatHuggingFace(llm=llm)
    return model


if __name__ == '__main__':
    prompt = PromptTemplate(
        template = 'Return name and age in json format. From this detail {detail}',
        input_variables=['detail']
    )
    model = load_model()
    
    parser = JsonOutputParser()
    
    chain = prompt | model | parser
    result = chain.invoke({"detail": "My name is Asghar and my age is 21 years old."})
    print(result)
    print(type(result))