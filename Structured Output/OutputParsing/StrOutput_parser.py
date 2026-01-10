from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
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
        template = "Write a summary on this topic {topic}",
        input_variables=["topic"]
    )

    prompt2 = PromptTemplate(
        template = "Write a 5 line summary on this topic {topic}",
        input_variables=["topic"]
    )
    
    model = load_model()
    
    parser = StrOutputParser()
    
    chain = prompt | model | prompt2 | model | parser
    
    result = chain.invoke({"topic":"Artifical Intelligence"})
    print(result)

