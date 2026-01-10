from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import  StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
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

def joke_chain(model):
    parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_template(
        "Create a joke on this topic {topic}"
    )
    return prompt | model | parser

def explaination_chain(model):
    parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_template(
        "Create an explaination of this joke {joke}"
    )
    return prompt | model | parser

def master_chain():
    model = load_model()
    parallel_chain = RunnableParallel(
        {"explaination":explaination_chain(model), "joke": RunnablePassthrough()}
    )
    
    full_chain = joke_chain(model) | parallel_chain
    
    return full_chain

if __name__ == '__main__':
    final_chain = master_chain()
    
    try:
        result = final_chain.invoke({"topic":"cricket"})
        # print(result)
        print('Explaination: ', result["explaination"])
        print('joke: ', result["joke"])
    except Exception as e:
        print(f'Stopped due to following exception: {e}')