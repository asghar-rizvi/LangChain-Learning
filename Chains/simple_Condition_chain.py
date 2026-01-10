from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
import datetime
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

def add_metadata(input_data):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"Processed at {timestamp}\nRESPONSE: {input_data}"

if __name__ == '__main__':
    
    user_email = "My credit card was charged twice for the same subscription!"
    
    model = load_model()
    
    # Step 1: Classify first problem
    prompt = PromptTemplate(
        template = "lassify this email as 'TECHNICAL', 'BILLING', or 'GENERAL'. Return only the word. Email: {email}"
    )
    
    # Step 2: Define String Parser for better readibility
    parser = StrOutputParser()

    classifier_chain = prompt | model | parser


    # Step3: Decides which action to take
    tech_chain = PromptTemplate.from_template("TECH SUPPORT: Troubleshoot this: {email}") | model | parser
    billing_chain = PromptTemplate.from_template("BILLING: Explain this charge: {email}") | model | parser
    general_chain = PromptTemplate.from_template("ASSISTANT: Help with this: {email}") | model | parser
    
    # Step 4: Adding meta data information for testing purpose
    metadata_lambda = RunnableLambda(add_metadata)
    
    # Step5: The condition/Smart Branch
    branch = RunnableBranch(
        (lambda x: "TECHNICAL" in x["topic"].upper(), tech_chain),
        (lambda x: "BILLING" in x["topic"].upper(), billing_chain),
        (lambda x: "GENERAL" in x["topic"].upper(), general_chain),
        general_chain
    )
    
    # Step6: Define Full Chain
    
    full_chain = (
        {
        "topic": classifier_chain, 
        "email" : RunnablePassthrough()
        }| branch | metadata_lambda
    )
    
    full_chain.get_graph().print_ascii()
    
    # result = full_chain.invoke({"email": user_email})
    
    # print('Result: ', result)