from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os 

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    max_new_tokens=1024,
    do_sample=False,
)
    
llm_engine = ChatHuggingFace(llm=llm)

result = llm_engine.invoke("What is the capital of Pakistan")

print(result.content)