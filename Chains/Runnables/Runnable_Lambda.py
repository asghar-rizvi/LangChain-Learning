from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import  StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
import re
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

def clean_text(text: str)->str:
    text = re.sub(r'[^\w\s]', '', text)
    text = " ".join(text.split())
    return text.lower()
    


def sentiment_analysis():
    model = load_model()
    parser = StrOutputParser()
    preprocessing = RunnableLambda(lambda x: clean_text(x))
    
    prompt = ChatPromptTemplate.from_template(
        "Analyze the sentiment of the following cleaned review. "
        "Classify as 'POSITIVE', 'NEGATIVE', or 'NEUTRAL'. "
        "Return only the label.\n\nReview: {cleaned_review}"
    )
    
    return {"cleaned_review": preprocessing} | prompt | model | parser    
 

if __name__ == "__main__":
    reviews = [
        " The food was EXCELLENT!!!   ",
        "worst... service... EVER... 1/10",
        "It was okay,   nothing special really. "
    ]
    
    full_chain = sentiment_analysis()
    
    results = full_chain.batch(reviews)
    print(results)