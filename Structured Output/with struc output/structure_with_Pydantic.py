from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from typing import Optional, Literal
from dotenv import load_dotenv

load_dotenv()

class Review(BaseModel):
    username: Optional[str] = Field(default=None, description="Name of the reviewer")
    product: str = Field(description="Name of the product")
    summary: str = Field(description="one line summary of review")
    sentiment: Literal["pos", "neg", "neutral"] = Field(description="Sentiment of review, either positive, neutral or negative")
    pros: Optional[list[str]] = Field(default=None, description="All the pros that are written")
    cons: Optional[list[str]] = Field(default=None, description="All the cons that are written")
    
    
def load_model():
    llm = HuggingFaceEndpoint(
        repo_id = "Qwen/Qwen2.5-7B-Instruct",
        task = "text-generation"
    )
    model = ChatHuggingFace(llm=llm)
    return model


if __name__ == '__main__': 
    review = """
    User: Asghar
    Product: SoundMax Ultra Wireless Headphones

    I've been using these for two weeks now. The sound quality is crisp and the active noise canceling (ANC) is 
    top-tier for my daily commute. However, they are quite heavy, and the companion app is a bit buggy.

    Pros:
    - Incredible battery life (40+ hours)
    - High-fidelity audio
    - Premium leather ear cups

    Cons:
    - App crashes occasionally
    - Expensive compared to rivals
    - A bit heavy for long flights
    """
    
    model = load_model()

    structured_model = model.with_structured_output(Review)
    
    result = structured_model.invoke(review)
    
    print(result)