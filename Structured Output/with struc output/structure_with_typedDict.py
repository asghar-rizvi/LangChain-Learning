from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict, Annotated, Literal, Optional
from dotenv import load_dotenv

load_dotenv()

class Review(TypedDict):
    summary: Annotated[str, "A one line summary of the review"]
    sentiment : Annotated[Literal["pos", "neg", "neutral"], "Return sentiment of the review, either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]], "All the pros in the review"]
    cons: Annotated[Optional[list[str]], "All the cons in the review"]
    name: Annotated[Optional[str], "Name of the reviewer"]
    product_name : Annotated[str, "Name of the product"]
    
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