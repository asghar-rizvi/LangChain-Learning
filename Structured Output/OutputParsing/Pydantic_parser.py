from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint  
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
load_dotenv()

class RealEstateLead(BaseModel):
    name: str = Field(description="The full name of the potential buyer")
    phone: Optional[str] = Field(description="The phone number, if provided")
    budget: int = Field(description="The maximum budget in numbers (no symbols)")
    locations: List[str] = Field(description="List of cities or neighborhoods they are interested in")
    urgency: str = Field(description="Categorize as 'High', 'Medium', or 'Low'")


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
    
    email = """
        Hi, I'm Asghar. I've been looking for a 3-bedroom apartment in Karachi, specifically 
        near Clifton or DHA. My budget is around 25000000. I'm currently staying in a hotel, 
        so I need to move in as soon as possible! You can reach me at 0300-1234567.
    """
    
    parser = PydanticOutputParser(pydantic_object=RealEstateLead)
    
    template = PromptTemplate(
        template = "Extract the buyer information from the email below.\n{format_instructions}\nEmail: {email}",
        input_variables=['email'],
        partial_variables={"format_instructions":parser.get_format_instructions()}
    )
    

    chain = template | model | parser
    
    
    try:
        result = chain.invoke({"email":email})
        print(f"Name: {result.name}")
        print(f"Budget: {result.budget:,} PKR") 
        print(f"Locations: {', '.join(result.locations)}")
        print(f"Urgency Score: {result.urgency}")

    except Exception as e:
        print(f"Parsing Error: {e}")