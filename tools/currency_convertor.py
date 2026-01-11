from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage
from langchain_core.tools import InjectedToolArg
from typing import Annotated
import requests
import json
from dotenv import load_dotenv

load_dotenv()

@tool
def get_conversion_factor(base_cur: str, target_cur: str) -> float:
    """Give the currency conversion factor for base currency and target currency"""
    url = f"https://v6.exchangerate-api.com/v6/{TOKEN}/pair/{base_cur}/{target_cur}"
    response = requests.get(url)
    
    return response.json()

@tool
def convert_currency(base_cur_val:float, converion_rate: Annotated[float, InjectedToolArg]) ->float:
    """"This function will calculate the converted currency from one currencty to another using conversion rate"""
    return base_cur_val * converion_rate


# print(get_conversion_factor.invoke({"base_cur":"USD","target_cur":"PKR"}))
# print(convert_currency.invoke({"base_cur_val":100, "converion_rate":282.0058}))

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=512,
)
chat_model = ChatHuggingFace(llm=llm)

llm_with_tools = chat_model.bind_tools([get_conversion_factor, convert_currency])

query = "What is the conversion factor between USD and PKR, and using that can you convert 150 USD into PKR"

messages = [HumanMessage(query)]

ai_message = llm_with_tools.invoke(messages)
messages.append(ai_message)
# print(ai_message.tool_calls)

for tool_call in ai_message.tool_calls:
    if tool_call["name"] == "get_conversion_factor":
        toolMessage1 = get_conversion_factor.invoke(tool_call)
        conversion_rate = json.loads(toolMessage1.content)['conversion_rate']
        messages.append(toolMessage1)
        
    elif tool_call["name"] == "convert_currency":
        tool_call["args"]["converion_rate"] = conversion_rate
        toolMessage2 = convert_currency.invoke(tool_call)
        messages.append(toolMessage2)
        
final_result = llm_with_tools.invoke(messages)
print(final_result.content)