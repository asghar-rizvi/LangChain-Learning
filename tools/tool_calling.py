from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=512,
)
chat_model = ChatHuggingFace(llm=llm)

@tool
def multiply_number(a:int, b:int) -> int:
    """Multiply 2 numbers and give their answer"""
    return a*b

model_with_tools = chat_model.bind_tools([multiply_number])

query = "Can you multiple 3 and 9"
messages = [HumanMessage(query)]

ai_message = model_with_tools.invoke(query)
messages.append(ai_message)

tool_result = multiply_number.invoke(ai_message.tool_calls[0])
messages.append(tool_result)


final_result = model_with_tools.invoke(messages)

print(final_result.content)