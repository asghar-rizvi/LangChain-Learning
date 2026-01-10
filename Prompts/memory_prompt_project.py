from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

endpoint = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Llama-3.1-8B-Instruct',
    task= 'text-generation'
)

model = ChatHuggingFace(llm=endpoint)


chat_history = [
    SystemMessage(content="You are a helpful Assistant")
]

while True:
    user_input = input("Enter Your message: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == "exit":
        break
    result = model.invoke(chat_history)
    print('AI: ', result.content)
    chat_history.append(AIMessage(content=result.content))

print('Chat history\n\n',chat_history)