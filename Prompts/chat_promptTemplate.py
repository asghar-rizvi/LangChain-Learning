from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a professional {role}"),
    ("human", "help me with {question}")
])

formatted_messages = chat_template.format_messages(role="cricketer", question="Who was th youngest cricket player in international world games")

print('Formatted message: ', formatted_messages)

# Output

# Formatted message:  [SystemMessage(content='You are a professional cricketer', additional_kwargs={}, response_metadata={}), HumanMessage(content='help me with Who was th youngest cricket player in international world games', additional_kwargs={}, response_metadata={})]