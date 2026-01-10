# Deep Implementation with Hugging Face API

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

# Always gonna use hugging face endpoint for chat models
llm = HuggingFaceEndpoint(
    repo_id='HuggingFaceH4/zephyr-7b-beta',
    task= 'text-generation'
)

chat_model = ChatHuggingFace(llm=llm)

# implementing prompts logic now

prompt = ChatPromptTemplate([
    ('system', 'Your are an expert in {topic}. I want you to answer each question in 2 answers'),
    ('human', 'Question: {question}')
])

# Defining llm role
# use partial, in projects best approach
math_prompt = prompt.partial(topic='mathematics') # its better to handle system prompt in partial, even if its from user input

# taking user input as the query
question_input = input('Enter the topic u want to know in maths: ')

# im using chains right now, this is how chains work 
chain = math_prompt | chat_model

# first go towards math_prompt so promting is appplied properly and than take that formatted prompt towards our chat model

result = chain.invoke({"question": question_input})

print('Result: ', result.content)