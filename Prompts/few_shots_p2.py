from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from dotenv import load_dotenv
load_dotenv()

# few shot examples that will be send to model
examples = [
    {"input": "I love this product", "output": "Label: HAPPY"},
    {"input": "This is the worst service ever", "output": "Label: ANGRY"},
]

#Define model
llm = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Llama-3.1-8B-Instruct',
    task= 'text-generation'
)

model = ChatHuggingFace(llm=llm)

# Aliging examples with few shot exmaple class
example_prompt = ChatPromptTemplate.from_messages([
    ("system","{input}"),
    ("ai", "{output}")
])

few_shots_prompt = FewShotChatMessagePromptTemplate(
    example_prompt = example_prompt,
    examples=examples
)

# Prepare full prompt
full_prompt = ChatPromptTemplate.from_messages([
    ('system', "You are a professional assistant who will be do sentiment analysis and give output label in CAPS."),
    few_shots_prompt,
    ('human','{input}')
])

# Chaining with prompt that is few shot prompting and going towards model
chain = full_prompt | model


# Taking input from user
user_query = input("Enter a sentiment u want to classify: ")

result = chain.invoke({"input":user_query})

print('result: ',result.content)