from langchain_core.prompts import load_prompt, PromptTemplate

# prompt = PromptTemplate.from_template('you are a {role}. Tell me about this topic {topic}')

# saving this prompt using this, save only once and loading it multiple times if want
# prompt.save("my_prompt.json")

# loaded_prompt = load_prompt('my_prompt.json')

# print(loaded_prompt.format(role ='Ai Engineer', topic='Neural Networks'))


#we can also do this saving prompt thing in chat models, with different roles

from langchain_core.prompts import load_prompt, ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful assistant for this role {role}'),
    ('human', 'Can u give me a 3 bulleted points answer only')
])

# Serialize to a JSON string
# This captures EVERYTHING (roles, variables, types)

# from langchain_core.load import dumps
# serialized_prompt = dumps(prompt)

# with open("chat_template.json", "w") as f:
#     f.write(serialized_prompt)

from langchain_core.load import loads

# 1. Read the file
with open("chat_template.json", "r") as f:
    data = f.read()

# 2. Reconstruct the object
# LangChain automatically knows this is a ChatPromptTemplate!
loaded_prompt = loads(data)

# 3. Use it exactly like before
print(loaded_prompt.format(role="Senior Developer"))