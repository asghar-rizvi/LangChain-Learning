from langchain_core.prompts import PromptTemplate

template = "Explain this {topic} in detail like i am a {age} years old"

prompt = PromptTemplate.from_template(template)

print(prompt.input_variables)

final_str = prompt.format(topic='Machine Learning', age='21')

print(final_str)

# This is the basic prompt template, just using prompt template thing and not f string, because of several benefits of this over simple
# f strings,


# Output
#             ['age', 'topic']
#             Explain this Machine Learning in detail like i am a 21 years old