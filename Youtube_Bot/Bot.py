from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from vector_store import create_vector_store
from dotenv import load_dotenv
load_dotenv()

def make_model():
    llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    max_new_tokens=1024,
    do_sample=False,
    )
    
    model = ChatHuggingFace(llm=llm)
    
    return model

def make_prompt():
    prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
    )
    
    return prompt

def format_document(fetched_content):
    return " ".join(doc.page_content for doc in fetched_content)
if __name__ == '__main__':
    model = make_model()
    video_id = "0jspaMLxBig"

    ###             without using chains        #####
    # retriever = create_vector_store(video_id)
    # question = input("Enter a query for this video : ")
    # retrieved_docs = retriever.invoke(question)
    # context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # prompt = make_prompt()
    # final_prompt = prompt.invoke({"context": context_text, "question": question})
    
    # result = model.invoke(final_prompt)
    
    # print(result.content)
    
    ###            With Using Chains        ######
    
    retriever = create_vector_store(video_id)
    parser = StrOutputParser()
    
    question = input("Enter a query for this video: ")
        
    prompt = make_prompt()
    
    parallel_chain = RunnableParallel({
        "context" : retriever | RunnableLambda(format_document),
        "question" : RunnablePassthrough()
    })
    
    full_chain = parallel_chain | prompt | model | parser
    
    result = full_chain.invoke(question)
    
    print('Result: ',result)