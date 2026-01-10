from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-mpnet-base-v2',
    model_kwargs={"device":"cpu"}
)

docs = [
    Document(page_content="The solar system has eight planets orbiting the sun.", metadata={"topic": "space"}),
    Document(page_content="Mars is known as the Red Planet and has a thin atmosphere.", metadata={"topic": "space"}),
    Document(page_content="Jupiter is the largest planet in our solar system.", metadata={"topic": "space"}),
    Document(page_content="Pasta is a staple food of traditional Italian cuisine.", metadata={"topic": "food"}),
    Document(page_content="Pizza is a popular Italian dish consisting of a flat base of dough.", metadata={"topic": "food"}),
    Document(page_content="The Great Wall of China is a series of fortifications.", metadata={"topic": "history"}),
]

vector_store = Chroma.from_documents(docs, embeddings)

# Normal similarity search with vector embeddings
# retriever = vector_store.as_retriever(search_kwargs={"k":2})


# MMR strategy, fetching diverse documents
retriver = vector_store.as_retriever(
    search_type= "mmr",   
    search_kwargs={"k":2, "lambda_mult":0.1})


results= retriver.invoke("Who is the largest planet on earth?")

# printing each result
for i, doc in enumerate(results):
    print('Result: ',i+1)
    print(doc.page_content)