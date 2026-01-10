from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    max_new_tokens=1024,
    do_sample=False,
)

docs = [
    # Topic A: Project "Solaris" (Technical/Corporate)
    Document(page_content="Project Solaris is our initiative to install solar panels on all factory roofs by 2027.", metadata={"category": "internal"}),
    Document(page_content="The Solaris program budget was approved at $4.2M for the initial phase.", metadata={"category": "finance"}),
    Document(page_content="Renewable energy transition (Code: Solaris) requires 500kW capacity per site.", metadata={"category": "tech"}),
    
    # Topic B: "Solaris" (The Movie/Sci-Fi Context - The Red Herring)
    Document(page_content="Solaris is a 1972 science fiction film directed by Andrei Tarkovsky about a space station.", metadata={"category": "media"}),
    Document(page_content="The novel Solaris, written by Stanislaw Lem, explores the limits of human communication with aliens.", metadata={"category": "literature"}),
    
    # Topic C: Employee Benefits (Vague wording)
    Document(page_content="Staff members are entitled to 20 days of annual leave.", metadata={"category": "hr"}),
    Document(page_content="Vacation time accrues at a rate of 1.6 days per month for full-time workers.", metadata={"category": "hr"}),
    Document(page_content="Holiday policy: Employees must submit time-off requests 2 weeks in advance.", metadata={"category": "hr"}),
    
    # Topic D: Security (Ambiguous terms)
    Document(page_content="The main gate is secured with a biometric 'Key'.", metadata={"category": "security"}),
    Document(page_content="Encryption 'keys' are rotated every 90 days for database safety.", metadata={"category": "it"}),
    Document(page_content="Access card 'keys' should be reported lost immediately to the security desk.", metadata={"category": "security"}),
    
    # Topic E: Financial Reports (Specific vs Generic)
    Document(page_content="Q3 Earnings report shows a 15% increase in net revenue.", metadata={"category": "finance"}),
    Document(page_content="The fiscal performance update indicates the company is profitable.", metadata={"category": "finance"}),
    Document(page_content="Our monetary gains this quarter exceeded the growth of our competitors.", metadata={"category": "finance"}),
    Document(page_content="Profit and loss statement for the autumn period is now available on the portal.", metadata={"category": "finance"}),
    Document(page_content="Revenue streams from the new product line have stabilized.", metadata={"category": "sales"})
]

embeddings = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-mpnet-base-v2',
    model_kwargs={"device":"cpu"}
)

vector_store = Chroma.from_documents(docs, embeddings)

multi_ret = MultiQueryRetriever.from_llm(
    retriever= vector_store.as_retriever(search_kwargs={"k":3}),
    llm = llm)

results = multi_ret.invoke("How many days can I take off for a trip?")

for i, doc in enumerate(results):
    print(f'result {i+1}')
    print(doc.page_content)