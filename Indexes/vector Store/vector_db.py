import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma

def get_embeddings():
    return HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

def create_store_vector_db(filename, db_directory):
    loader = PyPDFLoader(filename)
    docs = loader.load()
    
    embeddings = get_embeddings()
    
    semantic_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_amount=90,
        breakpoint_threshold_type="percentile"
    )

    chunks = semantic_splitter.split_documents(docs)
    
    
    vector_db = Chroma(
        embedding_function=embeddings,
        persist_directory=db_directory,
        collection_name="example_collection"
    )
    vector_db.add_documents(documents=chunks)
    return vector_db

def chat_with_document(vector_db):
    print('Enter exit to close this')
    
    while True:
        query = input("User: ")
        if query.lower() == "exit":
            break
        
        results = vector_db.similarity_search(query, k=3)

        print("\n---  Top Relevant Sections Found ---")
        for i, doc in enumerate(results):
            page_num = doc.metadata.get("page", "Unknown")
            print(f"\n[Result {i+1} | From Page {page_num}]:")
            print(doc.page_content.strip())
            print("-" * 50)
        print("\n")
        
        
    
if __name__=='__main__':
    FILE_NAME = "Anti-Money-Laundering-Act-2010-amended-upto-Sep. 2020.pdf"
    DB_PATH = "./chroma_db_aml"
    
    if not os.path.exists(FILE_NAME):
        print(f"Error: Could not find '{FILE_NAME}' in this folder.")
    else:
        if os.path.exists(DB_PATH):
            print("Found existing database. Loading...")
            db = Chroma(persist_directory=DB_PATH, embedding_function=get_embeddings())
        else:
            db = create_store_vector_db(FILE_NAME, DB_PATH)
        
        
        chat_with_document(db)   
    