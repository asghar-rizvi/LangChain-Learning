import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

def process_pdf_with_semantic(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    semantic_splitter = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type="standard_deviation",
        breakpoint_threshold_amount=1
    )
    
    semantic_chunks = semantic_splitter.split_documents(docs)
    return semantic_chunks

if __name__ == '__main__':
    PDF_PATH = 'pdc_2.pdf'
    
    chunks = process_pdf_with_semantic(PDF_PATH)
   
    for i, chunk in enumerate(chunks):
            print(f"--- CHUNK {i+1} ---")
            print(f"Source: Page {chunk.metadata['page']}")
            print(f"Content Sample: {chunk.page_content[:200]}...")
            print("-" * 40)