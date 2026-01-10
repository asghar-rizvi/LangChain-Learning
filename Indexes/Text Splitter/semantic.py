from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

semantic_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_amount=90,
    breakpoint_threshold_type="percentile"
)

raw_text = """
The Indian cricket team won the series after a stunning performance by the middle order batsmen. 
The crowd at the stadium was electrifying and cheered for every boundary. 
Meanwhile, in the capital city, the government has announced new tax reforms for the upcoming fiscal year. 
These changes aim to reduce the burden on middle-class families. 
Looking at the skies, a low-pressure system is developing over the coast. 
Meteorologists predict heavy rainfall and thunderstorms for the next forty-eight hours.
"""

docs = semantic_splitter.create_documents([raw_text])

for i, doc in enumerate(docs):
    print(f"--- CHUNK {i+1} ---")
    print(doc.page_content.strip())
    print("-" * 30)