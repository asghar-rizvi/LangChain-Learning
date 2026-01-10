from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_document(path):
    loader = PyPDFLoader(path)
    docs = loader.load()
    return docs

def split_document(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size= 100,
        chunk_overlap=20,
        length_function=len,
        add_start_index=True
    )
    chunks = splitter.split_documents(docs)
    return chunks

if __name__ == '__main__':
    docs = load_document('pdc_2.pdf')
    # print(docs[0].page_content)
    
    chunks = split_document(docs)
    print(chunks[0].page_content)
    print(chunks[0].metadata)
    
    print('chunk2')
    print(chunks[1].page_content)
    print(chunks[1].metadata)
    
    
    print(f"\nTotal Chunks created: {len(chunks)}")