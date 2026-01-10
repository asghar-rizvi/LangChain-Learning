from transcript_extractor import extract_transcript_from_video_id
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_texts(doc):
    splitter =RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    chunks = splitter.create_documents([doc])
    return chunks

def create_vector_store(video_id):
    transcript = extract_transcript_from_video_id(video_id)
    chunks = split_texts(transcript)
    embeddings = HuggingFaceEmbeddings(
        model_name = 'sentence-transformers/all-mpnet-base-v2',
        model_kwargs={"device":"cpu"}
    )
    vector_store = Chroma.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    return retriever