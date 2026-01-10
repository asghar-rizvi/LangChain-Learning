from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path = "law_dataset_books",
    glob ="*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True,
    use_multithreading=True,
    silent_errors=True
)

docs = loader.load()

print(docs[60].page_content)
print(docs[60].metadata)