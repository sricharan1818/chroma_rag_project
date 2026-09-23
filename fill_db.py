from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

loader = PyPDFDirectoryLoader("data")

documents = loader.load()

print(f"Number of documents loaded: {len(documents)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Number of chunks created: {len(chunks)}")

embedding_function = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully.")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_function,
    persist_directory="chroma_db"
)

print("Documents stored in ChromaDB successfully.")