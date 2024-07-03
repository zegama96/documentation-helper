import os
from langchain.document_loaders import (
    ReadTheDocsLoader,
    TextLoader,
    UnstructuredHTMLLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
from config import INDEX_NAME

load_dotenv()
pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)


def ingest_docs():
    loader = ReadTheDocsLoader(
        r"C:/Users/Teresa/Documents/documentation-helper/langchain-docs/langchain.readthedocs.io/en/latest",
        encoding="utf-8",
    )
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(raw_documents)
    print(f"Splited into {len(documents)} chunks")

    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.split("langchain-docs")[1]
        new_url = new_url.replace("\\", "/")
        new_url = "https:/" + new_url
        doc.metadata.update({"source": new_url})

    print(f"Going to insert {len(documents)} documents to Pinecone")
    return documents


if __name__ == "__main__":

    documents = ingest_docs()

    embedding = OpenAIEmbeddings()
    vectorstore = Pinecone.from_documents(
        documents=documents, embedding=embedding, index_name=INDEX_NAME
    )
