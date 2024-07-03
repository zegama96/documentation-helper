import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
from config import INDEX_NAME
from typing import Any, Tuple, List, Dict

load_dotenv()

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []) -> Any:
    embedding = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embedding)
    chat = ChatOpenAI(verbose=True, temperature=0)
    # return source documents -> returns vectors that answer query
    # legacy
    # qa = RetrievalQA.from_chain_type(
    #    llm=chat,
    #    chain_type="stuff",
    #    retriever=docsearch.as_retriever(),
    #    return_source_documents=True
    # )

    qa1 = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    )

    return qa1({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    print(run_llm(query="What is the RetrievalQA chain?"))
