from typing import List
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

def create_local_vectorstore(splits: List[Document], api_key: str) -> FAISS:
    """Genera embeddings y los almacena en una base de datos vectorial local (FAISS)."""
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore