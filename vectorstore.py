from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

def create_local_vectorstore(splits: List[Document], api_key: str = None) -> FAISS:
    """
    Genera embeddings locales totalmente gratuitos en tu máquina usando HuggingFace,
    sin consumir crédito ni depender de la API de OpenAI.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore