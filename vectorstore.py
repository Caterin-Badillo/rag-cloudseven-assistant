# Módulo responsable de generar representaciones vectoriales (embeddings) a partir de los fragmentos de texto y construir el índice de búsqueda rápida utilizando FAISS 100% local en memoria, sin depender de APIs pagas ni consumir créditos externos.
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

def create_local_vectorstore(splits: List[Document], api_key: str = None) -> FAISS:
    # Genera embeddings locales totalmente gratuitos en la máquina utilizando HuggingFace, transformando los fragmentos de texto en vectores numéricos y almacenándolos en un índice FAISS en memoria RAM.
    # Args:
        # splits (List[Document]): Lista de fragmentos de documentos segmentados (Chunks) listos para indexación.
        # api_key (str, optional): Parámetro opcional por compatibilidad (no requerido al trabajar con modelos locales).
    # Returns:
        # FAISS: Base de datos vectorial indexada que actúa como retriever para la cadena RAG.

    # Inicializar el modelo de embeddings local y gratuito desde HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Construir y poblar el índice vectorial FAISS en memoria a partir de los fragmentos de texto
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore