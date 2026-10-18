import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP

def load_and_split_documents(data_folder: str = "./data") -> List[Document]:
    """Carga archivos PDF y TXT desde una carpeta local y los divide en chunks."""
    documents = []
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        
    for file in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif file.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())
            
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.split_documents(documents)