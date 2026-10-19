# Módulo encargado de procesar los archivos cargados por el usuario desde la interfaz web, realizar la extracción de texto directamente en memoria (usando pypdf para PDFs y decodificación para TXT) y aplicar la técnica de fragmentación (chunking) con RecursiveCharacterTextSplitter.
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
import pypdf

def process_uploaded_files(uploaded_files) -> List[Document]:
    # Procesa una lista de archivos subidos en Streamlit (PDF o TXT), los almacena temporalmente para lectura y los divide en fragmentos manejables.    
    # Args:
        # uploaded_files: Lista de objetos UploadedFile de Streamlit.
    
    # Returns:
        # list: Lista de objetos Document fragmentados listos para indexación.
    
    documents = []

    for uploaded_file in uploaded_files:
        # Crear un archivo temporal en disco para permitir lectura por las librerías de LangChain
        filename = uploaded_file.name
        
        # Procesamiento de archivos PDF en memoria utilizando pypdf
        if filename.endswith(".pdf"):
            pdf_reader = pypdf.PdfReader(uploaded_file)
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text:
                    # Limpiar espacios en blanco excesivos y saltos de línea extraños
                    cleaned_text = " ".join(text.split())
                    documents.append(
                        Document(
                            page_content=cleaned_text,
                            metadata={"source": filename, "page": page_num + 1}
                        )
                    )
        # Procesamiento de archivos de texto plano (.txt) en memoria
        elif filename.endswith(".txt"):
            content = uploaded_file.getvalue().decode("utf-8")
            if content:
                cleaned_text = " ".join(content.split())
                documents.append(
                    Document(
                        page_content=cleaned_text,
                        metadata={"source": filename}
                    )
                )

    # Si no se logró extraer ningún texto válido, retornar lista vacía
    if not documents:
        return []

    # Aplicar segmentación de texto (Chunking) para optimizar la búsqueda vectorial
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.split_documents(documents)