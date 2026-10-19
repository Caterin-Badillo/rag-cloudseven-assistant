import io
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP
import pypdf

def process_uploaded_files(uploaded_files) -> List[Document]:
    """Procesa los archivos subidos directamente desde Streamlit (en memoria)."""
    documents = []

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        
        # Procesar PDF en memoria
        if filename.endswith(".pdf"):
            pdf_reader = pypdf.PdfReader(uploaded_file)
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if raw_text:
                    # Limpia saltos de línea y espacios múltiples, uniendo el texto en un párrafo limpio
                    cleaned_text = " ".join(raw_text.split())
                    documents.append(
                        Document(
                            page_content=cleaned_text,
                            metadata={"source": filename, "page": page_num + 1}
                        )
                    )
        # Procesar TXT en memoria
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

    if not documents:
        return []

    # Aplicar Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.split_documents(documents)