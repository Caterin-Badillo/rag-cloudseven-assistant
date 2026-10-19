import os
from dotenv import load_dotenv

load_dotenv()

# Ajustes por defecto de Chunking y RAG
CHUNK_SIZE = 500
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 3
MODEL_NAME = "gpt-4o-mini"