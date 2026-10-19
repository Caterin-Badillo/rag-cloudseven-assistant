import os

# Modelos de HuggingFace utilizados
MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Parámetros de recuperación y fragmentación (Chunking)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 2