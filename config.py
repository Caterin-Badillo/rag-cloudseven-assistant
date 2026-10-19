# Configuración general del proyecto RAG (Retrieval-Augmented Generation).
# Define los identificadores de modelos de HuggingFace y parámetros de fragmentación (chunking).

# Identificador del modelo LLM local (SmolLM2 optimizado para instrucciones)
MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
# Identificador del modelo de embeddings para conversión semántica de texto
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Parámetros de recuperación y segmentación de documentos
CHUNK_SIZE = 500       # Número de fragmentos más relevantes a recuperar de la base vectorial
CHUNK_OVERLAP = 200    # Tamaño máximo de caracteres por cada fragmento de texto
TOP_K_RESULTS = 2      # Solapamiento entre fragmentos consecutivos para mantener contexto