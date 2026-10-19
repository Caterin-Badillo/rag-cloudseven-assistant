import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import MODEL_ID, EMBEDDING_MODEL_ID, TOP_K_RESULTS

def create_local_vectorstore(splits):
    """Crea la base vectorial en memoria usando FAISS y embeddings de HuggingFace."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_ID)
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

def clean_model_output(text: str) -> str:
    """Función de post-procesamiento para limpiar instrucciones residuales del modelo."""
    if not text:
        return ""
    
    # Si el modelo incluye las etiquetas de chat, nos quedamos solo con la parte del asistente
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1]
    
    # Si quedan restos de etiquetas del sistema o usuario, limpiamos
    for tag in ["<|im_start|>", "<|im_end|>", "system", "user", "Respuesta clara y directa:"]:
        text = text.replace(tag, "")
        
    return text.strip()

def build_rag_chain(vectorstore):
    """Construye la cadena RAG con un filtro de limpieza para asegurar respuestas limpias."""
    
    # 1. Cargar Tokenizer y Modelo local
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map="auto"
    )

    # 2. Pipeline de HuggingFace
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        temperature=0.1,
        do_sample=True,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # 3. Prompt estructurado con formato nativo de SmolLM2
    template = """<|im_start|>system
Eres un asistente virtual experto y preciso. Utiliza exclusivamente el contexto para responder a la pregunta de forma directa, clara y concisa (máximo 1 o 2 oraciones). Si la respuesta no está en el texto, responde exactamente: 'No dispongo de información suficiente en los documentos cargados para responder a esta pregunta.'<|im_end|>
<|im_start|>user
Contexto:
{context}

Pregunta: {input}<|im_end|>
<|im_start|>assistant
"""

    prompt = PromptTemplate.from_template(template)

    # 4. Configurar el retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RESULTS})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 5. Construir la cadena con LCEL y aplicar la limpieza automática
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
        | clean_model_output  # <--- ¡Filtro de seguridad para eliminar basura e instrucciones!
    )

    rag_chain_with_source = RunnablePassthrough.assign(
        context=lambda x: retriever.invoke(x["input"])
    ).assign(
        answer=rag_chain_from_docs
    )

    return rag_chain_with_source