# Modulo central del sistema RAG. Configura el modelo LLM local (SmolLM2-1.7B-Instruct), establece el prompt estructurado con el formato de chat nativo, implementa una función de post-procesamiento para limpiar la salida y construye la cadena ejecutable (LCEL) junto con la trazabilidad.
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import MODEL_ID, EMBEDDING_MODEL_ID, TOP_K_RESULTS

def create_local_vectorstore(splits):
    # Crea la base vectorial en memoria utilizando FAISS y los embeddings de HuggingFace.
    # Args:
        # splits (list): Lista de fragmentos de texto procesados.
    # Returns:
        # FAISS: Base de datos vectorial indexada.

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_ID)
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

def clean_model_output(text: str) -> str:
    # Función de post-procesamiento para limpiar instrucciones residuales o etiquetas técnicas que los modelos de lenguaje pequeños puedan generar de forma accidental en la salida.
    # Args:
        # text (str): Texto crudo devuelto por el pipeline del modelo.        
    # Returns:
        # str: Respuesta limpia, directa y libre de marcas de control para el usuario final.
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
    # Construye y configura la cadena RAG completa empleando LCEL (LangChain Expression Language), integrando el recuperador de documentos, el prompt estricto, el pipeline de generación local y el filtro automático de limpieza de salida.
    # Args:
        # vectorstore: Instancia de FAISS con los documentos indexados en memoria.
    # Returns:
        # Runnable: Cadena ejecutable lista para responder consultas con contexto y trazabilidad.
    
    # 1. Cargar el tokenizador y el modelo de lenguaje causal desde HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map="auto"
    )

    # 2. Configurar el pipeline de generación de texto optimizado para inferencia local
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        temperature=0.1,
        do_sample=True,
        return_full_text=False,  # Evita que devuelva el prompt de entrada junto con la respuesta generada
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # 3. Definir el prompt estructurado utilizando el formato de chat nativo de SmolLM2-Instruct
    template = """<|im_start|>system
    Eres un asistente virtual experto y preciso. Utiliza exclusivamente el contexto para responder a la pregunta de forma directa, clara y concisa (máximo 1 o 2 oraciones). Si la respuesta no está en el texto, responde exactamente: 'No dispongo de información suficiente en los documentos cargados para responder a esta pregunta.'<|im_end|>
    <|im_start|>user
    Contexto:
    {context}

    Pregunta: {input}<|im_end|>
    <|im_start|>assistant
    """

    prompt = PromptTemplate.from_template(template)

    # 4. Configurar el retriever para extraer los K fragmentos más relevantes de FAISS
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RESULTS})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 5. Construcción de la cadena RAG con LCEL (Retrieval + Prompt + LLM + Parser + Filtro de limpieza)
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
        | clean_model_output  # Filtro de seguridad para remover texto técnico residual
    )

    # Enlazar la recuperación de los metadatos y fuentes junto con la respuesta final generada
    rag_chain_with_source = RunnablePassthrough.assign(
        context=lambda x: retriever.invoke(x["input"])
    ).assign(
        answer=rag_chain_from_docs
    )

    return rag_chain_with_source