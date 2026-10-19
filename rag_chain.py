from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config import TOP_K_RESULTS
import torch

def format_docs(docs):
    """Concatena el texto de todos los fragmentos recuperados."""
    return "\n\n".join(doc.page_content for doc in docs)

class LocalRAGChain:
    """Cadena RAG 100% gratuita ejecutada localmente en tu PC."""
    def __init__(self, retriever, chain):
        self.retriever = retriever
        self.chain = chain

    def invoke(self, inputs: dict):
        query = inputs.get("input", "")
        # 1. Recuperar los documentos para la trazabilidad
        docs = self.retriever.invoke(query)
        # 2. Generar respuesta localmente
        answer = self.chain.invoke(query)
        return {
            "answer": answer,
            "context": docs
        }

def build_rag_chain(vectorstore, api_key: str = None):
    """Construye la cadena RAG local y gratuita usando HuggingFace Transformers moderno."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RESULTS})
    
    # Usar un modelo moderno de causal LM compatible con la nueva API de pipelines
    model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        temperature=0.1,
        do_sample=True
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    
    # Prompt optimizado para modelos de chat locales
    system_prompt = (
        "Responde a la pregunta del usuario utilizando únicamente el siguiente contexto. "
        "Si la respuesta no se encuentra en el texto, responde exactamente: "
        "'No dispongo de información suficiente en los documentos cargados para responder a esta pregunta.'\n\n"
        "Contexto:\n{context}\n"
        "Pregunta: {input}\n"
        "Respuesta:"
    )
    
    prompt = ChatPromptTemplate.from_template(system_prompt)
    
    chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return LocalRAGChain(retriever, chain)