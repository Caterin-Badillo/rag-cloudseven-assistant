from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from config import MODEL_NAME, TOP_K_RESULTS

def build_rag_chain(vectorstore, api_key: str):
    """Construye la cadena RAG con un prompt estricto anti-alucinaciones."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RESULTS})
    
    system_prompt = (
        "Eres un asistente documental estricto de I Cloud Seven. Responde únicamente "
        "utilizando la información explícita del contexto proporcionado. "
        "Si la información no está presente en el contexto, responde exactamente: "
        "'No dispongo de información suficiente en los documentos cargados para responder a esta pregunta.'\n\n"
        "Contexto:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0, openai_api_key=api_key)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    return create_retrieval_chain(retriever, question_answer_chain)