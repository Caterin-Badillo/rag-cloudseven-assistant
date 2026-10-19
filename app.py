import streamlit as st
from loader import process_uploaded_files
from vectorstore import create_local_vectorstore
from rag_chain import build_rag_chain

st.set_page_config(page_title="Prueba Técnica - Asistente RAG", page_icon="🤖", layout="wide")

st.title("🤖 Asistente Documental RAG")
st.caption("Prueba Técnica - AI Developer Engineer Junior | I Cloud Seven")

# Sidebar para API Key y carga de archivos
with st.sidebar:
    st.header("⚙️ Configuración")
    api_key = st.text_input("OpenAI API Key (Opcional si usas local)", type="password")
    uploaded_files = st.file_uploader("Cargar documentos (PDF / TXT)", accept_multiple_files=True, type=["pdf", "txt"])

if api_key:
    if uploaded_files:
        with st.spinner("Procesando documento(s) e indexando en FAISS..."):
            splits = process_uploaded_files(uploaded_files)
            
            if splits:
                vectorstore = create_local_vectorstore(splits, api_key)
                rag_chain = build_rag_chain(vectorstore, api_key)
                st.success(f"✅ {len(uploaded_files)} documento(s) procesado(s) exitosamente.")
            else:
                st.error("No se pudo extraer texto válido de los documentos cargados.")
                st.stop()

        # Chat interactivo
        user_query = st.text_input("🔍 Escribe tu pregunta sobre los documentos cargados:")
        if user_query:
            with st.spinner("Buscando en documentos..."):
                response = rag_chain.invoke({"input": user_query})
                
                st.subheader("💡 Respuesta:")
                st.write(response["answer"])

                with st.expander("📌 Ver fragmentos de contexto utilizados (Trazabilidad)"):
                    for idx, doc in enumerate(response["context"]):
                        source = doc.metadata.get("source", "Desconocido")
                        page = doc.metadata.get("page", None)
                        page_str = f" — Pág. {page}" if page else ""
                        st.markdown(f"**Fragmento {idx+1} — Fuente: `{source}`{page_str}**")
                        st.text(doc.page_content)
    else:
        st.info("👈 Por favor, sube al menos un documento en el panel izquierdo para comenzar.")
else:
    st.warning("👈 Ingresa tu OpenAI API Key en el panel izquierdo para activar el asistente.")