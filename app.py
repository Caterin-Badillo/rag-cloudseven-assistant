import streamlit as st
from loader import process_uploaded_files
from vectorstore import create_local_vectorstore
from rag_chain import build_rag_chain

st.set_page_config(page_title="Prueba Técnica - Asistente RAG", page_icon="🤖", layout="wide")

st.title("🤖 Asistente Documental RAG")
st.caption("Prueba Técnica - AI Developer Engineer Junior | I Cloud Seven")

# Definir si el uploader debe estar deshabilitado
# Por ejemplo, se deshabilita si ya existe una cadena RAG activa en memoria
is_locked = st.session_state.get("rag_chain") is not None

# Sidebar para carga de archivos
with st.sidebar:
    st.header("⚙️ Configuración")
    # El parámetro disabled=is_locked bloquea el uploader para que no puedan borrar ni cambiar archivos
    uploaded_files = st.file_uploader(
        "Cargar documentos (PDF / TXT)", 
        accept_multiple_files=True, 
        type=["pdf", "txt"],
        disabled=is_locked
    )
    
    if is_locked:
        st.info("🔒 Archivos bloqueados en memoria. Reinicia la página si deseas cargar otros diferentes.")

# Inicializar estado de la sesión para evitar reprocesar innecesariamente
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "processed_files_sig" not in st.session_state:
    st.session_state.processed_files_sig = None

if uploaded_files:
    # Crear una firma con los nombres de los archivos actuales
    current_files_sig = tuple(f.name for f in uploaded_files)
    
    # Procesar solo si es un archivo nuevo o no ha sido indexado en la sesión
    if st.session_state.rag_chain is None or st.session_state.processed_files_sig != current_files_sig:
        with st.spinner("Procesando documento(s) e indexando en FAISS..."):
            splits = process_uploaded_files(uploaded_files)
            if splits:
                vectorstore = create_local_vectorstore(splits)
                st.session_state.rag_chain = build_rag_chain(vectorstore)
                st.session_state.processed_files_sig = current_files_sig
                st.success(f"✅ {len(uploaded_files)} documento(s) procesado(s) exitosamente.")
            else:
                st.error("No se pudo extraer texto válido de los documentos cargados.")
                st.stop()
    else:
        st.success(f"✅ {len(uploaded_files)} documento(s) cargado(s) correctamente (en memoria).")

    # Chat interactivo (las preguntas ya no reprocesan los documentos)
    user_query = st.text_input("🔍 Escribe tu pregunta sobre los documentos cargados:")
    if user_query:
        with st.spinner("Buscando respuesta en el documento..."):
            response = st.session_state.rag_chain.invoke({"input": user_query})
            
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
    # Limpiar estado si se borran los archivos
    st.session_state.rag_chain = None
    st.session_state.processed_files_sig = None
    st.info("👈 Por favor, sube al menos un documento en el panel izquierdo para comenzar.")