# Orquestador principal de la interfaz de usuario desarrollada con Streamlit.
# Gestiona la carga de documentos en memoria, el bloqueo dinámico del uploader, el mantenimiento del estado de la sesión, la interacción por chat y la visualización de la trazabilidad (fragmentos de contexto utilizados).

import streamlit as st
from loader import process_uploaded_files
from vectorstore import create_local_vectorstore
from rag_chain import build_rag_chain

# Configuración inicial de la página web en Streamlit (diseño ancho y metadatos)
st.set_page_config(page_title="Prueba Técnica - Asistente RAG", page_icon="🤖", layout="wide")

st.title("🤖 Asistente Documental RAG")
st.caption("Prueba Técnica - AI Developer Engineer Junior | I Cloud Seven")

# Definir el mecanismo de bloqueo del cargador de archivos:
# Se deshabilita automáticamente si ya existe una cadena RAG activa en memoria,
# evitando modificaciones accidentales durante la sesión de chat.
is_locked = st.session_state.get("rag_chain") is not None

# Barra lateral (Sidebar) para la configuración y carga de archivos
with st.sidebar:
    st.header("⚙️ Configuración")

    # El parámetro disabled=is_locked bloquea el uploader para proteger el estado activo
    uploaded_files = st.file_uploader(
        "Cargar documentos (PDF / TXT)", 
        accept_multiple_files=True, 
        type=["pdf", "txt"],
        disabled=is_locked
    )
    
    # Mensaje informativo para guiar al usuario cuando los archivos están bloqueados
    if is_locked:
        st.info("🔒 Archivos bloqueados en memoria. Reinicia la página si deseas cargar otros diferentes.")

# Inicializar las variables de estado de la sesión para evitar reprocesamientos innecesarios
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "processed_files_sig" not in st.session_state:
    st.session_state.processed_files_sig = None

# Lógica condicional de procesamiento cuando el usuario proporciona documentos
if uploaded_files:
    # Crear una firma única basada en los nombres de los archivos actuales cargados
    current_files_sig = tuple(f.name for f in uploaded_files)
    
    # Procesar e indexar únicamente si es un archivo nuevo o la cadena RAG no ha sido inicializada
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

    # Interfaz de chat interactivo (las preguntas ya no vuelven a procesar los documentos gracias a la sesión)
    user_query = st.text_input("🔍 Escribe tu pregunta sobre los documentos cargados:")

    if user_query:
        with st.spinner("Buscando respuesta en el documento..."):
            # Invocar la cadena RAG almacenada en la sesión con la consulta del usuario
            response = st.session_state.rag_chain.invoke({"input": user_query})
            
            # Mostrar la respuesta generada por el modelo local
            st.subheader("💡 Respuesta:")
            st.write(response["answer"])

            # Sección desplegable para mostrar la trazabilidad y fragmentos exactos consultados
            with st.expander("📌 Ver fragmentos de contexto utilizados (Trazabilidad)"):
                for idx, doc in enumerate(response["context"]):
                    source = doc.metadata.get("source", "Desconocido")
                    page = doc.metadata.get("page", None)
                    page_str = f" — Pág. {page}" if page else ""
                    st.markdown(f"**Fragmento {idx+1} — Fuente: `{source}`{page_str}**")
                    st.text(doc.page_content)
else:
    # Restablecer el estado de la sesión si se eliminan los archivos del uploader
    st.session_state.rag_chain = None
    st.session_state.processed_files_sig = None
    st.info("👈 Por favor, sube al menos un documento en el panel izquierdo para comenzar.")