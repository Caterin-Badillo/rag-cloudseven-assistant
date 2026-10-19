# Asistente Documental RAG Local — Cloud Seven

Aplicación web interactiva basada en la arquitectura RAG (Retrieval-Augmented Generation) diseñada para realizar consultas inteligentes sobre documentos locales (PDF y TXT) utilizando un modelo de lenguaje de código abierto ejecutado 100% en memoria local, sin coste de licencias ni dependencia de APIs externas.

# Descripción de la solución

El sistema implementa un flujo RAG completo en memoria utilizando Python y Streamlit. Cuando el usuario carga un documento, este se procesa, limpia y divide en fragmentos (chunks). Posteriormente, se generan embeddings vectoriales mediante SentenceTransformers de HuggingFace y se indexan localmente en un vector store con FAISS. Para la generación de respuestas, se utiliza el modelo local de pesos ligeros SmolLM2-1.7B-Instruct a través del framework de LangChain (LCEL), integrando un filtro de seguridad en post-procesamiento para garantizar salidas limpias, precisas y libres de instrucciones técnicas residuales.

# Dependencias

El proyecto requiere Python 3.10 o superior y las siguientes librerías especificadas en requirements.txt:

* streamlit: Interfaz gráfica de usuario web.
  
* langchain, langchain-community, langchain-huggingface: Framework principal para cadenas RAG y conectores.
  
* faiss-cpu: Base de datos vectorial eficiente en memoria.
  
* transformers, torch: Carga y ejecución local del modelo LLM (SmolLM2-1.7B-Instruct).
  
* pypdf: Extracción de texto estructurado de archivos PDF.
  
* accelerate: Optimización de hardware para el modelo.

# Pasos de instalación
Clonar el repositorio:
* git clone https://github.com/Caterin-Badillo/rag-cloudseven-assistant.git
* cd rag-cloudseven-assistant

Crear y activar un entorno virtual:
* python -m venv venv
* venv\Scripts\activate

Instalar las dependencias del proyecto:
* python -m pip install --upgrade pip
* pip install -r requirements.txt

# Pasos de ejecución
Asegúrate de tener el entorno virtual activo.

Inicia la aplicación ejecutando Streamlit desde la terminal:
* python -m streamlit run app.py

Abre en tu navegador web la URL local proporcionada (usualmente http://localhost:8501).

# Cómo cargar documentos
En el panel lateral izquierdo (Configuración), haz clic en el botón de carga de archivos (Browse files).

Selecciona uno o varios documentos en formato PDF o TXT (por ejemplo, el documento de prueba del proyecto).

El sistema procesará el texto automáticamente y mostrará un mensaje de éxito (✅ Documento(s) procesado(s) exitosamente). El uploader se bloqueará con un candado virtual para mantener la integridad de la sesión en memoria.

# Cómo hacer preguntas
Una vez cargado y procesado el documento, aparecerá un campo de texto en la interfaz principal con el ícono de búsqueda: "Escribe tu pregunta sobre los documentos cargados".

Escribe tu consulta en lenguaje natural (ej. sobre datos operativos, financieros o técnicos del texto).

Presiona Enter. El sistema buscará los fragmentos más relevantes y mostrará la respuesta directa, junto con un bloque desplegable de trazabilidad para verificar las fuentes y páginas consultadas.

# Limitaciones conocidas
Al tratarse de un modelo de lenguaje local pequeño (1.7 mil millones de parámetros), su razonamiento complejo es limitado en comparación con modelos comerciales gigantes (como GPT-4).

Todo el procesamiento y almacenamiento de vectores se realiza de manera volátil en la memoria RAM (st.session_state); si se recarga o cierra la pestaña del navegador, la sesión se reinicia.

# Mejoras futuras
Persistencia opcional del índice vectorial en disco utilizando FAISS local guardado para evitar reprocesar documentos en cada inicio.

Integración de un historial de conversación multi-turno para mantener contexto entre preguntas consecutivas.

# Uso de Herramientas de Desarrollo Asistido por IA (AI-Assisted Development)
Durante el desarrollo de esta prueba técnica, se utilizaron herramientas de inteligencia artificial de forma complementaria para optimizar el flujo de trabajo:

* Asistencia con Gemini: Se utilizó durante la fase de diseño e ingeniería de prompts para estructurar las instrucciones del sistema orientadas a restringir al modelo local y manejar respuestas parciales o vacíos de información. Dado que los modelos pequeños tienden a filtrar instrucciones o a divagar, Gemini ayudó a iterar en la estrategia de post-procesamiento en código (clean_model_output) para asegurar una salida limpia en la interfaz de usuario.
* Asistencia con Kiro IDE: Se empleó como entorno de desarrollo integrado para estructurar la arquitectura modular del proyecto (app.py, loader.py, vectorstore.py, rag_chain.py, config.py), organizar el control de dependencias y redactar de forma limpia y estandarizada la documentación técnica y los comentarios del código.
* Revisiones manuales: Se realizaron revisiones y ajustes manuales exhaustivos sobre la gestión de estados en Streamlit (st.session_state), la lógica de bloqueo del uploader (is_locked), el manejo de rutas temporales y la validación de la estabilidad de la ejecución local con torch y transformers.
* Aprendizajes clave: Se profundizó en el comportamiento operativo de los LLMs locales ligeros (Small Language Models), comprendiendo la importancia crítica del formateo de prompts con etiquetas nativas de chat (<|im_start|>) y la necesidad de aplicar filtros de post-procesamiento para garantizar interfaces de usuario de calidad profesional.
