from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA


import os
import tempfile
import streamlit as st

# Verificar si la clave de API de OpenAI está configurada en Streamlit Secrets
if "OPENAI_API_KEY" not in st.secrets:
    st.error("La clave de API de OpenAI no está configurada. Por favor, configura la clave de API de OpenAI en Streamlit Secrets.")
    st.stop()
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Funciones para la creación del chatbot RAG
@st.cache_resource
def load_data(pdf_file):
    """Carga datos desde un documento PDF."""
    try:
        # Crear un archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_file_path = tmp_file.name
        loader = PyPDFLoader(tmp_file_path)  # Pasar la ruta del archivo temporal
        documentos = loader.load()
        os.remove(tmp_file_path)  # Limpiar el archivo temporal
        return documentos
    except Exception as e:
        st.error(f"Error al cargar el PDF: {e}")
        return None

@st.cache_resource
def crear_vector_store(_documentos):
    """Crea un vector store FAISS a partir de documentos."""
    if _documentos is None:
        return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    textos = text_splitter.split_documents(_documentos)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_documents(textos, embeddings)
    return vector_store


@st.cache_resource
def crear_cadena_qa(_vector_store, temperature: float = 0, k: int = 5, chain_type: str = "stuff"):
    """
    Crea una cadena de pregunta-respuesta utilizando OpenAI y un vector store (por ejemplo, FAISS).

    Parámetros:
      _vector_store: Objeto que contiene el índice vectorial y que debe implementar el método `as_retriever`.
                     Se utiliza un nombre con guión bajo para que Streamlit no lo incluya en la clave de caché.
      temperature: Grado de aleatoriedad en las respuestas del modelo de lenguaje (por defecto, 0).
      k: Número de documentos a recuperar para formular la respuesta (por defecto, 5).
      chain_type: Tipo de cadena a usar en el proceso de QA (por defecto, "stuff").

    Retorna:
      Una instancia de RetrievalQA si _vector_store es válido; de lo contrario, retorna None.
    """
    if _vector_store is None:
        st.error("No se proporcionó un vector store válido.")
        return None

    if not hasattr(_vector_store, "as_retriever"):
        raise ValueError("El vector store proporcionado no tiene el método 'as_retriever'.")

    # Crear el modelo de lenguaje con la configuración especificada.
    llm = OpenAI(temperature=temperature, openai_api_key=openai_api_key)

    # Configurar el recuperador, especificando el número de documentos a recuperar.
    retriever = _vector_store.as_retriever(search_kwargs={"k": k})

    # Construir la cadena de QA utilizando el modelo, el recuperador y el tipo de cadena indicado.
    cadena_qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type=chain_type
    )
    return cadena_qa


# Interfaz de usuario de Streamlit
def main():
    st.title("Chatbot RAG con PDF, Streamlit, Langchain y FAISS")
    pdf_file = st.file_uploader("Carga tu documento PDF:", type="pdf")
    if pdf_file is not None:
        documentos = load_data(pdf_file)
        if documentos:
            vector_store = crear_vector_store(documentos)
            if vector_store:
                cadena_qa = crear_cadena_qa(vector_store)
                if cadena_qa:
                    if "messages" not in st.session_state:
                        st.session_state.messages = []
                    # Mostrar mensajes de chat del historial
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])
                    # Entrada de chat
                    if prompt := st.chat_input("Escribe tu pregunta sobre el PDF:"):
                        # Añadir mensaje del usuario al historial de chat
                        st.session_state.messages.append({"role": "user", "content": prompt})
                        with st.chat_message("user"):
                            st.markdown(prompt)
                        with st.chat_message("assistant"):
                            message_placeholder = st.empty()
                            try:
                                with st.spinner("Procesando..."):
                                    respuesta = cadena_qa.run(prompt)
                                    message_placeholder.markdown(respuesta)
                                    st.session_state.messages.append({"role": "assistant", "content": respuesta})
                            except Exception as e:
                                st.error(f"Error: {e}")
                                message_placeholder.markdown("Lo siento, hubo un error al procesar tu pregunta sobre el PDF.")
                else:
                    st.error("Error al crear la cadena de pregunta-respuesta.")
            else:
                st.error("Error al crear el vector store.")
    else:
        st.info("Por favor, carga un documento PDF para empezar a chatear.")

if __name__ == "__main__":
    main()