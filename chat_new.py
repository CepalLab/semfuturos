import streamlit as st
from langchain.document_loaders import DirectoryLoader
from langchain.schema.runnable import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from operator import itemgetter
import os
from dotenv import load_dotenv

# Configuración de la página de Streamlit
st.set_page_config(page_title="Chatbot Seminarios", page_icon="🧠")

# Colocando el título y el logo en columnas
col1, col2 = st.columns([1, 4])
with col1:
    st.image("cepal.png", width=100)
with col2:
    st.title("Chatbot Cepal Lab")

st.write("""
Hola soy un asistente virtual que brinda información respecto a la Primera Conferencia 
Regional de las Comisiones de Futuro Parlamentarias realizada en CEPAL el Santiago, 20 y 21 de junio de Junio. 
Esta conferencia organizada por la CEPAL y los parlamentos de Chile y Uruguay, convocó a expertos y parlamentarios
de la región y del mundo para conversar acerca de los principales temas de futuro y de las diversas experiencias 
respecto a la construcción de institucionalidad de prospectiva y de futuro.
A través de este chat podrás conocer en detalle aspectos tratados en esta importante conferencia.
""")

# Función con caché para cargar y vectorizar documentos
@st.cache_resource
def load_and_vectorize_documents():
    """
    Carga los PDFs y crea el vectorstore. 
    Se ejecuta solo una vez y se mantiene en caché.
    """
    loader = DirectoryLoader('transcripciones/', glob="**/*.pdf")
    pags = loader.load_and_split()
    
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    
    vectorstore = DocArrayInMemorySearch.from_documents(pags, embedding=embeddings)
    return vectorstore

# Función con caché para inicializar el modelo
@st.cache_resource
def initialize_model():
    """
    Inicializa el modelo de chat.
    Se ejecuta solo una vez y se mantiene en caché.
    """
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    model = ChatOpenAI(
        model_name="gpt-5-nano",
        #model_name="gpt-4o", 
        openai_api_key=openai_api_key, 
        #temperature=0, 
        streaming=True
    )
    return model

# Inicialización de componentes con caché
vectorstore = load_and_vectorize_documents()
retriever = vectorstore.as_retriever()
model = initialize_model()

# Parser y prompt (estos son ligeros, no necesitan caché)
parser = StrOutputParser()
prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente útil. Usa el siguiente contexto para responder la pregunta: {context}. No contestes preguntas que no se relacionen con el contexto"),
    ("human", "{question}")
])

# Configuración de la memoria
msgs = StreamlitChatMessageHistory(key="langchain_messages")

# Definición de la cadena
chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question")
    }
    | prompt
    | model
    | parser
)

# Función para ejecutar la cadena
def run_chain(question):
    result = chain.invoke({"question": question})
    return result

# Interfaz de usuario de Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_input := st.chat_input("¿Qué quieres saber?"):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)
    
    with st.chat_message("assistant"):
        response = run_chain(prompt_input)
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# Botón para limpiar el historial de chat
if st.button("Limpiar historial"):
    if 'msgs' in locals() or 'msgs' in globals():
        msgs.clear()
    
    if 'messages' in st.session_state:
        st.session_state.messages = []
    
    st.rerun()


