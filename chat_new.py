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
    st.image("cepal.png", width=100)  # Asegúrate de proporcionar la ruta correcta al logo

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

# Inicialización de componentes
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
parser = StrOutputParser()
loader = DirectoryLoader('transcripciones/', glob="**/*.pdf")
pags = loader.load_and_split()
openai_api_key = OPENAI_API_KEY
embeddings = OpenAIEmbeddings(api_key=openai_api_key)
vectorstore = DocArrayInMemorySearch.from_documents(pags, embedding=embeddings)
retriever = vectorstore.as_retriever()

model = ChatOpenAI(model_name="gpt-4o", openai_api_key=openai_api_key, temperature=0, streaming=True)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente útil. Usa el siguiente contexto para responder la pregunta: {context}. No contestes preguntas que no se relacionen con el contexto"),
    ("human", "{question}")
])

# Configuración de la memoria
msgs = StreamlitChatMessageHistory(key="langchain_messages")
#memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

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

# Función para ejecutar la cadena y actualizar la memoria
def run_chain(question):
    result = chain.invoke({"question": question})
    #memory.save_context({"question": question}, {"output": result})
    return result

# Interfaz de usuario de Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("¿Qué quieres saber?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = run_chain(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Botón para limpiar el historial de chat
if st.button("Limpiar historial"):
    # Limpiar las mensajes
    if 'msgs' in locals() or 'msgs' in globals():
        msgs.clear()
    
    # Limpiar el estado de la sesión
    if 'messages' in st.session_state:
        st.session_state.messages = []
    
    # Usar rerun() en lugar de experimental_rerun()
    st.rerun()
