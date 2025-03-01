
import streamlit as st
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI

st.title("Chat avec IMEE 2024_0.pdf")

# Initialiser la conversation
@st.cache_resource
def init_conversation():
    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    vectorstore = FAISS.load_local("vectorstore")
    llm = ChatOpenAI(temperature=0, api_key=st.secrets["OPENAI_API_KEY"])
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return conversation

# Initialiser l'historique des messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

try:
    conversation = init_conversation()

    # Afficher l'historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Zone de chat
    if prompt := st.chat_input("Posez votre question sur le PDF"):
        # Afficher la question
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Obtenir et afficher la réponse
        with st.chat_message("assistant"):
            response = conversation({"question": prompt, "chat_history": []})
            st.markdown(response["answer"])
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

except Exception as e:
    st.error("Erreur lors du chargement du chatbot. Veuillez réessayer plus tard.")
