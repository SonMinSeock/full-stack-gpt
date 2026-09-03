import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import Chroma
from langchain.storage import LocalFileStore
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler


st.set_page_config(
    page_title="Streamlit 챌린지",
    page_icon="🤖",
)


# Session State
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
    )


# File Embedding
@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file, api_key):
    file_content = file.read()

    # Streamlit Cloud에서도 필요한 캐시 폴더 자동 생성
    os.makedirs("./.cache/files", exist_ok=True)
    os.makedirs("./.cache/embeddings", exist_ok=True)

    file_path = f"./.cache/files/{file.name}"

    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(
        f"./.cache/embeddings/{file.name}"
    )

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = UnstructuredFileLoader(file_path)

    docs = loader.load_and_split(
        text_splitter=splitter,
    )

    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key,
    )

    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        cache_dir,
    )

    vectorstore = Chroma.from_documents(
        docs,
        cached_embeddings,
    )

    return vectorstore.as_retriever()


# Chat Functions
def save_message(message, role):
    st.session_state["messages"].append(
        {
            "message": message,
            "role": role,
        }
    )


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)

    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


# Streaming Callback
class ChatCallbackHandler(BaseCallbackHandler):

    def on_llm_start(self, *args, **kwargs):
        self.message = ""
        self.message_box = st.empty()

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

    def on_llm_end(self, *args, **kwargs):
        save_message(
            self.message,
            "ai",
        )


# RAG Functions
def format_docs(docs):
    return "\n\n".join(
        document.page_content
        for document in docs
    )


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful AI assistant.

        Answer the user's question using only the following context.
        If you don't know the answer based on the context, say you don't know.

        Context:
        {context}
        """
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])


# UI
st.title("Streamlit 챌린지")

st.markdown(
    """
    Welcome!

    Use this chatbot to ask questions to an AI about your files!

    Upload your file on the sidebar.
    """
)


# Sidebar
with st.sidebar:
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
    )

    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

    st.markdown(
        "[View the code on GitHub](https://github.com/SonMinSeock/full-stack-gpt)"
    )

if file and api_key:
    retriever = embed_file(
        file,
        api_key,
    )

    send_message(
        "I'm ready! Ask away!",
        "ai",
        save=False,
    )

    paint_history()

    message = st.chat_input(
        "Ask anything about your file..."
    )

    if message:
        send_message(
            message,
            "human",
        )

        def retrieve_docs(inputs):
            docs = retriever.invoke(
                inputs["question"]
            )
            return format_docs(docs)

        memory = st.session_state["memory"]

        history = memory.load_memory_variables(
            {}
        )["history"]

        llm = ChatOpenAI(
            temperature=0.1,
            openai_api_key=api_key,
            streaming=True,
            callbacks=[
                ChatCallbackHandler(),
            ],
        )

        chain = (
            RunnablePassthrough.assign(
                context=retrieve_docs,
            )
            | prompt
            | llm
        )

        with st.chat_message("ai"):
            response = chain.invoke(
                {
                    "question": message,
                    "history": history,
                }
            )

        memory.save_context(
            {
                "input": message,
            },
            {
                "output": response.content,
            },
        )


elif file and not api_key:
    st.info(
        "Please enter your OpenAI API Key."
    )

else:
    st.session_state["messages"] = []

    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
    )