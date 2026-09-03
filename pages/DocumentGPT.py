import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import Chroma
from langchain.storage import LocalFileStore

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📄",
)

def embed_file(file):
    if file:
        file_content = file.read()
        file_path = f"./.cache/files/{file.name}"
        
        with open(file_path, "wb") as f:
            f.write(file_content)
        # 캐시 경로 설정
        cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )

        loader = UnstructuredFileLoader("./files/chapter_one.txt")

        docs = loader.load_and_split(text_splitter=splitter)

        embeddings = OpenAIEmbeddings()

        cached_embenddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings,
            cache_dir
        )

        #vector store는 일종의 데이터베이스라고 생각하면된다. 벡터 공간에서 검색 할 수 있게한다.
        vectorstore = Chroma.from_documents(docs, cached_embenddings)

        retriever = vectorstore.as_retriever() # document들의 list로 반환 할거다.
        return retriever


st.title("DocumentGPT")

st.markdown("""
Welcome!

Use this chatbot to ask questions to an AI about your files!
"""
)

file = st.file_uploader("Upload a .txt .pdf or .docx file", type=["pdf", "txt", "docx"])

if file:
    retriever = embed_file(file)
    response = retriever.invoke("winston")
    response