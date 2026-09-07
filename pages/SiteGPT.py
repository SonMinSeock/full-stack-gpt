from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
import streamlit as st

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

html2text_transformer = Html2TextTransformer() # HTML 받아서 Text로 변환 해준다.

st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)

with st.sidebar:
    url = st.text_input("Write down a URL", placeholder="https://example.com")

if url:
    loader = AsyncChromiumLoader([url])
    docs = loader.load()
    transformed = html2text_transformer.transform_documents(docs)
    st.write(transformed)