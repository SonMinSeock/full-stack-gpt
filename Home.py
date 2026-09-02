import streamlit as st
from langchain.prompts import PromptTemplate

# write
# st.write("hello")
# st.write([1, 2, 3, 4])
# st.write({"x" : 1})
# st.write(PromptTemplate)

# prompt = PromptTemplate.from_template("xxxx")

# st.write(prompt)

# magic
"hello"
[1, 2, 3, 4]
{"x": 1}

prompt = PromptTemplate.from_template("xxxx")
prompt

st.selectbox("Choose your model", ("GPT-3", "GPT-4"))