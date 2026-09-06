from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import WikipediaRetriever
import streamlit as st
import wikipedia
import json
import os

USER_AGENT = "QuizGPTBot/1.0 (https://github.com/sonminseock)"

wikipedia.set_user_agent(USER_AGENT)

wikipedia.wikipedia.requests.Session().headers.update(
    {
        "User-Agent": USER_AGENT
    }
)

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

# Function Calling Schema
function = {
    "name": "create_quiz",
    "description": "Create a quiz from the provided context.",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": [
                                    "answer",
                                    "correct",
                                ],
                            },
                        },
                    },
                    "required": [
                        "question",
                        "answers",
                    ],
                },
            },
        },
        "required": [
            "questions",
        ],
    },
}

# 문서 → 문자열
def format_docs(docs):
    return "\n\n".join(
        document.page_content
        for document in docs
    )

# 파일 분할
@st.cache_data(show_spinner="Loading file...")
def split_file(file):

    file_content = file.read()

    os.makedirs(
        "./.cache/quiz_files",
        exist_ok=True,
    )

    file_path = f"./.cache/quiz_files/{file.name}"

    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = UnstructuredFileLoader(file_path)

    docs = loader.load_and_split(
        text_splitter=splitter
    )

    return docs

# Wikipedia 검색
@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(
        top_k_results=5
    )
    docs = retriever.get_relevant_documents(term)
    return docs

# Function Calling으로 Quiz 생성
@st.cache_data(show_spinner="Making Quiz...")
def run_quiz_chain(_docs, topic, difficulty, api_key):

    llm = ChatOpenAI(
        temperature=0.1,
        openai_api_key=api_key,
    ).bind(
        function_call={
            "name": "create_quiz"
        },
        functions=[
            function
        ],
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a helpful teacher.

Based ONLY on the following context,
create 10 multiple choice questions.

Each question must have exactly 4 answers.
Only one answer must be correct.

The difficulty of the quiz must be {difficulty}.

Easy:
Create simple questions that test basic facts
directly stated in the context.

Hard:
Create more challenging questions that require
careful understanding of the context.

Context:

{context}
                """,
            )
        ]
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "context": format_docs(_docs),
            "difficulty": difficulty,
        }
    )

    arguments = response.additional_kwargs[
        "function_call"
    ]["arguments"]

    return json.loads(arguments)

# Sidebar UI
with st.sidebar:
    st.title("Settings")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
    )

    difficulty = st.selectbox(
        "Choose quiz difficulty",
        (
            "Easy",
            "Hard",
        ),
    )

    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )

    st.divider()

    st.markdown(
        "[View the source code on GitHub]"
        "(https://github.com/sonminseock)"
    )

# API Key 확인
if not api_key:
    st.info(
        "Enter your OpenAI API Key in the sidebar "
        "to start the quiz."
    )
    st.stop()

# 데이터 선택
docs = None
topic = None
file = None

if choice == "File":
    file = st.sidebar.file_uploader(
        "Upload a .docx, .txt or .pdf file",
        type=[
            "pdf",
            "txt",
            "docx",
        ],
    )
    if file:
        docs = split_file(file)
else:
    topic = st.sidebar.text_input(
        "Search Wikipedia..."
    )

    if topic:

        try:
            docs = wiki_search(topic)

        except Exception:
            st.error(
                "Wikipedia search failed. "
                "Please try again or upload a file."
            )

if not docs:
    st.markdown(
        """
Welcome to QuizGPT.

I will make a quiz from Wikipedia articles
or files you upload to test your knowledge
and help you study.

### How to start

1. Enter your OpenAI API Key.
2. Choose Easy or Hard.
3. Upload a file or search Wikipedia.
4. Take the quiz!
        """
    )

    st.stop()

# Quiz 생성
source_name = topic if topic else file.name

response = run_quiz_chain(
    docs,
    source_name,
    difficulty,
    api_key,
)

questions = response["questions"]

# Quiz Form UI
with st.form("questions_form"):
    user_answers = []
    for index, question in enumerate(questions):
        st.subheader(
            f"Question {index + 1}"
        )

        st.write(
            question["question"]
        )

        value = st.radio(
            "Select an option.",
            [
                answer["answer"]
                for answer
                in question["answers"]
            ],
            index=None,
            key=f"question_{index}",
        )

        user_answers.append(value)

    submitted = st.form_submit_button(
        "Submit Quiz"
    )

# 채점
if submitted:
    score = 0
    for index, question in enumerate(questions):
        selected_answer = user_answers[index]
        if {
            "answer": selected_answer,
            "correct": True,
        } in question["answers"]:
            score += 1

    st.divider()

    st.subheader(
        f"Score: {score} / {len(questions)}"
    )

    if score == len(questions):
        st.success(
            "Perfect score! 🎉"
        )
        st.balloons()

    # 재시험
    else:
        st.error(
            f"You got "
            f"{len(questions) - score} "
            f"question(s) wrong."
        )
        st.info(
            "You can change your answers above "
            "and submit the quiz again."
        )