from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st

st.set_page_config(
    page_title="Cloudflare SiteGPT",
    page_icon="☁️",
)

SITEMAP_URL = "https://developers.cloudflare.com/sitemap-0.xml"

with st.sidebar:
    st.title("Cloudflare SiteGPT")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
    )

    st.markdown("---")

    st.markdown(
        "[View the code on GitHub](https://github.com/SonMinSeock/full-stack-gpt)"
    )

st.markdown(
    """
    # ☁️ Cloudflare SiteGPT

    Ask questions about Cloudflare documentation.

    This chatbot can answer questions about:

    - AI Gateway
    - Cloudflare Vectorize
    - Workers AI
    """
)

if not api_key:
    st.info("Please enter your OpenAI API Key in the sidebar.")
    st.stop()


llm = ChatOpenAI(
    temperature=0.1,
    openai_api_key=api_key,
)

# Map Prompt
answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question.

    If you can't answer the question using the context,
    just say "I don't know".

    Do not make anything up.

    Then, give the answer a score between 0 and 5.

    If the answer fully answers the user's question,
    the score should be high.

    If the context does not contain the answer,
    the score should be low.

    Make sure to always include the answer's score,
    even if it is 0.

    Context:
    {context}

    Examples:

    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5

    Question: How far away is the sun?
    Answer: I don't know.
    Score: 0

    Your turn!

    Question: {question}
    """
)

# MAP
def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]

    answers_chain = answers_prompt | llm

    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {
                        "question": question,
                        "context": doc.page_content,
                    }
                ).content,
                "source": doc.metadata.get("source", ""),
                "date": doc.metadata.get("lastmod", ""),
            }
            for doc in docs
        ],
    }


# Re-Rank Prompt
choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers
            to answer the user's question.

            Use the answers that have the highest score.
            Higher scores mean that the answer is more helpful
            and relevant to the user's question.

            If multiple answers have high scores,
            favor the most recent ones.

            If none of the answers contain enough information,
            say "I don't know".

            Do not make anything up.

            Cite the sources used in your final answer.

            Return source URLs exactly as they are.
            Do not modify the URLs.

            Answers:

            {answers}
            """,
        ),
        (
            "human",
            "{question}",
        ),
    ]
)

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]

    choose_chain = choose_prompt | llm

    condensed = "\n\n".join(
        f"""
Answer:
{answer["answer"]}

Source:
{answer["source"]}

Date:
{answer["date"]}
"""
        for answer in answers
    )

    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )

# Cloudflare Page Parser
def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")

    if header:
        header.decompose()

    if footer:
        footer.decompose()

    return (
        soup.get_text(" ", strip=True)
        .replace("\xa0", " ")
    )

# Load Cloudflare Documentation
@st.cache_resource(show_spinner="Loading Cloudflare documentation...")
def load_website(api_key):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )

    loader = SitemapLoader(
        SITEMAP_URL,
        filter_urls=[
            r"https://developers\.cloudflare\.com/ai-gateway/.*",
            r"https://developers\.cloudflare\.com/vectorize/.*",
            r"https://developers\.cloudflare\.com/workers-ai/.*",
        ],
        parsing_function=parse_page,
    )

    loader.requests_per_second = 2

    docs = loader.load_and_split(
        text_splitter=splitter
    )

    if not docs:
        raise ValueError("No Cloudflare documentation was loaded.")

    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key,
    )

    batch_size = 100

    first_batch = docs[:batch_size]

    vector_store = FAISS.from_documents(
        first_batch,
        embeddings,
    )

    for i in range(batch_size, len(docs), batch_size):
        batch = docs[i:i + batch_size]

        vector_store.add_documents(
            batch
        )

    return vector_store.as_retriever(
        search_kwargs={
            "k": 6,
        }
    )

try:
    retriever = load_website(api_key)

except Exception as e:
    st.error(
        f"Failed to load Cloudflare documentation: {e}"
    )
    st.stop()

query = st.text_input(
    "Ask a question about Cloudflare.",
    placeholder="What can I do with Cloudflare's AI Gateway?",
)

# Map Re-Rank Chain
if query:
    chain = (
        {
            "docs": retriever,
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(get_answers)
        | RunnableLambda(choose_answer)
    )

    with st.spinner("Searching Cloudflare documentation..."):
        try:
            result = chain.invoke(query)

            st.markdown("### Answer")

            st.markdown(
                result.content.replace("$", "\\$")
            )

        except Exception as e:
            st.error(
                f"An error occurred: {e}"
            )