import os
import streamlit as st
from typing import Type
from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, DuckDuckGoSearchResults
from langchain.utilities import WikipediaAPIWrapper
from langchain.document_loaders import WebBaseLoader
from langchain.agents import initialize_agent, AgentType
from langchain.schema import SystemMessage
from langchain.callbacks.base import BaseCallbackHandler

st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔎",
)

st.title("Research Assistant")
st.caption("AI research assistant powered by OpenAI")

with st.sidebar:
    st.title("Settings")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
    )

    st.divider()

    st.markdown(
        "[Github Repository]"
        "(https://github.com/sonminseock/full-stack-gpt)"
    )

if not api_key:
    st.info("Please enter your OpenAI API Key in the sidebar.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key
class StreamHandler(BaseCallbackHandler):

    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token

        self.container.markdown(
            self.text
        )



# Wikipedia Tool
class WikipediaSearchToolArgsSchema(BaseModel):

    query: str = Field(
        description="The topic to search for on Wikipedia."
    )

class WikipediaSearchTool(BaseTool):
    name = "WikipediaSearchTool"

    description = """
    Use this tool to search Wikipedia for information
    about a topic.

    It takes a search query as an argument.
    """

    args_schema: Type[
        WikipediaSearchToolArgsSchema
    ] = WikipediaSearchToolArgsSchema

    def _run(self, query):

        wikipedia = WikipediaAPIWrapper()

        return wikipedia.run(query)

# DuckDuckGo Tool
class DuckDuckGoSearchToolArgsSchema(BaseModel):
    query: str = Field(
        description="The query to search for on DuckDuckGo."
    )

class DuckDuckGoSearchTool(BaseTool):
    name = "DuckDuckGoSearchTool"

    description = """
    Use this tool to search the web using DuckDuckGo.

    Use it when you need to find websites or additional
    information about a topic.

    The results may contain URLs that can be passed to
    WebsiteScrapingTool.
    """

    args_schema: Type[
        DuckDuckGoSearchToolArgsSchema
    ] = DuckDuckGoSearchToolArgsSchema

    def _run(self, query):

        ddg = DuckDuckGoSearchResults()

        return ddg.run(query)

# Website Scraping Tool
class WebsiteScrapingToolArgsSchema(BaseModel):
    url: str = Field(
        description="The URL of the website to scrape."
    )

class WebsiteScrapingTool(BaseTool):
    name = "WebsiteScrapingTool"

    description = """
    Use this tool to extract the text content of a website.

    Pass a complete URL found from DuckDuckGo search results.

    Example:
    https://example.com/article
    """

    args_schema: Type[
        WebsiteScrapingToolArgsSchema
    ] = WebsiteScrapingToolArgsSchema

    def _run(self, url):

        loader = WebBaseLoader(url)

        docs = loader.load()

        return "\n\n".join(
            doc.page_content
            for doc in docs
        )

# Save To File Tool

class SaveToFileToolArgsSchema(BaseModel):
    content: str = Field(
        description=(
            "The complete research content "
            "that should be saved."
        )
    )

class SaveToFileTool(BaseTool):
    name = "SaveToFileTool"

    description = """
    Use this tool to save the completed research report
    to research.txt.

    You MUST use this tool after completing the research.

    Pass the entire final research report as the
    content argument.
    """

    args_schema: Type[
        SaveToFileToolArgsSchema
    ] = SaveToFileToolArgsSchema

    def _run(self, content):

        os.makedirs(
            "./files/research",
            exist_ok=True,
        )

        filename = "./files/research/research.txt"

        with open(
            filename,
            "w",
            encoding="utf-8",
        ) as file:
            file.write(content)

        return (
            f"Research successfully saved to {filename}"
        )

# Conversation History
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )

message = st.chat_input(
    "What do you want to research?"
)

if message:
    st.session_state["messages"].append(
        {
            "role": "user",
            "content": message,
        }
    )

    with st.chat_message("user"):
        st.markdown(message)

    with st.chat_message("assistant"):
        status = st.status(
            "🔎 Researching...",
            expanded=True,
        )

        status.write(
            "Searching for information..."
        )

        stream_container = st.empty()

        stream_handler = StreamHandler(
            stream_container
        )

        llm = ChatOpenAI(
            temperature=0.1,
            streaming=True,
            callbacks=[
                stream_handler
            ],
        )

        agent = initialize_agent(
            llm=llm,
            verbose=True,
            agent=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True,
            tools=[
                WikipediaSearchTool(),
                DuckDuckGoSearchTool(),
                WebsiteScrapingTool(),
                SaveToFileTool(),
            ],
            agent_kwargs={
                "system_message": SystemMessage(
                    content="""
                    You are a research assistant.

                    When the user asks you to research a topic:

                    1. Search Wikipedia and DuckDuckGo.

                    2. Find useful websites from the
                       search results.

                    3. Visit useful websites using
                       WebsiteScrapingTool.

                    4. Use the collected information
                       to write a detailed research report.

                    5. Include the sources used.

                    6. ALWAYS invoke SaveToFileTool
                       with the complete research report.

                    You MUST invoke SaveToFileTool before
                    returning your final answer.

                    Never finish a research task without
                    saving the report.
                    """
                )
            },
        )

        try:
            result = agent.invoke(
                message
            )

            answer = result["output"]

            status.update(
                label="✅ Research complete!",
                state="complete",
                expanded=False,
            )

        except Exception as e:
            answer = (
                f"An error occurred: {e}"
            )

            status.update(
                label="❌ Research failed",
                state="error",
                expanded=True,
            )

            stream_container.error(
                answer
            )

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": answer,
        }
    )