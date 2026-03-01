import streamlit as st
import os
from dotenv import load_dotenv
from graph import ResearchAgent
from langchain_openai import AzureChatOpenAI

# Load environment variables from .env file
load_dotenv()

def check_env_variables():
    """Checks if all required environment variables are set."""
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "OPENAI_API_VERSION",
        "OPENAI_API_DEPLOYMENT_NAME",
        "OPENAI_API_KEY",
        "QDRANT_URL",
        "SEARX_URL",
        "BROWSER_WS_ENDPOINT",
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        st.error(
            "The following required environment variables are not set: "
            f"`{', '.join(missing_vars)}`. Please check your `.env` file."
        )
        st.stop()

def get_callbacks(provider: str):
    """Initializes and returns callbacks based on the selected provider."""
    callbacks = []
    if provider == "langsmith":
        st.info("Using LangSmith for observability.")
        if not all(os.getenv(k) for k in ["LANGCHAIN_API_KEY", "LANGCHAIN_ENDPOINT", "LANGCHAIN_PROJECT"]):
            st.error("LangSmith environment variables are not fully set. Please check your .env file.")
            st.stop()
    elif provider == "langfuse":
        from langfuse.callback import CallbackHandler
        st.info("Using Langfuse for observability.")
        if not all(os.getenv(k) for k in ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"]):
            st.error("Langfuse environment variables are not fully set. Please check your .env file.")
            st.stop()
        callbacks.append(CallbackHandler())
    else:
        st.info("Observability is disabled.")
    return callbacks

# --- Streamlit UI ---
st.set_page_config(page_title="Agentic Research System", layout="wide")
st.title("Agentic Research System")

# Perform environment variable check at the beginning
check_env_variables()

st.sidebar.header("Configuration")
query = st.sidebar.text_input("Research Query", placeholder="What is the future of AI?")

questions_input = st.sidebar.text_area(
    "Specific Questions (one per line)",
    placeholder="What are the latest trends in AI?\\nHow is AI impacting the job market?",
    height=150
)

observability_provider = st.sidebar.selectbox(
    "Observability Provider",
    options=["None", "Langsmith", "Langfuse"],
    index=0
)

start_research = st.sidebar.button("Start Research")

# --- Main Content Area ---
if start_research:
    if not query:
        st.error("Please enter a research query.")
    else:
        questions = [q.strip() for q in questions_input.split('\n') if q.strip()]
        st.info(f"Starting research for: '{query}'...")
        try:
            callbacks = get_callbacks(observability_provider.lower())
            model = AzureChatOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                openai_api_version=os.environ["OPENAI_API_VERSION"],
                deployment_name=os.environ["OPENAI_API_DEPLOYMENT_NAME"],
                openai_api_key=os.environ["OPENAI_API_KEY"],
                openai_api_type="azure",
            )
            agent = ResearchAgent(model)
            with st.spinner("Research in progress... Please wait."):
                result = agent.run(query, {"callbacks": callbacks}, questions=questions or None)
            st.success("Research complete!")
            st.markdown("---")
            st.header("Research Report")
            st.markdown(result.get("report", "No report generated."))
        except Exception as e:
            st.exception(f"An error occurred during the research process: {e}")
else:
    st.info("Enter a query and click 'Start Research' to begin.")
