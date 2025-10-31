import streamlit as st
import os
from dotenv import load_dotenv
from graph import ResearchAgent
from langchain_openai import AzureChatOpenAI

# Load environment variables from .env file
load_dotenv()

def get_callbacks(provider: str):
    """Initializes and returns callbacks based on the selected provider."""
    callbacks = []
    if provider == "langsmith":
        # LangSmith is configured automatically via environment variables
        st.info("Using LangSmith for observability.")
    elif provider == "langfuse":
        from langfuse.callback import CallbackHandler
        st.info("Using Langfuse for observability.")
        # Ensure Langfuse environment variables are set
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

st.sidebar.header("Configuration")
query = st.sidebar.text_input("Research Query", placeholder="What is the future of AI?")

questions_input = st.sidebar.text_area(
    "Specific Questions (one per line)",
    placeholder="What are the latest trends in AI?\nHow is AI impacting the job market?",
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
        # Prepare questions list
        questions = [q.strip() for q in questions_input.split('\n') if q.strip()]

        st.info(f"Starting research for: '{query}'...")

        try:
            # Initialize callbacks
            callbacks = get_callbacks(observability_provider.lower())

            # Initialize the Azure OpenAI model
            model = AzureChatOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                openai_api_version=os.environ["OPENAI_API_VERSION"],
                deployment_name=os.environ["OPENAI_API_DEPLOYMENT_NAME"],
                openai_api_key=os.environ["OPENAI_API_KEY"],
                openai_api_type="azure",
            )

            # Initialize and run the research agent
            agent = ResearchAgent(model)
            with st.spinner("Research in progress... Please wait."):
                result = agent.run(query, {"callbacks": callbacks}, questions=questions or None)

            # Display the report
            st.success("Research complete!")
            st.markdown("---")
            st.header("Research Report")
            st.markdown(result.get("report", "No report generated."))

        except Exception as e:
            st.exception(f"An error occurred during the research process: {e}")
else:
    st.info("Enter a query and click 'Start Research' to begin.")
