import os
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from graph import ResearchAgent
from langchain_openai import AzureChatOpenAI

# Load environment variables from .env file
load_dotenv()

def check_env_variables():
    """Checks if all required environment variables are set and exits if not."""
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
        print(
            "Error: The following required environment variables are not set: "
            f"`{', '.join(missing_vars)}`. Please check your `.env` file."
        )
        sys.exit(1)

def get_callbacks(provider: str):
    """Initializes and returns callbacks based on the selected provider."""
    callbacks = []
    if provider == "langsmith":
        print("Using LangSmith for observability.")
        if not all(os.getenv(k) for k in ["LANGCHAIN_API_KEY", "LANGCHAIN_ENDPOINT", "LANGCHAIN_PROJECT"]):
            print("Error: LangSmith environment variables are not fully set. Please check your .env file.")
            sys.exit(1)
    elif provider == "langfuse":
        from langfuse.callback import CallbackHandler
        print("Using Langfuse for observability.")
        if not all(os.getenv(k) for k in ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"]):
            print("Error: Langfuse environment variables are not fully set. Please check your .env file.")
            sys.exit(1)
        callbacks.append(CallbackHandler())
    else:
        print("Observability is disabled.")
    return callbacks

def main():
    """Main execution function for the research agent."""
    # Perform environment variable check at the beginning
    check_env_variables()

    observability_provider = os.getenv("OBSERVABILITY_PROVIDER", "none").lower()
    callbacks = get_callbacks(observability_provider)

    model = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        openai_api_version=os.environ["OPENAI_API_VERSION"],
        deployment_name=os.environ["OPENAI_API_DEPLOYMENT_NAME"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_api_type="azure",
    )

    agent = ResearchAgent(model)
    query = "hi"  # Example query

    print(f"Starting research for: '{query}'...")
    result = agent.run(query, {"callbacks": callbacks})

    print("\\n--- Research Report ---")
    print(result.get("report", "No report generated."))

if __name__ == "__main__":

   




    main()
