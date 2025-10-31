import os
from dotenv import load_dotenv
load_dotenv()

from graph import ResearchAgent
from langchain_openai import AzureChatOpenAI

def main():
    observability_provider = os.getenv("OBSERVABILITY_PROVIDER", "none").lower()
    callbacks = []

    if observability_provider == "langsmith":
        # LangSmith is configured automatically via environment variables
        print("Using LangSmith for observability.")
    elif observability_provider == "langfuse":
        from langfuse.callback import CallbackHandler

        print("Using Langfuse for observability.")
        # Langfuse is configured via environment variables:
        # LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
        handler = CallbackHandler()
        callbacks.append(handler)
    else:
        print("Observability is disabled.")

    model = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        openai_api_version=os.environ["OPENAI_API_VERSION"],
        deployment_name=os.environ["OPENAI_API_DEPLOYMENT_NAME"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_api_type="azure",
    )

    agent = ResearchAgent(model)
    query = "What is langgraph?"
    result = agent.run(query, {"callbacks": callbacks})
    print(result["report"])

if __name__ == "__main__":
    main()
