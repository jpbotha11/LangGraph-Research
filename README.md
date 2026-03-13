# Agentic Research System

This project is a Python-based agentic system that uses LangGraph to perform web research, store findings in a vector database, and generate comprehensive reports. It is powered by an Azure OpenAI LLM and features a Streamlit-based web interface for easy interaction.

## Features

-   **Automated Web Research**: Provide a query and the agent will search the web to gather relevant information.
-   **Vector Database Storage**: Uses Qdrant to store and retrieve scraped web content efficiently.
-   **Customizable Questions**: You can provide a list of specific questions for the agent to answer, or let it generate them automatically.
-   **Report Generation**: Generates a final report that includes a summary of the research and answers to the key questions.
-   **Interactive Frontend**: A user-friendly web interface built with Streamlit allows for easy configuration and execution of the research tasks.
-   **Flexible Observability**: Integrated with both Langsmith and Langfuse, which can be selected via a feature flag for monitoring and debugging.

## Setup

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Create a Virtual Environment

It is recommended to use a virtual environment to manage project dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\\Scripts\\activate`
```

### 3. Install Dependencies

All required `pip` packages are listed in the `requirements.txt` file. Install them with the following command:

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

The application requires several environment variables for API keys and service endpoints.

1.  Make a copy of the example configuration file:
    ```bash
    cp .env.example .env
    ```
2.  Open the `.env` file and fill in the placeholder values with your actual credentials for the following services:
    -   **Observability**: Choose between `langsmith`, `langfuse`, or `none`. Configure the corresponding API keys and endpoints.
    -   **Azure OpenAI**: Provide your endpoint, API version, deployment name, and API key.
    -   **Services**: Set the URLs for your Qdrant instance, SearxNG search provider, and the Browserless WebSocket endpoint.

## Usage

The primary way to interact with the research agent is through the Streamlit web interface.

### Running the Frontend

1.  Ensure your virtual environment is activated and all dependencies are installed.
2.  Run the following command in your terminal:
    ```bash
    streamlit run app.py
    ```
3.  Open a web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

### Interacting with the UI

Once the application is running, you can use the sidebar to configure and start your research:

1.  **Research Query**: Enter the main topic or question you want the agent to research.
2.  **Specific Questions**: (Optional) Provide a list of specific questions you want the agent to answer, with one question per line. If you leave this blank, the agent will generate its own questions based on the research content.
3.  **Observability Provider**: Select your preferred observability tool (`Langsmith`, `Langfuse`, or `None`) from the dropdown menu.
4.  Click the **Start Research** button to begin the process.

The main content area will show the status of the research and display the final report once it is complete.
