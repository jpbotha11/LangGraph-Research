from __future__ import annotations
from typing import List, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain_core.documents import Document
from utils import LocalSearxSearchAndScrapeToolBrowseless, store_documents, get_qdrant_retriever
import uuid

class AgentState(TypedDict):
    messages: List[BaseMessage]
    documents: List[Document]
    collection_name: str
    # The questions to answer. These can be user-provided or generated.
    questions: List[str]
    answers: List[str]
    report: str
    # User-provided questions, if any.
    user_questions: List[str] | None

class ResearchAgent:
    def __init__(self, model: AzureChatOpenAI):
        self.model = model
        self.search_tool = LocalSearxSearchAndScrapeToolBrowseless()
        self.graph = self._build_graph()

    def _build_graph(self):
        """Builds the graph for the research agent."""
        graph = StateGraph(AgentState)

        # Define the nodes
        graph.add_node("search", self.search_node)
        graph.add_node("store_documents", self.store_documents_node)
        graph.add_node("generate_questions", self.generate_questions_node)
        graph.add_node("generate_answers", self.generate_answers_node)
        graph.add_node("generate_report", self.generate_report_node)

        # Define the edges and entry point
        graph.set_entry_point("search")
        graph.add_edge("search", "store_documents")

        # Conditional edge: if user provides questions, skip generation
        graph.add_conditional_edges(
            "store_documents",
            self.should_generate_questions,
            {
                "generate": "generate_questions",
                "skip": "generate_answers",
            },
        )

        graph.add_edge("generate_questions", "generate_answers")
        graph.add_edge("generate_answers", "generate_report")
        graph.add_edge("generate_report", END)

        return graph.compile()

    def should_generate_questions(self, state: AgentState) -> Literal["generate", "skip"]:
        """Determines whether to generate questions or use user-provided ones."""
        if state.get("user_questions"):
            print("--- Using user-provided questions ---")
            return "skip"
        else:
            print("--- Generating questions ---")
            return "generate"

    def search_node(self, state: AgentState):
        query = state["messages"][-1].content
        collection_name = str(uuid.uuid4())
        documents = self.search_tool._run(query,collection_name)
        

        return {"documents": documents, "collection_name": collection_name}

    def store_documents_node(self, state: AgentState):
        store_documents(state["collection_name"], state["documents"])
        # If user questions exist, pass them to the 'questions' field for the next step
        if state.get("user_questions"):
            return {"questions": state["user_questions"]}
        return {}

    def generate_questions_node(self, state: AgentState):
        retriever = get_qdrant_retriever(state["collection_name"])
        q = state["messages"][-1].content
        documents = retriever.invoke(q)
        prompt = f"Based on the following documents, please generate a comma separated list of 3 relevant questions that can be answered from the documents:\\n\\n{documents}. Only return a list of questions."
        response = self.model.invoke([HumanMessage(content=prompt)])
        questions = response.content.strip().split(",")
        return {"questions": questions}

    def generate_answers_node(self, state: AgentState):
        retriever = get_qdrant_retriever(state["collection_name"])
        answers = []
        for question in state["questions"]:
            if not question:
                continue
            documents = retriever.invoke(question)
            prompt = f"Based on the following documents, please answer the question: {question}\\n\\n{documents}"
            response = self.model.invoke([HumanMessage(content=prompt)])
            answers.append(response.content)
        return {"answers": answers}

    def generate_report_node(self, state: AgentState):
        prompt = f"Generate a report based on the following questions and answers:\\n\\n"
        for q, a in zip(state["questions"], state["answers"]):
            prompt += f"Q: {q}\\nA: {a}\\n\\n"
        prompt += "The report should contain a summary of what was researched as well as the important topics on the research."
        response = self.model.invoke([HumanMessage(content=prompt)])
        return {"report": response.content}

    def run(self, query: str, config: dict = None, questions: List[str] | None = None):
        return self.graph.invoke(
            {"messages": [HumanMessage(content=query)], "user_questions": questions},
            config=config,
        )
