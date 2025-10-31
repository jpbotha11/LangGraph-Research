import os
import time
from playwright.sync_api import sync_playwright

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
SCRAPE_TIMEOUT = 30_000
SCRAPE_SLEEP = 4

def run_browser(url: str) -> tuple[str, str]:
    ws_endpoint = os.getenv("BROWSER_WS_ENDPOINT", "ws://127.0.0.1:3001")

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(ws_endpoint)
        page = browser.new_page(user_agent=USER_AGENT)
        page.goto(url, timeout=SCRAPE_TIMEOUT)
        time.sleep(SCRAPE_SLEEP)

        html = page.content()
        text = page.inner_text("body")

        page.close()
        browser.disconnect()
        return html, text


import requests
from typing import List
from langchain.tools import BaseTool

SEARX_URL = os.getenv("SEARX_URL", "http://127.0.0.1:8888")
MAX_SEARCH_RESULTS = 5


class LocalSearxSearchAndScrapeToolBrowseless(BaseTool):
    def __init__(self):
        super().__init__(
            name="LocalSearchAndScrape",
            description="Search local SearXNG and scrape result pages via Browserless",
            args_type={"query": str},
            return_type=List[str],
        )

    def _run(self, query: str) -> List[str]:
        pages: List[str] = []
        try:
            resp = requests.get(
                SEARX_URL, params={"q": query, "format": "json"}, timeout=20
            )
            hits = resp.json().get("results", [])[:MAX_SEARCH_RESULTS]
        except Exception as e:
            return [f"Search failed: {e}"]

        for r in hits:
            url = r.get("url")
            if not url:
                continue
            try:
                print(f"Scraping {url}")
                txt = run_browser(url)[1]
                if isinstance(txt, str) and len(txt.strip()) > 500:
                    pages.append(txt)
                else:
                    pages.append(txt or "")
            except Exception as e:
                pages.append(f"Error scraping {url}: {e}")
        return pages


from langchain_qdrant import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_qdrant_retriever(collection_name: str):
    """
    Returns a Qdrant retriever for the given collection name.
    """
    qdrant = Qdrant(
        client=None,
        collection_name=collection_name,
        embeddings=OpenAIEmbeddings(),
        url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"),
    )
    return qdrant.as_retriever()


def store_documents(collection_name: str, documents: List[str]):
    """
    Stores the given documents in the Qdrant collection.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    docs = text_splitter.create_documents(documents)
    Qdrant.from_documents(
        docs,
        OpenAIEmbeddings(),
        url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"),
        collection_name=collection_name,
    )
