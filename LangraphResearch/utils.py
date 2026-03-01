import os
import time
from playwright.sync_api import sync_playwright
import requests
from typing import List
from langchain.tools import BaseTool
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
from langchain_core.embeddings import Embeddings
import io
import pypdf


USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
SCRAPE_TIMEOUT = 30_000
SCRAPE_SLEEP = 4
MAX_SEARCH_RESULTS = 30


# ── Embedder ──────────────────────────────────────────────────────────────────

class LMStudioEmbedder(Embeddings):
    """
    Generates embeddings via LM Studio.
    Inherits from LangChain Embeddings for full compatibility.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model or os.getenv("LMSTUDIO_MODEL", "text-embedding-nomic-embed-text-v1.5")
        self.base_url = (base_url or os.getenv("LMSTUDIO_URL", "")).rstrip("/")

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(
            f"{self.base_url}/v1/embeddings",
            json={"model": self.model, "input": texts},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    # Required by LangChain Embeddings base class
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_one(text)

    @property
    def vector_size(self) -> int:
        return len(self.embed_one("test"))


# ── Qdrant Store ──────────────────────────────────────────────────────────────

class QdrantStore:
    """Stores and retrieves vectors in a Qdrant collection."""

    def __init__(
        self,
        collection_name: str,
        vector_size: int,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
        distance: Distance = Distance.COSINE,
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = QdrantClient(
            url=qdrant_url or os.getenv("QDRANT_URL", ""),
            api_key=qdrant_api_key,
        )
        self._ensure_collection(distance)

    def _ensure_collection(self, distance: Distance):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=distance),
            )
            print(f"Created collection '{self.collection_name}'")
        else:
            print(f"Using existing collection '{self.collection_name}'")

    def upsert(
        self,
        vectors: list[list[float]],
        documents: list[str],
        metadata: list[dict] | None = None,
    ) -> int:
        if metadata is None:
            metadata = [{} for _ in documents]
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": doc, **meta},
            )
            for vector, doc, meta in zip(vectors, documents, metadata)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"Upserted {len(points)} points into '{self.collection_name}'")
        return len(points)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: float | None = None,
    ):
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
        )
        return results.points


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_embeddings() -> LMStudioEmbedder:
    return LMStudioEmbedder()


def get_qdrant_retriever(collection_name: str):
    embeddings = get_embeddings()
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=collection_name,
        url=os.getenv("QDRANT_URL", ""),
        content_payload_key="text",  # <-- tell LangChain where the text is
    )
    return vector_store.as_retriever()

def store_documents(collection_name: str, documents: List[str], batch_size: int = 32):
    """Chunk, embed, and store documents into Qdrant."""
    embedder = get_embeddings()
    store = QdrantStore(collection_name=collection_name, vector_size=embedder.vector_size)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = [chunk.page_content for chunk in text_splitter.create_documents(documents)]

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        print(f"Embedding batch {i // batch_size + 1} ({i}-{min(i + batch_size, len(chunks)) - 1})...")
        vectors = embedder.embed(batch)
        store.upsert(vectors, batch)


def store_documents(collection_name: str, documents: List[str], source_url: str = None, batch_size: int = 32):
    embedder = get_embeddings()
    store = QdrantStore(collection_name=collection_name, vector_size=embedder.vector_size)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = [chunk.page_content for chunk in text_splitter.create_documents(documents)]

    meta = {"source_url": source_url} if source_url else {}

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        print(f"Embedding batch {i // batch_size + 1} ({i}-{min(i + batch_size, len(chunks)) - 1})...")
        vectors = embedder.embed(batch)
        store.upsert(vectors, batch, metadata=[meta for _ in batch])


# ── Browser / Scraper ─────────────────────────────────────────────────────────

def run_browser(url: str) -> tuple[str, str]:
    ws_endpoint = os.getenv("BROWSER_WS_ENDPOINT")
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(ws_endpoint)
        page = browser.new_page(user_agent=USER_AGENT)
        page.goto(url, timeout=SCRAPE_TIMEOUT)
        time.sleep(SCRAPE_SLEEP)
        html = page.content()
        text = page.inner_text("body")
        page.close()
        return html, text

def is_pdf_url(url: str, response_headers: dict) -> bool:
    """Detect PDF by URL extension or content-type header."""
    content_type = response_headers.get("content-type", "")
    return url.lower().endswith(".pdf") or "application/pdf" in content_type


def extract_text_from_pdf(url: str) -> str:
    """Download and extract text from a PDF, using Browserless to bypass 403s."""
    try:
        # First try direct download
        resp = requests.get(url, timeout=30, headers={"User-Agent": USER_AGENT}, allow_redirects=True)
        pdf_bytes = resp.content if resp.status_code == 200 else None
    except Exception:
        pdf_bytes = None

    # Fall back to Browserless if direct download failed
    if not pdf_bytes or len(pdf_bytes) < 100:
        print(f"Direct download failed, fetching via browser: {url}")
        pdf_bytes = fetch_pdf_via_browser(url)

    if not pdf_bytes:
        return f"Could not retrieve PDF: {url}"

    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages.append(f"[Page {i + 1}]\n{text}")
        return "\n\n".join(pages) if pages else "PDF had no extractable text."
    except Exception as e:
        return f"Failed to parse PDF: {e}"


def fetch_pdf_via_browser(url: str) -> bytes | None:
    ws_endpoint = os.getenv("BROWSER_WS_ENDPOINT")
    try:
        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp(ws_endpoint)
            context = browser.new_context(user_agent=USER_AGENT)
            page = context.new_page()

            # Visit the base domain first to get cookies/session
            base_url = "/".join(url.split("/")[:3])
            page.goto(base_url, timeout=SCRAPE_TIMEOUT, wait_until="networkidle")
            time.sleep(2)

            # Now fetch the PDF using the browser's own request context
            # This shares cookies, TLS fingerprint, and headers with the browser session
            api_response = context.request.get(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/pdf,*/*",
                    "Referer": base_url,
                },
                timeout=30000,
            )

            page.close()

            if api_response.status == 200:
                return api_response.body()
            else:
                print(f"API request failed with status {api_response.status}")
                return None

    except Exception as e:
        print(f"Browser PDF fetch failed: {e}")
        return None

def is_url_indexed(store: QdrantStore, url: str) -> bool:
    """Check if a URL has already been scraped and stored."""
    results = store.client.scroll(
        collection_name=store.collection_name,
        scroll_filter={
            "must": [
                {"key": "source_url", "match": {"value": url}}
            ]
        },
        limit=1,
    )
    return len(results[0]) > 0


class LocalSearxSearchAndScrapeToolBrowseless(BaseTool):
    def __init__(self):
        super().__init__(
            name="LocalSearchAndScrape",
            description="Search local SearXNG and scrape result pages via Browserless",
            args_type={"query": str},
            return_type=List[str],
        )

    def _run(self, query: str, collection_name: str) -> List[str]:

        embedder = get_embeddings()
        store = QdrantStore(collection_name=collection_name, vector_size=embedder.vector_size)
        pages: List[str] = []

        try:
            resp = requests.get(
                os.getenv("SEARX_URL"), params={"q": query, "format": "json"}, timeout=20
            )
            hits = resp.json().get("results", [])[:MAX_SEARCH_RESULTS]
        except Exception as e:
            return [f"Search failed: {e}"]

        for r in hits:
            url = r.get("url")
            if not url:
                continue

            if is_url_indexed(store, url):
                print(f"Skipping {url} - already indexed")
                continue

            try:
                # --- Detect PDF ---
                head = requests.head(url, timeout=10, headers={"User-Agent": USER_AGENT})
                if is_pdf_url(url, dict(head.headers)):
                    print(f"PDF detected: {url}")
                    txt = extract_text_from_pdf(url)
                else:
                    print(f"Scraping {url}")
                    txt = run_browser(url)[1]

                if isinstance(txt, str) and len(txt.strip()) > 500:
                    store_documents(collection_name, [txt], source_url=url)
                    pages.append(txt)
                else:
                    pages.append(txt or "")

            except Exception as e:
                pages.append(f"Error processing {url}: {e}")

        return pages