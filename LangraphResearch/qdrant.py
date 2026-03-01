from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import requests
import uuid


class LMStudioEmbedder:
    """Generates embeddings using LM Studio's OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "text-embedding-nomic-embed-text-v1.5",
        base_url: str = "",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, returns a list of vectors."""
        response = requests.post(
            f"{self.base_url}/v1/embeddings",
            json={"model": self.model, "input": texts},
        )
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]

    def embed_one(self, text: str) -> list[float]:
        """Convenience method to embed a single string."""
        return self.embed([text])[0]

    @property
    def vector_size(self) -> int:
        """Detect vector size by running a test embedding."""
        return len(self.embed_one("test"))


class QdrantStore:
    """Stores and retrieves vectors in a Qdrant collection."""

    def __init__(
        self,
        collection_name: str,
        vector_size: int,
        qdrant_url: str = "",
        qdrant_api_key: str | None = None,
        distance: Distance = Distance.COSINE,
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self._ensure_collection(distance)

    def _ensure_collection(self, distance: Distance):
        """Create the collection if it does not already exist."""
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
        """Upsert vectors with their source text and optional metadata."""
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
        """Search for similar vectors, returns Qdrant ScoredPoint results."""
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
        )


def embed_documents_to_qdrant(
    documents: list[str],
    collection_name: str,
    embedder: LMStudioEmbedder | None = None,
    store: QdrantStore | None = None,
    batch_size: int = 32,
    metadata: list[dict] | None = None,
) -> int:
    """
    Convenience function that wires LMStudioEmbedder and QdrantStore together.
    Creates default instances if none are provided.
    """
    embedder = embedder or LMStudioEmbedder()
    store = store or QdrantStore(
        collection_name=collection_name,
        vector_size=embedder.vector_size,
    )

    total = len(documents)
    all_vectors = []

    for i in range(0, total, batch_size):
        batch = documents[i : i + batch_size]
        print(f"Embedding batch {i // batch_size + 1} ({i}-{min(i + batch_size, total) - 1})...")
        all_vectors.extend(embedder.embed(batch))

    return store.upsert(all_vectors, documents, metadata)