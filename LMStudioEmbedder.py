import requests
import os

#config
LMSTUDIO_URL = os.environ["LMSTUDIO_URL"]

class LMStudioEmbedder:
    """Generates embeddings using LM Studio's OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "text-embedding-nomic-embed-text-v1.5",
        base_url: str = LMSTUDIO_URL,
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
