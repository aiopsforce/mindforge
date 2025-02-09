import numpy as np
import requests
from ...core.base_model import BaseEmbeddingModel


class OllamaEmbeddingModel(BaseEmbeddingModel):
    def __init__(
        self, model_name: str = "llama2", base_url: str = "http://localhost:11434"
    ):
        self.model_name = model_name
        self.base_url = base_url
        self._dimension = self._get_dimension()

    def _get_dimension(self) -> int:
        """Get embedding dimension by testing with a sample text."""
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model_name, "prompt": "test"},
            )
            response.raise_for_status()
            return len(response.json()["embedding"])
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama: {e}")
        except KeyError:
            raise ValueError("Could not determine embedding dimension from Ollama response.")

    @property
    def dimension(self) -> int:
        return self._dimension

    def get_embedding(self, text: str) -> np.ndarray:
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model_name, "prompt": text},
            )
            response.raise_for_status()
            return np.array(response.json()["embedding"])
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama: {e}")
        except KeyError:
            raise ValueError("Could not retrieve embedding from Ollama response")