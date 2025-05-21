
import requests
from typing import Dict, Any, List
from ...core.base_model import BaseChatModel
from ...utils.errors import ModelError


class OllamaChatModel(BaseChatModel):
    def __init__(self, model_name: str = "llama2", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url

    def generate_response(self, context: Dict[str, Any], query: str) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": f"Context: {context}\nQuery: {query}",
                    "stream": False,
                },
            )
            response.raise_for_status()  # Raise HTTPError for bad requests (4xx or 5xx)
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama: {e}")


    def extract_concepts(self, text: str) -> List[str]:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": f"Extract key concepts from this text, separated by commas: {text}",
                    "stream": False,
                },
            )
            response.raise_for_status()
            # Handle potential extra spaces and empty strings.  Also handle cases where response isn't a simple comma separated list.
            concepts_raw = response.json().get("response")
            if concepts_raw is None:
                return []
            concepts = concepts_raw.split(",")
            return [c.strip() for c in concepts if c.strip()]
        except requests.exceptions.RequestException as e:
            raise ModelError(f"Ollama API error during concept extraction: {e}")
        except ValueError:  # JSONDecodeError is a subclass of ValueError
            raise ModelError(f"Failed to decode JSON response from Ollama during concept extraction: {e}")