import numpy as np
import litellm
from typing import Dict, Any
from ...core.base_model import BaseEmbeddingModel
from ...utils.errors import ModelError


class LiteLLMEmbeddingModel(BaseEmbeddingModel):
    """Embedding model wrapper using litellm."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        dimension: int = 1536,
        extra_params: Dict[str, Any] | None = None,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self._dimension = dimension
        self.extra_params = extra_params or {}

    @property
    def dimension(self) -> int:
        return self._dimension

    def _call_embedding(self, text: str):
        return litellm.embedding(
            model=self.model_name,
            input=[text],
            api_key=self.api_key,
            api_base=self.base_url,
            **self.extra_params,
        )

    def get_embedding(self, text: str) -> np.ndarray:
        try:
            resp = self._call_embedding(text)
            item = resp.data[0]
            embedding = item["embedding"] if isinstance(item, dict) else item.embedding
            return np.array(embedding)
        except Exception as e:
            raise ModelError(f"LiteLLM embedding error: {e}")
