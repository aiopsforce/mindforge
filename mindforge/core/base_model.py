from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of embeddings produced by this model."""
        pass

    @abstractmethod
    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for the given text."""
        pass


class BaseChatModel(ABC):
    """Abstract base class for chat models."""

    @abstractmethod
    def generate_response(self, context: Dict[str, Any], query: str) -> str:
        """Generate a response given context and query."""
        pass

    @abstractmethod
    def extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from the text."""
        pass
