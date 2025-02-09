
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any


class BaseStorage(ABC):
    """Abstract base class for storage implementations."""

    @abstractmethod
    def store_memory(
        self,
        memory_data: Dict[str, Any],
        memory_type: str = "short_term",
        user_id: str = None,
        session_id: str = None,
    ) -> None:
        """Store memory with appropriate type and metadata."""
        pass

    @abstractmethod
    def retrieve_memories(
        self,
        query_embedding: np.ndarray,
        concepts: List[str],
        memory_type: str = None,
        user_id: str = None,
        session_id: str = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories with multi-level context."""
        pass


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
