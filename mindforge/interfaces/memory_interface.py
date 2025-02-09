from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np


class MemoryInterface(ABC):
    """Interface for memory operations."""

    @abstractmethod
    def add(
        self,
        content: Dict[str, Any],
        memory_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add new memory content."""
        pass

    @abstractmethod
    def retrieve(
        self, query: str, embedding: np.ndarray, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories."""
        pass

    @abstractmethod
    def update(self, memory_id: str, updates: Dict[str, Any]) -> None:
        """Update existing memory."""
        pass

    @abstractmethod
    def delete(self, memory_id: str) -> None:
        """Delete memory."""
        pass