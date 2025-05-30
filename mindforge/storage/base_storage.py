from abc import ABC, abstractmethod
from typing import Dict, List, Any
import numpy as np


class BaseStorage(ABC):  # Moved to its own file
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

    @abstractmethod
    def update_memory_level(
        self,
        memory_id: str,
        new_memory_level: str,
        user_id: str = None, # Required if new_memory_level is 'user'
        session_id: str = None # Required if new_memory_level is 'session'
    ) -> bool:
        """Update the level/type of an existing memory and its associations."""
        pass