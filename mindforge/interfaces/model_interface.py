from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np


class ModelInterface(ABC):
    """Interface for AI model operations."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding."""
        pass

    @abstractmethod
    def generate(
        self, prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate response."""
        pass

    @abstractmethod
    def analyze(self, content: str, analysis_type: str) -> Dict[str, Any]:
        """Analyze content."""
        pass