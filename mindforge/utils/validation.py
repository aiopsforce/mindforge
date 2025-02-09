
from typing import Any, Dict, List
import numpy as np


class InputValidator:
    """Validate inputs to the system."""

    @staticmethod
    def validate_embedding(embedding: np.ndarray, expected_dim: int) -> bool:
        """Validate embedding format and dimension."""
        if not isinstance(embedding, np.ndarray):
            return False
        if embedding.ndim != 1:
            return False
        if embedding.shape[0] != expected_dim:
            return False
        if not np.isfinite(embedding).all():
            return False
        return True

    @staticmethod
    def validate_memory_content(content: Dict[str, Any]) -> List[str]:
        """Validate memory content format."""
        errors = []
        required_fields = ["id", "prompt", "response", "embedding", "timestamp"] # Added id

        for field in required_fields:
            if field not in content:
                errors.append(f"Missing required field: {field}")

        if "prompt" in content and not isinstance(content["prompt"], str):
            errors.append("Field 'prompt' must be a string")
        if "response" in content and not isinstance(content["response"], str):
            errors.append("Field 'response' must be a string")

        if "timestamp" in content and not isinstance(
            content["timestamp"], (int, float)
        ):
            errors.append("Field 'timestamp' must be a number")

        if "embedding" in content and not isinstance(content["embedding"], np.ndarray):
            errors.append("Field 'embedding' must be a numpy array")

        return errors
