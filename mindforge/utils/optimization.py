
import numpy as np
from typing import List, Dict, Any
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA, TruncatedSVD


class VectorOptimizer:
    """Utilities for optimizing vector operations."""

    @staticmethod
    def compress_embeddings(
        embeddings: np.ndarray, method: str = "pca", target_dim: int = 256
    ) -> np.ndarray:
        """Compress embeddings to lower dimension."""
        if embeddings.shape[1] <= target_dim:
            return embeddings # Nothing to do.

        if method == "pca":
            pca = PCA(n_components=target_dim)
            return pca.fit_transform(embeddings)
        elif method == "svd":
            # TruncatedSVD works better with sparse matrices, but can also handle dense ones.
            svd = TruncatedSVD(n_components=target_dim)
            return svd.fit_transform(embeddings)
        return embeddings

    @staticmethod
    def quantize_vectors(embeddings: np.ndarray, bits: int = 8) -> np.ndarray:
        """Quantize vectors to reduced precision."""
        min_vals = embeddings.min(axis=0, keepdims=True)  # per-dimension min
        max_vals = embeddings.max(axis=0, keepdims=True)  # per-dimension max

        # Avoid division by zero
        scale = (max_vals - min_vals)
        scale[scale == 0] = 1e-6 # Replace 0s to avoid division by zero

        quantized = np.round(((embeddings - min_vals) / scale) * (2**bits - 1))
        return quantized.astype(np.uint8)
