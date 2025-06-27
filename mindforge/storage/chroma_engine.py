from __future__ import annotations

import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

from chromadb import Client
from chromadb.config import Settings

from .base_storage import BaseStorage


class ChromaDBEngine(BaseStorage):
    """Storage engine backed by ChromaDB."""

    def __init__(self, collection_name: str = "mindforge", embedding_dim: int = 1536):
        self.embedding_dim = embedding_dim
        self.client = Client(Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(collection_name)

    def store_memory(
        self,
        memory_data: Dict[str, Any],
        memory_type: str = "short_term",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        metadata = {
            "prompt": memory_data.get("prompt"),
            "response": memory_data.get("response"),
            "memory_type": memory_type,
        }
        if user_id:
            metadata["user_id"] = user_id
        if session_id:
            metadata["session_id"] = session_id

        self.collection.add(  # type: ignore[arg-type]
            ids=[memory_data["id"]],
            embeddings=[memory_data["embedding"].tolist()],
            metadatas=[metadata],
        )

    def retrieve_memories(
        self,
        query_embedding: np.ndarray,
        concepts: List[str],
        memory_type: str = None,
        user_id: str = None,
        session_id: str = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        metadata_filter = {}
        if memory_type:
            metadata_filter["memory_type"] = memory_type
        if user_id:
            metadata_filter["user_id"] = user_id
        if session_id:
            metadata_filter["session_id"] = session_id

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=limit,
            where=metadata_filter or None,
        )

        memories = []
        for idx, mem_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][idx]
            memory = {
                "id": mem_id,
                "prompt": meta.get("prompt"),
                "response": meta.get("response"),
                "memory_type": meta.get("memory_type"),
                "relevance_score": results["distances"][0][idx],
            }
            memories.append(memory)
        return memories

    def update_memory_level(
        self,
        memory_id: str,
        new_memory_level: str,
        user_id: str = None,
        session_id: str = None,
    ) -> bool:
        try:
            results = self.collection.get(ids=[memory_id])
        except Exception:
            return False

        if not results["ids"]:
            return False

        metadata = results["metadatas"][0]
        metadata["memory_type"] = new_memory_level
        if user_id:
            metadata["user_id"] = user_id
        if session_id:
            metadata["session_id"] = session_id

        self.collection.update(ids=[memory_id], metadatas=[metadata])
        return True
