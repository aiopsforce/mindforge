from __future__ import annotations

import numpy as np
from typing import List, Dict, Any, Optional
import redis
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema

from .base_storage import BaseStorage


class RedisVectorEngine(BaseStorage):
    """Storage engine using RedisVL for vector search."""

    def __init__(
        self, redis_url: str = "redis://localhost:6379", embedding_dim: int = 1536
    ):
        self.redis = redis.from_url(redis_url)
        self.embedding_dim = embedding_dim
        schema_dict = {
            "index": {"name": "mindforge", "prefix": "memory:"},
            "fields": [
                {"name": "id", "type": "tag"},
                {"name": "prompt", "type": "text"},
                {"name": "response", "type": "text"},
                {"name": "memory_type", "type": "tag"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {"algorithm": "hnsw", "dims": embedding_dim},
                },
            ],
        }
        schema = IndexSchema.from_dict(schema_dict)
        self.index = SearchIndex(schema, redis_client=self.redis)
        try:
            self.index.create(overwrite=False, drop=False)
        except Exception:
            pass

    def store_memory(
        self,
        memory_data: Dict[str, Any],
        memory_type: str = "short_term",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        doc = {
            "id": memory_data["id"],
            "prompt": memory_data.get("prompt"),
            "response": memory_data.get("response"),
            "memory_type": memory_type,
            "embedding": memory_data["embedding"].tolist(),
        }
        self.index.load([doc])

    def retrieve_memories(
        self,
        query_embedding: np.ndarray,
        concepts: List[str],
        memory_type: str = None,
        user_id: str = None,
        session_id: str = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        query = self.index.query.vector(
            query_embedding.tolist(), "embedding", num_results=limit
        )
        if memory_type:
            query = query.filter(f"@memory_type:{{{memory_type}}}")
        results = self.index.query(query)
        memories = []
        for r in results:
            memory = {
                "id": r["id"],
                "prompt": r.get("prompt"),
                "response": r.get("response"),
                "memory_type": r.get("memory_type"),
                "relevance_score": r.get("vector_score", 0.0),
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
        key = f"memory:{memory_id}"
        if not self.redis.exists(key):
            return False
        self.redis.hset(key, "memory_type", new_memory_level)
        return True
