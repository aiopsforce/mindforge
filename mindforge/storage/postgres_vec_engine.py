from __future__ import annotations

import numpy as np
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.extras import Json
from pgvector.psycopg2 import register_vector

from .base_storage import BaseStorage


class PostgresVectorEngine(BaseStorage):
    """Storage engine using PostgreSQL with pgvector."""

    def __init__(self, dsn: str, embedding_dim: int = 1536):
        self.dsn = dsn
        self.embedding_dim = embedding_dim
        self._initialize_db()

    def _initialize_db(self) -> None:
        with psycopg2.connect(self.dsn) as conn:
            register_vector(conn)
            cur = conn.cursor()
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    prompt TEXT,
                    response TEXT,
                    memory_type TEXT,
                    embedding vector({self.embedding_dim})
                )
                """
            )
            conn.commit()

    def store_memory(
        self,
        memory_data: Dict[str, Any],
        memory_type: str = "short_term",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        with psycopg2.connect(self.dsn) as conn:
            register_vector(conn)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO memories (id, prompt, response, memory_type, embedding) VALUES (%s, %s, %s, %s, %s)",
                (
                    memory_data["id"],
                    memory_data.get("prompt"),
                    memory_data.get("response"),
                    memory_type,
                    memory_data["embedding"].tolist(),
                ),
            )
            conn.commit()

    def retrieve_memories(
        self,
        query_embedding: np.ndarray,
        concepts: List[str],
        memory_type: str = None,
        user_id: str = None,
        session_id: str = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        with psycopg2.connect(self.dsn) as conn:
            register_vector(conn)
            cur = conn.cursor()
            sql = "SELECT id, prompt, response, memory_type, embedding <-> %s AS score FROM memories"
            params = [query_embedding.tolist()]
            if memory_type:
                sql += " WHERE memory_type = %s"
                params.append(memory_type)
            sql += " ORDER BY embedding <-> %s LIMIT %s"
            params.extend([query_embedding.tolist(), limit])
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "prompt": r[1],
                "response": r[2],
                "memory_type": r[3],
                "relevance_score": r[4],
            }
            for r in rows
        ]

    def update_memory_level(
        self,
        memory_id: str,
        new_memory_level: str,
        user_id: str = None,
        session_id: str = None,
    ) -> bool:
        with psycopg2.connect(self.dsn) as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE memories SET memory_type = %s WHERE id = %s",
                (new_memory_level, memory_id),
            )
            updated = cur.rowcount
            conn.commit()
        return updated > 0
