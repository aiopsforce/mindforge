
import sqlite3
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path  # For path manipulation
import sqlite_vec  # For vector operations
from .base_storage import BaseStorage  # Import the BaseStorage class

class SQLiteVecEngine(BaseStorage):
    """Enhanced SQLite storage engine using sqlite-vec for vector search."""

    def __init__(self, db_path: str = "mindforge.db", embedding_dim: int = 1536):
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize database with vector search capabilities."""
        with sqlite3.connect(self.db_path) as conn:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)


            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")

            # Create optimized schema
            conn.executescript(
                f"""
                -- Vector table for embeddings
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors 
                USING vec0(embedding float[{self.embedding_dim}]);

                -- Core memory table
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 1,
                    last_access REAL NOT NULL,
                    memory_type TEXT CHECK(memory_type IN ('short_term', 'long_term')) NOT NULL,
                    decay_factor REAL DEFAULT 1.0
                );

                -- Concept associations
                CREATE TABLE IF NOT EXISTS concepts (
                    memory_id TEXT NOT NULL,
                    concept TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    PRIMARY KEY (memory_id, concept),
                    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                );

                -- Concept graph
                CREATE TABLE IF NOT EXISTS concept_graph (
                    source TEXT NOT NULL,
                    target TEXT NOT NULL,
                    weight REAL NOT NULL DEFAULT 1.0,
                    last_updated REAL NOT NULL,
                    PRIMARY KEY (source, target)
                );

                -- Optimized indices
                CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
                CREATE INDEX IF NOT EXISTS idx_memories_access ON memories(last_access);
                CREATE INDEX IF NOT EXISTS idx_concepts_concept ON concepts(concept);
            """
            )

    def store_memory(
        self,
        memory_data: Dict[str, Any],
        memory_type: str = "short_term",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None: # Implemented store memory
        """Add new memory with vector embedding."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            current_time = datetime.now().timestamp()

            # Insert memory data
            conn.execute(
                """
                INSERT INTO memories (
                    id, prompt, response, timestamp,
                    access_count, last_access, memory_type,
                    decay_factor
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    memory_data["id"],
                    memory_data["prompt"],
                    memory_data["response"],
                    current_time,
                    1,
                    current_time,
                    memory_type,  # Use provided memory_type
                    1.0,
                ),
            )

            # Insert vector embedding
            embedding_json = json.dumps(memory_data["embedding"].tolist())
            conn.execute(
                """
                INSERT INTO memory_vectors(rowid, embedding)
                VALUES (?, ?)
            """,
                (memory_data["id"], embedding_json),
            )

            # Insert concepts
            if "concepts" in memory_data:
                conn.executemany(
                    """
                    INSERT INTO concepts (memory_id, concept, weight)
                    VALUES (?, ?, ?)
                """,
                    [(memory_data["id"], c, 1.0) for c in memory_data["concepts"]],
                )

                # Update concept graph
                self._update_concept_graph(conn, memory_data["concepts"], current_time)


    def retrieve_memories(
        self,
        query_embedding: np.ndarray,
        concepts: List[str],
        memory_type: str = None,  # Added memory_type
        user_id: str = None,  # Added user_id
        session_id: str = None,  # Added session_id
        limit: int = 10,
        similarity_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories using vector similarity and concept matching."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row  # Use Row factory
            current_time = datetime.now().timestamp()
            query_vector = json.dumps(query_embedding.tolist())

            # Get vector similarity matches
            vector_matches = conn.execute(
                """
                WITH vector_scores AS (
                    SELECT 
                        rowid as memory_id,
                        distance as vector_distance
                    FROM memory_vectors
                    WHERE embedding MATCH ?
                    ORDER BY distance
                    LIMIT ?
                )
                SELECT 
                    m.*,
                    v.vector_distance,
                    GROUP_CONCAT(c.concept) as concepts
                FROM vector_scores v
                JOIN memories m ON m.id = v.memory_id
                LEFT JOIN concepts c ON m.id = c.memory_id
                GROUP BY m.id
                ORDER BY v.vector_distance
            """,
                (query_vector, limit * 2),
            ).fetchall()  # Fetch all results

            # Get concept activations
            activated_concepts = self._spread_activation(conn, concepts)

            # Calculate final scores and filter results
            results = []
            for row in vector_matches:
                memory_data = dict(row)  # Convert Row to dict
                # Convert comma-separated concepts string to a list
                memory_data["concepts"] = (memory_data.get("concepts") or "").split(",")

                # Calculate concept score
                concept_score = sum(
                    activated_concepts.get(c, 0) for c in memory_data["concepts"]
                ) / max(len(memory_data["concepts"]), 1)

                # Calculate vector similarity score (convert distance to similarity)
                vector_similarity = 1 / (1 + memory_data["vector_distance"])

                # Combined score with decay
                memory_data["relevance_score"] = (
                    vector_similarity * 0.7 + concept_score * 0.3
                ) * memory_data["decay_factor"]

                if memory_data["relevance_score"] >= similarity_threshold:
                    results.append(memory_data)

            # Update access patterns for retrieved memories
            self._update_access_patterns(conn, [r["id"] for r in results])

            # Sort and limit the results based on relevance score
            return sorted(results, key=lambda x: x["relevance_score"], reverse=True)[
                :limit
            ]

    def _spread_activation(
        self, conn: sqlite3.Connection, initial_concepts: List[str], depth: int = 2
    ) -> Dict[str, float]:
        """Implement spreading activation through concept graph."""
        activated = {concept: 1.0 for concept in initial_concepts}
        seen = set(initial_concepts)

        for _ in range(depth):
            current_concepts = list(activated.keys())
            if not current_concepts:
                break

            # Batch retrieve related concepts
            placeholders = ",".join("?" * len(current_concepts))
            related = conn.execute(
                f"""
                SELECT source, target, weight
                FROM concept_graph
                WHERE source IN ({placeholders})
                OR target IN ({placeholders})
            """,
                current_concepts + current_concepts,
            ).fetchall()

            new_activations = {}
            for source, target, weight in related:
                if source in activated:
                    current_val = activated[source]
                    other_node = target
                else:
                    current_val = activated[target]
                    other_node = source

                if other_node not in seen:
                    new_score = current_val * weight * 0.5  # Decay factor
                    new_activations[other_node] = max(
                        new_activations.get(other_node, 0), new_score
                    )
                    seen.add(other_node)

            activated.update(new_activations)

        return activated

    def _update_access_patterns(
        self, conn: sqlite3.Connection, memory_ids: List[str]
    ) -> None:
        """Update memory access patterns."""
        current_time = datetime.now().timestamp()
        conn.executemany(
            """
            UPDATE memories 
            SET access_count = access_count + 1,
                last_access = ?,
                decay_factor = decay_factor * 1.1
            WHERE id = ?
        """,
            [(current_time, mid) for mid in memory_ids],
        )

    def _update_concept_graph(
        self, conn: sqlite3.Connection, concepts: List[str], timestamp: float
    ) -> None:
        """Update concept relationships in graph."""
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i + 1 :]:
                conn.execute(
                    """
                    INSERT INTO concept_graph (source, target, weight, last_updated)
                    VALUES (?, ?, 1, ?)
                    ON CONFLICT (source, target) DO UPDATE
                    SET weight = weight + 0.1,  
                        last_updated = excluded.last_updated
                """,
                    (c1, c2, timestamp),
                )
