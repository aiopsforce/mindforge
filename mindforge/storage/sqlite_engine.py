
import sqlite3
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import sqlite_vec  # For vector operations
from .base_storage import BaseStorage  # Import BaseStorage


class SQLiteEngine(BaseStorage):
    """Enhanced SQLite storage with vector search and multi-level memory."""

    def __init__(self, db_path: str = "mindforge.db", embedding_dim: int = 1536):
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize database with vector search and memory levels."""
        with sqlite3.connect(self.db_path) as conn:
            conn.enable_load_extension(True)  # Enable extension loading
            sqlite_vec.load(conn)  # Load sqlite-vec extension
            conn.enable_load_extension(False) # Disable extension loading

            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")

            conn.executescript(
                f"""
                -- Vector storage
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors 
                USING vec0(embedding float[{self.embedding_dim}]);

                -- Core memories table
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 1,
                    last_access REAL NOT NULL,
                    memory_type TEXT CHECK(
                        memory_type IN ('short_term', 'long_term', 'user', 'session', 'agent')
                    ) NOT NULL,
                    decay_factor REAL DEFAULT 1.0
                );

                -- Concepts table
                CREATE TABLE IF NOT EXISTS concepts (
                    memory_id TEXT NOT NULL,
                    concept TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    PRIMARY KEY (memory_id, concept),
                    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                );

                -- Concept relationships
                CREATE TABLE IF NOT EXISTS concept_graph (
                    source TEXT NOT NULL,
                    target TEXT NOT NULL,
                    weight REAL NOT NULL DEFAULT 1.0,
                    last_updated REAL NOT NULL,
                    PRIMARY KEY (source, target)
                );

                -- User-specific memories
                CREATE TABLE IF NOT EXISTS user_memories (
                    user_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    preference REAL DEFAULT 0.0,
                    history REAL DEFAULT 0.0,
                    PRIMARY KEY (user_id, memory_id),
                    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                );

                -- Session-specific memories
                CREATE TABLE IF NOT EXISTS session_memories (
                    session_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    recent_activity REAL DEFAULT 0.0,
                    context REAL DEFAULT 0.0,
                    PRIMARY KEY (session_id, memory_id),
                    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                );

                -- Agent-specific memories
                CREATE TABLE IF NOT EXISTS agent_memories (
                    memory_id TEXT NOT NULL,
                    knowledge REAL DEFAULT 0.0,
                    adaptability REAL DEFAULT 0.0,
                    PRIMARY KEY (memory_id),
                    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                );

                -- Create optimized indices
                CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
                CREATE INDEX IF NOT EXISTS idx_memories_access ON memories(last_access);
                CREATE INDEX IF NOT EXISTS idx_concepts_concept ON concepts(concept);
                CREATE INDEX IF NOT EXISTS idx_user_memories ON user_memories(user_id);
                CREATE INDEX IF NOT EXISTS idx_session_memories ON session_memories(session_id);
            """
            )

    def store_memory(
        self,
        memory_data: Dict[str, Any],
        memory_type: str = "short_term",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Store memory with appropriate type and metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row  # Use Row factory for dict-like access
            current_time = datetime.now().timestamp()

            # Store core memory
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
                    memory_type,
                    1.0,
                ),
            )

            # Store vector embedding
            embedding_json = json.dumps(memory_data["embedding"].tolist())
            conn.execute(
                """
                INSERT INTO memory_vectors(rowid, embedding)
                VALUES (?, ?)
            """,
                (memory_data["id"], embedding_json),
            )

            # Store concepts
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

            # Store type-specific data
            if memory_type == "user" and user_id:
                conn.execute(
                    """
                    INSERT INTO user_memories (user_id, memory_id, preference, history)
                    VALUES (?, ?, ?, ?)
                """,
                    (
                        user_id,
                        memory_data["id"],
                        memory_data.get("preference", 0.0),
                        memory_data.get("history", 0.0),
                    ),
                )

            elif memory_type == "session" and session_id:
                conn.execute(
                    """
                    INSERT INTO session_memories 
                    (session_id, memory_id, recent_activity, context)
                    VALUES (?, ?, ?, ?)
                """,
                    (
                        session_id,
                        memory_data["id"],
                        memory_data.get("recent_activity", 0.0),
                        memory_data.get("context", 0.0),
                    ),
                )

            elif memory_type == "agent":
                conn.execute(
                    """
                    INSERT INTO agent_memories 
                    (memory_id, knowledge, adaptability)
                    VALUES (?, ?, ?)
                """,
                    (
                        memory_data["id"],
                        memory_data.get("knowledge", 0.0),
                        memory_data.get("adaptability", 0.0),
                    ),
                )

    def retrieve_memories(
        self,
        query_embedding: np.ndarray,
        concepts: List[str],
        memory_type: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories with multi-level context."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            query_vector = json.dumps(query_embedding.tolist())

            # Base query for vector similarity
            base_query = """
                WITH vector_scores AS (
                    SELECT 
                        rowid as memory_id,
                        distance as vector_distance
                    FROM memory_vectors
                    WHERE embedding MATCH ?
                    ORDER BY distance
                    LIMIT ?
                )
            """

            # Add type-specific joins and conditions
            if memory_type == "user" and user_id:
                type_join = """
                    JOIN user_memories um ON m.id = um.memory_id
                    WHERE um.user_id = ?
                """
                params = (query_vector, limit * 2, user_id)  # Increased limit for filtering
            elif memory_type == "session" and session_id:
                type_join = """
                    JOIN session_memories sm ON m.id = sm.memory_id
                    WHERE sm.session_id = ?
                """
                params = (query_vector, limit * 2, session_id)
            elif memory_type == "agent":
                type_join = """
                    JOIN agent_memories am ON m.id = am.memory_id
                """
                params = (query_vector, limit * 2)
            else:
                type_join = ""
                params = (query_vector, limit * 2)

            # Complete query with all components
            query = f"""
                {base_query}
                SELECT 
                    m.*,
                    v.vector_distance,
                    GROUP_CONCAT(c.concept) as concepts,
                    COALESCE(um.preference, 0) as user_preference,
                    COALESCE(um.history, 0) as user_history,
                    COALESCE(sm.recent_activity, 0) as session_activity,
                    COALESCE(sm.context, 0) as session_context,
                    COALESCE(am.knowledge, 0) as agent_knowledge,
                    COALESCE(am.adaptability, 0) as agent_adaptability
                FROM vector_scores v
                JOIN memories m ON m.id = v.memory_id
                LEFT JOIN concepts c ON m.id = c.memory_id
                LEFT JOIN user_memories um ON m.id = um.memory_id  
                LEFT JOIN session_memories sm ON m.id = sm.memory_id
                LEFT JOIN agent_memories am ON m.id = am.memory_id
                {type_join}
                GROUP BY m.id
                ORDER BY v.vector_distance
            """

            results = []
            for row in conn.execute(query, params):
                memory_data = dict(row)
                # Convert comma-separated concepts string to a list
                memory_data["concepts"] = (
                    memory_data.get("concepts", "").split(",")
                    if memory_data.get("concepts")
                    else []
                )

                # Calculate combined relevance score
                vector_similarity = 1 / (
                    1 + memory_data["vector_distance"]
                )  # Convert distance to similarity
                concept_score = self._calculate_concept_score(
                    conn, memory_data["concepts"], concepts
                )

                # Weight different aspects
                memory_data["relevance_score"] = (
                    vector_similarity * 0.4
                    + concept_score * 0.2
                    + memory_data.get("user_preference", 0) * 0.15
                    + memory_data.get("session_activity", 0) * 0.15
                    + memory_data.get("agent_knowledge", 0) * 0.1
                ) * memory_data["decay_factor"]

                results.append(memory_data)

            # Update access patterns
            self._update_access_patterns(conn, [r["id"] for r in results])

            # Sort by relevance and apply final limit
            return sorted(results, key=lambda x: x["relevance_score"], reverse=True)[
                :limit
            ]

    def _calculate_concept_score(
        self,
        conn: sqlite3.Connection,
        memory_concepts: List[str],
        query_concepts: List[str],
    ) -> float:
        """Calculate concept-based relevance score."""
        if not memory_concepts or not query_concepts:
            return 0.0

        activated = self._spread_activation(conn, query_concepts)
        return sum(activated.get(c, 0) for c in memory_concepts) / len(memory_concepts)

    def _spread_activation(
        self,
        conn: sqlite3.Connection,
        initial_concepts: List[str],
        depth: int = 2,
    ) -> Dict[str, float]:
        """Implement spreading activation through concept graph."""
        activated = {concept: 1.0 for concept in initial_concepts}
        seen = set(initial_concepts)

        for _ in range(depth):
            current_concepts = list(activated.keys())
            if not current_concepts:
                break

            # Efficiently retrieve related concepts using SQL
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
                    new_score = current_val * weight * 0.5  # Apply decay
                    new_activations[other_node] = max(
                        new_activations.get(other_node, 0), new_score
                    )
                    seen.add(other_node)

            activated.update(new_activations)  # More efficient update

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
                decay_factor = CASE
                    WHEN memory_type = 'short_term' THEN decay_factor * 1.1  
                    ELSE decay_factor
                END
            WHERE id = ?
        """,
            [(current_time, mid) for mid in memory_ids],
        )
    
    def _update_concept_graph(self, 
                            conn: sqlite3.Connection,
                            concepts: List[str],
                            timestamp: float) -> None:
        """Update concept relationships in graph."""
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i+1:]:
                conn.execute("""
                    INSERT INTO concept_graph (source, target, weight, last_updated)
                    VALUES (?, ?, 1, ?)
                    ON CONFLICT (source, target) DO UPDATE
                    SET weight = weight + 0.1,
                        last_updated = excluded.last_updated
                """, (c1, c2, timestamp))

