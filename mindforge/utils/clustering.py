import sqlite3
import numpy as np
from sklearn.cluster import KMeans
import json


class MemoryClustering:
    """Utility class for semantic clustering of memories."""

    def __init__(self, sqlite_engine):
        self.engine = sqlite_engine

    def cluster_memories(self, n_clusters: int = 10):
        """Cluster memories based on their embeddings."""
        with sqlite3.connect(self.engine.db_path) as conn:
            # Get all embeddings
            embeddings = []
            memory_ids = []
            for row in conn.execute(
                """
                SELECT rowid, embedding FROM memory_vectors
            """
            ):
                memory_ids.append(row[0])
                embeddings.append(json.loads(row[1]))  # Load embedding from JSON

            if not embeddings:
                return

            # Perform clustering
            embeddings_array = np.array(embeddings)
            kmeans = KMeans(
                n_clusters=min(n_clusters, len(embeddings)),  # Ensure n_clusters <= num samples
                init='k-means++',
                max_iter=300,
                n_init=10,
                random_state=0
            )
            labels = kmeans.fit_predict(embeddings_array)

            #  There is no semantic_clusters table.
            # Store concept <-> memory links instead.
            conn.executemany("""
                INSERT INTO concepts (memory_id, concept)
                VALUES (?, ?)
                ON CONFLICT (memory_id, concept) DO NOTHING
            """, [(memory_id, f"cluster_{label}") for memory_id, label in zip(memory_ids, labels)])
