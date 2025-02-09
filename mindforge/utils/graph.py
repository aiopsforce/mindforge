import sqlite3
from datetime import datetime
from typing import Dict


class ConceptGraph:
    """Utility class for managing concept relationships."""

    def __init__(self, sqlite_engine):
        self.engine = sqlite_engine

    def add_relationship(self, concept1: str, concept2: str, weight: float = 1.0):
        """Add or update a relationship between concepts."""
        with sqlite3.connect(self.engine.db_path) as conn:
            current_time = datetime.now().timestamp()
            conn.execute(
                """
                INSERT INTO concept_graph (source, target, weight, last_updated)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (source, target) DO UPDATE
                SET weight = weight + ?,
                    last_updated = ?
            """,
                (concept1, concept2, weight, current_time, weight * 0.1, current_time),
            )

    def get_related_concepts(self, concept: str, min_weight: float = 0.1
    ) -> Dict[str, float]:
        """Get concepts related to the given concept."""
        with sqlite3.connect(self.engine.db_path) as conn:
            related = {}
            for row in conn.execute(
                """
                SELECT 
                    CASE 
                        WHEN source = ? THEN target 
                        ELSE source 
                    END as related_concept,
                    weight
                FROM concept_graph
                WHERE (source = ? OR target = ?)
                AND weight >= ?
            """,
                (concept, concept, concept, min_weight),
            ):
                related[row[0]] = row[1]
            return related