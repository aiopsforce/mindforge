
import numpy as np
import faiss  # Using FAISS directly
from collections import defaultdict
from typing import Dict, List, Any, Optional
from datetime import datetime


class MemoryStore:
    """Enhanced memory store with vector search and multi-level memory."""

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.short_term_memory = []
        self.long_term_memory = []
        self.embeddings = []
        self.timestamps = []
        self.access_counts = []
        self.concepts_list = []

        # Multi-level memory tracking
        self.user_memory = {}
        self.session_memory = {}
        self.agent_memory = {}

        # Initialize the vector store
        self._init_vector_store()

    def _init_vector_store(self):
        """Initialize vector storage components."""
        self.index = faiss.IndexFlatL2(self.dimension)  # FAISS index
        self.semantic_memory = defaultdict(list)
        self.cluster_labels = []

    def add_interaction(
        self, interaction: Dict[str, Any], memory_level: str = "short_term"
    ) -> None:
        """
        Add new interaction with appropriate memory level tracking.
        """
        interaction_id = interaction["id"]
        # Ensure embedding is reshaped correctly
        embedding = np.array(interaction["embedding"]).reshape(1, -1)
        timestamp = interaction.get("timestamp", datetime.now().timestamp())

        # Store in appropriate memory level
        if memory_level == "user":
            self.user_memory[interaction_id] = interaction
        elif memory_level == "session":
            self.session_memory[interaction_id] = interaction
        elif memory_level == "agent":
            self.agent_memory[interaction_id] = interaction

        # Add to short-term memory and FAISS index
        self.short_term_memory.append(interaction)
        self.embeddings.append(embedding)
        self.index.add(embedding)  # Add to FAISS index
        self.timestamps.append(timestamp)
        self.access_counts.append(1)
        self.concepts_list.append(set(interaction.get("concepts", [])))

    def retrieve(
        self,
        query_embedding: np.ndarray,
        query_concepts: List[str],
        memory_level: Optional[str] = None,
        similarity_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories with multi-level context consideration.
        """
        # Ensure query embedding is reshaped correctly
        query_embedding = query_embedding.reshape(1, -1)

        # Get basic vector similarity matches using FAISS
        relevant_interactions = self._get_vector_matches(
            query_embedding, similarity_threshold
        )

        # Apply memory level filtering if specified
        if memory_level:
            relevant_interactions = self._filter_by_memory_level(
                relevant_interactions, memory_level
            )

        # Add concept-based relevance
        relevant_interactions = self._add_concept_relevance(
            relevant_interactions, query_concepts
        )

        return relevant_interactions

    def _get_vector_matches(
        self, query_embedding: np.ndarray, similarity_threshold: float
    ) -> List[Dict[str, Any]]:
        """Get matches based on vector similarity using FAISS."""
        if len(self.embeddings) == 0:
            return []

        # Use FAISS for vector search
        distances, indices = self.index.search(query_embedding, len(self.embeddings))

        matches = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx != -1:  # Valid index
                similarity = 1 / (1 + distance)  # Convert distance to similarity
                if similarity >= similarity_threshold:
                    match = self.short_term_memory[idx].copy()
                    match["similarity"] = similarity
                    matches.append(match)

        return matches

    def _filter_by_memory_level(
        self, interactions: List[Dict[str, Any]], memory_level: str
    ) -> List[Dict[str, Any]]:
        """Filter interactions by memory level."""
        memory_store = {
            "user": self.user_memory,
            "session": self.session_memory,
            "agent": self.agent_memory,
        }.get(memory_level)

        if not memory_store:
            return interactions

        return [
            interaction
            for interaction in interactions
            if interaction["id"] in memory_store
        ]

    def _add_concept_relevance(
        self, interactions: List[Dict[str, Any]], query_concepts: List[str]
    ) -> List[Dict[str, Any]]:
        """Add concept-based relevance scores."""
        activated_concepts = self._spread_activation(query_concepts)

        for interaction in interactions:
            concept_score = sum(
                activated_concepts.get(c, 0) for c in interaction.get("concepts", [])
            )
            # Combine vector similarity with concept score
            interaction["relevance_score"] = (
                interaction["similarity"] * 0.7 + concept_score * 0.3
            )

        return sorted(
            interactions, key=lambda x: x["relevance_score"], reverse=True
        )

    def _spread_activation(
        self, initial_concepts: List[str], depth: int = 2
    ) -> Dict[str, float]:
        """Implement spreading activation for concepts."""
        activated = {concept: 1.0 for concept in initial_concepts}
        seen = set(initial_concepts)

        for _ in range(depth):
            new_activations = {}
            for concept in activated:
                # Get related concepts
                related = self._get_related_concepts(concept)
                for rel_concept, weight in related.items():
                    if rel_concept not in seen:
                        new_score = (
                            activated[concept] * weight * 0.5
                        )  # Added decay factor
                        new_activations[rel_concept] = max(
                            new_activations.get(rel_concept, 0), new_score
                        )
                        seen.add(rel_concept)

            activated.update(new_activations)

        return activated

    def _get_related_concepts(self, concept: str) -> Dict[str, float]:
        """Get concepts related to the given concept."""
        related = {}
        for idx, concepts in enumerate(self.concepts_list):
            if concept in concepts:
                for other in concepts:
                    if other != concept:
                        related[other] = related.get(other, 0) + 1

        # Normalize weights
        total = sum(related.values()) or 1
        return {k: v / total for k, v in related.items()}
