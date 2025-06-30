
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
        # user_memory stores user_id -> list of interaction_ids
        self.user_memory = defaultdict(list)
        # session_memory stores session_id -> list of interaction_ids
        self.session_memory = defaultdict(list)
        # agent_memory stores a list of interaction_ids
        self.agent_memory = []
        # Additional memory types
        self.persona_memory = []
        self.toolbox_memory = []
        self.conversation_memory = []
        self.workflow_memory = []
        self.episodic_memory = []
        self.registry_memory = []
        self.entity_memory = []

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

        # Add to short-term memory and FAISS index (main storage for indexed interactions)
        self.short_term_memory.append(interaction)
        # Associated metadata lists are updated in sync with short_term_memory
        self.embeddings.append(embedding) # Appending the raw embedding
        self.index.add(embedding)  # Add to FAISS index
        self.timestamps.append(timestamp)
        self.access_counts.append(1) # Default access count
        self.concepts_list.append(set(interaction.get("concepts", [])))

        # Store interaction_id in appropriate memory level for filtering
        if memory_level == "user":
            if "user_id" in interaction:
                self.user_memory[interaction["user_id"]].append(interaction_id)
            else:
                # Handle missing user_id, perhaps log a warning or error
                print(f"Warning: user_id missing for user-level memory interaction {interaction_id}")
        elif memory_level == "session":
            if "session_id" in interaction:
                self.session_memory[interaction["session_id"]].append(interaction_id)
            else:
                # Handle missing session_id
                print(f"Warning: session_id missing for session-level memory interaction {interaction_id}")
        elif memory_level == "agent":
            self.agent_memory.append(interaction_id)
        elif memory_level == "long_term":
            # long_term_memory stores the full interaction, not just ID
            self.long_term_memory.append(interaction)
        elif memory_level == "persona":
            self.persona_memory.append(interaction_id)
        elif memory_level == "toolbox":
            self.toolbox_memory.append(interaction_id)
        elif memory_level == "conversation":
            self.conversation_memory.append(interaction_id)
        elif memory_level == "workflow":
            self.workflow_memory.append(interaction_id)
        elif memory_level == "episodic":
            self.episodic_memory.append(interaction_id)
        elif memory_level == "registry":
            self.registry_memory.append(interaction_id)
        elif memory_level == "entity":
            self.entity_memory.append(interaction_id)
        # No specific action for "short_term" as it's already in short_term_memory by default

    def retrieve(
        self,
        query_embedding: np.ndarray,
        query_concepts: List[str],
        memory_level: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        similarity_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories with multi-level context consideration.
        """
        # Ensure query embedding is reshaped correctly
        query_embedding = query_embedding.reshape(1, -1)

        # Get basic vector similarity matches using FAISS
        # These matches are from the global short_term_memory pool
        relevant_interactions = self._get_vector_matches(
            query_embedding, similarity_threshold
        )

        # Apply memory level filtering if specified
        if memory_level:
            relevant_interactions = self._filter_by_memory_level(
                relevant_interactions, memory_level, user_id, session_id
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
        self,
        interactions: List[Dict[str, Any]],
        memory_level: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Filter interactions by memory level using interaction IDs."""
        allowed_ids = set()
        perform_filtering = True

        if memory_level == "user":
            if user_id and user_id in self.user_memory:
                allowed_ids = set(self.user_memory[user_id])
            else:
                if not user_id:
                    print("Warning: user_id not provided for user-level memory filtering.")
                else:
                    print(f"Warning: No user memory found for user_id {user_id}.")
                return [] # Return empty if requested user memory level but no user_id or no data
        elif memory_level == "session":
            if session_id and session_id in self.session_memory:
                allowed_ids = set(self.session_memory[session_id])
            else:
                if not session_id:
                    print("Warning: session_id not provided for session-level memory filtering.")
                else:
                    print(f"Warning: No session memory found for session_id {session_id}.")
                return [] # Return empty if requested session memory level but no session_id or no data
        elif memory_level == "agent":
            allowed_ids = set(self.agent_memory)
        elif memory_level == "long_term":
            # Assuming long_term_memory stores full interaction objects
            allowed_ids = {lt_interaction["id"] for lt_interaction in self.long_term_memory}
        elif memory_level == "persona":
            allowed_ids = set(self.persona_memory)
        elif memory_level == "toolbox":
            allowed_ids = set(self.toolbox_memory)
        elif memory_level == "conversation":
            allowed_ids = set(self.conversation_memory)
        elif memory_level == "workflow":
            allowed_ids = set(self.workflow_memory)
        elif memory_level == "episodic":
            allowed_ids = set(self.episodic_memory)
        elif memory_level == "registry":
            allowed_ids = set(self.registry_memory)
        elif memory_level == "entity":
            allowed_ids = set(self.entity_memory)
        elif memory_level == "short_term":
            # Interactions are already from short_term_memory via FAISS, so no ID filtering needed here.
            perform_filtering = False
        else:
            print(f"Warning: Unknown memory_level '{memory_level}' specified. No filtering applied.")
            perform_filtering = False

        if perform_filtering:
            if not allowed_ids and memory_level in [
                "user",
                "session",
                "agent",
                "long_term",
                "persona",
                "toolbox",
                "conversation",
                "workflow",
                "episodic",
                "registry",
                "entity",
            ]:
                # If allowed_ids is empty for these specific levels, it means no relevant memories exist.
                return []
            return [
                interaction for interaction in interactions if interaction["id"] in allowed_ids
            ]
        else:
            return interactions

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

    # --- Memory Promotion Methods ---

    def promote_to_long_term(self, interaction_id: str) -> bool:
        """
        Promotes an interaction from short-term memory to long-term memory.
        The interaction remains in short-term memory and FAISS index.
        Long-term memory is an additive store.

        Args:
            interaction_id: The ID of the interaction to promote.

        Returns:
            True if the interaction was promoted, False if it was already
            in long-term memory or not found in short-term memory.
        """
        # Check if already in long-term memory
        for lt_interaction in self.long_term_memory:
            if lt_interaction["id"] == interaction_id:
                return False  # Already exists

        # Find in short-term memory and add to long-term memory
        for st_interaction in self.short_term_memory:
            if st_interaction["id"] == interaction_id:
                self.long_term_memory.append(st_interaction.copy()) # Add a copy
                return True

        return False # Not found in short-term memory

    def promote_session_to_user(self, interaction_id: str, user_id: str) -> bool:
        """
        Promotes an interaction ID from session-level tracking to user-level tracking.
        The interaction itself remains in short_term_memory and FAISS.

        Args:
            interaction_id: The ID of the interaction to promote.
            user_id: The ID of the user to associate the interaction with.

        Returns:
            True if the interaction ID was added to the user's memory,
            False if it was already there, or if the interaction_id was not
            found in any session, or if user_id is invalid.
        """
        if not user_id:
            print("Warning: user_id cannot be empty for promote_session_to_user.")
            return False

        # Check if the interaction_id exists in any session
        found_in_session = False
        for session_interactions in self.session_memory.values():
            if interaction_id in session_interactions:
                found_in_session = True
                break
        
        if not found_in_session:
            # Optionally, one might also check if interaction_id is valid (i.e., in short_term_memory)
            # but the primary check here is if it's part of *any* session.
            # If it's not in any session, there's nothing to "promote" from session to user.
            return False

        # Check if already in the target user's memory
        if interaction_id in self.user_memory[user_id]:
            return False

        self.user_memory[user_id].append(interaction_id)
        return True

    def mark_as_agent_knowledge(self, interaction_id: str) -> bool:
        """
        Marks an interaction ID as agent-level knowledge.
        The interaction itself remains in short_term_memory and FAISS.

        Args:
            interaction_id: The ID of the interaction to mark.

        Returns:
            True if the interaction ID was marked as agent knowledge,
            False if it was already marked or if the interaction_id is invalid
            (e.g. not found in short_term_memory, though this check is optional
            as agent_memory is just a list of IDs).
        """
        # Optional: Check if interaction_id is valid (exists in short_term_memory)
        # For this implementation, we'll assume interaction_id is valid if provided.
        # A more robust check would be:
        # if not any(st_interaction["id"] == interaction_id for st_interaction in self.short_term_memory):
        #     print(f"Warning: Interaction ID {interaction_id} not found in short-term memory.")
        #     return False

        if interaction_id in self.agent_memory:
            return False  # Already marked

        self.agent_memory.append(interaction_id)
        return True
