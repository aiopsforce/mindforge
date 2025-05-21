
from typing import Optional, Dict, Any, List
import uuid
from datetime import datetime
from .base_model import BaseChatModel, BaseEmbeddingModel
from ..storage.base_storage import BaseStorage # Import BaseStorage
from ..storage.sqlite_engine import SQLiteEngine
from ..utils.graph import ConceptGraph
from ..utils.clustering import MemoryClustering
from ..config import AppConfig  # Import AppConfig
import numpy as np


class MemoryManager:
    """
    Coordinates multi-level memory management, vector search, and AI model responses.
    """

    def __init__(
        self,
        chat_model: BaseChatModel,
        embedding_model: BaseEmbeddingModel,
        storage_engine: BaseStorage, # Changed parameter
        config: AppConfig, # config might still be needed for other things
    ):
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.config = config  # Store the config
        self.storage = storage_engine # Use the passed-in engine
        self.concept_graph = ConceptGraph(self.storage) # Uses the passed-in engine
        self.clustering = MemoryClustering(self.storage) # Uses the passed-in engine
        self.interactions_since_last_clustering = 0

    def process_input(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        memory_type: str = "short_term",
    ) -> str:
        """
        Process user input with multi-level memory context.
        """
        # Generate embedding and extract concepts
        query_embedding = self.embedding_model.get_embedding(query)
        concepts = self.chat_model.extract_concepts(query)

        # Retrieve relevant memories
        memories = self.storage.retrieve_memories(
            query_embedding=query_embedding,
            concepts=concepts,
            memory_level=memory_type, # Changed from memory_type to memory_level
            user_id=user_id,
            session_id=session_id,
            # No specific limit from config is mentioned for retrieval in this task,
            # but SQLiteEngine.retrieve_memories has a default limit of 10.
        )

        # Filter memories by similarity threshold
        if memories: # Ensure there are memories to filter
            filtered_memories = [
                mem for mem in memories
                if mem.get('relevance_score', 0) >= self.config.memory.similarity_threshold
            ]
        else:
            filtered_memories = []

        # Build context for response generation
        context = self._build_context(filtered_memories, query)

        # Generate response
        response = self.chat_model.generate_response(context, query)

        # Store the interaction
        self._store_interaction(
            query=query,
            response=response,
            embedding=query_embedding,
            concepts=concepts,
            memory_type=memory_type,
            user_id=user_id,
            session_id=session_id,
        )

        return response

    def _build_context(
        self, memories: List[Dict[str, Any]], query: str
    ) -> Dict[str, Any]:
        """
        Build comprehensive context from retrieved memories.
        """
        context = {
            "query": query,
            "relevant_memories": memories[:5],  # Top 5 most relevant
            "user_context": {},
            "session_context": {},
            "agent_context": {},
        }

        # Aggregate context by memory type, handling missing keys gracefully
        for memory in memories:
            if memory.get("user_preference") is not None:
                context["user_context"].update(
                    {
                        "preference": memory.get("user_preference", 0.0),
                        "history": memory.get("user_history", 0.0),
                    }
                )
            if memory.get("session_activity") is not None:
                context["session_context"].update(
                    {
                        "recent_activity": memory.get("session_activity", 0.0),
                        "context": memory.get("session_context", 0.0),
                    }
                )
            if memory.get("agent_knowledge") is not None:
                context["agent_context"].update(
                    {
                        "knowledge": memory.get("agent_knowledge", 0.0),
                        "adaptability": memory.get("agent_adaptability", 0.0),
                    }
                )

        return context

    def _store_interaction(
        self,
        query: str,
        response: str,
        embedding: np.ndarray,
        concepts: List[str],
        memory_type: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Store interaction with appropriate memory type and metadata.
        """
        memory_data = {
            "id": str(uuid.uuid4()),
            "prompt": query,
            "response": response,
            "embedding": embedding,
            "concepts": concepts,
            "timestamp": datetime.now().timestamp(),
        }

        # Add user_id and session_id to memory_data if available
        if user_id:
            memory_data['user_id'] = user_id
        if session_id:
            memory_data['session_id'] = session_id

        # Add type-specific metadata, handling optional values
        if memory_type == "user":
            memory_data.update(
                {"preference": 1.0, "history": 1.0}
            )  # No need to check for None
        elif memory_type == "session":
            memory_data.update({"recent_activity": 1.0, "context": 1.0})
        elif memory_type == "agent":
            memory_data.update({"knowledge": 1.0, "adaptability": 1.0})

        # Store the memory
        self.storage.store_memory(
            memory_data,
            memory_level=memory_type, # Changed from memory_type to memory_level
            user_id=user_id, # user_id and session_id are passed explicitly for MemoryStore.add_interaction's direct params
            session_id=session_id, # and also included in memory_data for its internal logic
        )

        self.interactions_since_last_clustering += 1
        if self.interactions_since_last_clustering >= self.config.memory.clustering_trigger_threshold:
            self.clustering.cluster_memories()
            self.interactions_since_last_clustering = 0