
import pytest
from unittest.mock import MagicMock
import numpy as np
from mindforge.core.memory_manager import MemoryManager
from mindforge.core.memory_types import MemoryType
from mindforge.config import AppConfig

class TestMemoryLevels:
    @pytest.fixture
    def mock_components(self):
        chat_model = MagicMock()
        chat_model.extract_concepts.return_value = ["concept1", "concept2"]
        chat_model.generate_response.return_value = "Mock response"

        embedding_model = MagicMock()
        embedding_model.get_embedding.return_value = np.array([0.1] * 1536)

        storage_engine = MagicMock()
        storage_engine.retrieve_memories.return_value = []

        config = AppConfig()

        return chat_model, embedding_model, storage_engine, config

    @pytest.fixture
    def memory_manager(self, mock_components):
        chat_model, embedding_model, storage_engine, config = mock_components
        return MemoryManager(
            chat_model=chat_model,
            embedding_model=embedding_model,
            storage_engine=storage_engine,
            config=config
        )

    def test_all_memory_types_storage(self, memory_manager, mock_components):
        """
        Test that all memory types can be passed to process_input and stored correctly.
        """
        _, _, storage_engine, _ = mock_components

        test_cases = [
            (MemoryType.SHORT_TERM, {}),
            (MemoryType.LONG_TERM, {}),
            (MemoryType.USER_SPECIFIC, {"preference": 1.0, "history": 1.0}),
            (MemoryType.SESSION_SPECIFIC, {"recent_activity": 1.0, "context": 1.0}),
            (MemoryType.AGENT_SPECIFIC, {"knowledge": 1.0, "adaptability": 1.0}),
            (MemoryType.PERSONA, {"relationship_strength": 1.0, "interaction_style": "friendly"}),
            (MemoryType.TOOLBOX, {"tool_schema": {}, "usage_count": 0}),
            (MemoryType.CONVERSATION, {"message_role": "user"}),
            (MemoryType.WORKFLOW, {"outcome": "success", "duration": 0.0}),
            (MemoryType.EPISODIC, {"significance": 0.8, "location": "chat_interface"}),
            (MemoryType.AGENT_REGISTRY, {"capabilities": [], "status": "active"}),
            (MemoryType.ENTITY, {"entity_type": "unknown", "attributes": {}}),
        ]

        for memory_type, expected_metadata_subset in test_cases:
            query = f"Query for {memory_type.name}"

            memory_manager.process_input(query=query, memory_type=memory_type)

            # Verify store_memory was called
            assert storage_engine.store_memory.called

            # Get the arguments passed to store_memory
            call_args = storage_engine.store_memory.call_args
            args, kwargs = call_args
            memory_data = args[0]
            passed_memory_type = kwargs.get('memory_type')

            # Verify memory type is passed as string
            assert passed_memory_type == memory_type.value

            # Verify metadata
            for key, value in expected_metadata_subset.items():
                assert key in memory_data
                if key == "token_count": # Dynamic value
                     continue
                assert memory_data[key] == value

            # Reset mock for next iteration
            storage_engine.store_memory.reset_mock()

    def test_process_input_handles_string_memory_type(self, memory_manager, mock_components):
        """
        Test that process_input still handles string memory types for backward compatibility.
        """
        _, _, storage_engine, _ = mock_components

        memory_manager.process_input(query="test", memory_type="custom_type")

        assert storage_engine.store_memory.called
        _, kwargs = storage_engine.store_memory.call_args
        assert kwargs['memory_type'] == "custom_type"

    def test_build_context_integration(self, memory_manager):
        """
        Test that _build_context handles the specific keys associated with different memory types.
        """
        # Mock memories with keys specific to our memory types
        memories = [
            {"user_preference": 0.9, "user_history": 0.5},
            {"session_activity": 0.8, "session_context": 0.4},
            {"agent_knowledge": 0.7, "agent_adaptability": 0.3},
            {"relationship_strength": 0.6}, # Should be safe even if not explicitly handled in _build_context yet
        ]

        context = memory_manager._build_context(memories, "query")

        assert context["user_context"]["preference"] == 0.9
        assert context["session_context"]["recent_activity"] == 0.8
        assert context["agent_context"]["knowledge"] == 0.7
        # Verify it didn't crash on unknown keys
