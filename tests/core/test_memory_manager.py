import pytest
from unittest.mock import MagicMock, call
import numpy as np

from mindforge.core.memory_manager import MemoryManager
from mindforge.config import AppConfig, MemoryConfig, VectorConfig, ModelConfig, StorageConfig # For creating a mock config

# Define a consistent embedding dimension for tests, matching SQLiteEngine tests if possible
TEST_EMBEDDING_DIM = 10

@pytest.fixture
def mock_chat_model():
    model = MagicMock()
    model.extract_concepts.return_value = ["mock_concept1", "mock_concept2"]
    model.generate_response.return_value = "Mocked LLM response"
    return model

@pytest.fixture
def mock_embedding_model():
    model = MagicMock()
    model.get_embedding.return_value = np.array([0.5] * TEST_EMBEDDING_DIM).astype(float)
    model.dimension = TEST_EMBEDDING_DIM # Ensure the mock model has the dimension attribute
    return model

@pytest.fixture
def mock_storage_engine():
    engine = MagicMock()
    # Sample retrieved memory. retrieve_memories should return a list.
    mock_memory = {
        "id": "retrieved_mem_1",
        "prompt": "Retrieved prompt",
        "response": "Retrieved response",
        "embedding": np.array([0.4] * TEST_EMBEDDING_DIM).astype(float),
        "concepts": ["retrieved_concept"],
        "relevance_score": 0.9, # Important for similarity_threshold filtering
        "memory_type": "short_term"
    }
    engine.retrieve_memories.return_value = [mock_memory]
    engine.store_memory.return_value = None
    return engine

@pytest.fixture
def mock_app_config():
    config = AppConfig(
        memory=MemoryConfig(similarity_threshold=0.7, clustering_trigger_threshold=5), # Example values
        vector=VectorConfig(embedding_dim=TEST_EMBEDDING_DIM), # Ensure this matches mock_embedding_model.dimension
        model=ModelConfig(), # Defaults are fine for most MemoryManager tests not directly using model config details
        storage=StorageConfig() # Defaults are fine
    )
    return config

@pytest.fixture
def memory_manager(mock_chat_model, mock_embedding_model, mock_storage_engine, mock_app_config):
    """Fixture to create a MemoryManager instance with mocked dependencies."""
    manager = MemoryManager(
        chat_model=mock_chat_model,
        embedding_model=mock_embedding_model,
        storage_engine=mock_storage_engine,
        config=mock_app_config
    )
    return manager

def test_process_input_basic_flow(
    memory_manager, mock_chat_model, mock_embedding_model, mock_storage_engine, mock_app_config
):
    """Test the basic flow of the process_input method."""
    test_query = "What is MindForge?"
    user_id = "test_user"
    session_id = "test_session"
    memory_type = "short_term" # Default in process_input if not specified

    # Call the method to test
    final_response = memory_manager.process_input(
        query=test_query,
        user_id=user_id,
        session_id=session_id,
        memory_type=memory_type
    )

    # Assertions
    # 1. Embedding model was called
    mock_embedding_model.get_embedding.assert_called_once_with(test_query)

    # 2. Chat model extracted concepts
    mock_chat_model.extract_concepts.assert_called_once_with(test_query)

    # 3. Storage engine retrieved memories
    mock_storage_engine.retrieve_memories.assert_called_once_with(
        query_embedding=mock_embedding_model.get_embedding.return_value,
        concepts=mock_chat_model.extract_concepts.return_value,
        memory_type=memory_type,
        user_id=user_id,
        session_id=session_id
    )
    
    # 4. Context was built (implicitly tested by generate_response call if context is an arg)
    #    _build_context uses the result of retrieve_memories.
    #    The filtering by similarity_threshold happens before _build_context.
    #    Our mock memory has relevance_score 0.9, and threshold is 0.7, so it should pass.
    retrieved_memories = mock_storage_engine.retrieve_memories.return_value
    filtered_memories_for_context = [
        mem for mem in retrieved_memories 
        if mem.get('relevance_score', 0) >= mock_app_config.memory.similarity_threshold
    ]


    # 5. Chat model generated response
    #    The context passed to generate_response includes query and filtered memories.
    #    We can check the call to generate_response. Its first arg is context.
    #    The context is a dictionary, so we need to be careful with asserting the whole dict.
    #    We can assert that it was called and that parts of the context are as expected.
    assert mock_chat_model.generate_response.call_count == 1
    args, kwargs = mock_chat_model.generate_response.call_args
    context_arg = args[0] # First positional argument is context
    assert context_arg['query'] == test_query
    assert context_arg['relevant_memories'] == filtered_memories_for_context[:5] # _build_context takes top 5

    # 6. Storage engine stored the interaction
    #    The store_memory call happens with the generated embedding, concepts, response etc.
    mock_storage_engine.store_memory.assert_called_once()
    store_args, store_kwargs = mock_storage_engine.store_memory.call_args_list[0]
    
    # store_args is a tuple (memory_data_dict, ), store_kwargs is a dict for other named args
    # Example: call(memory_data, memory_type='short_term', user_id='test_user', session_id='test_session')
    # So memory_data_dict is store_args[0]
    # And other params are in store_kwargs
    
    stored_memory_data_dict = store_args[0]

    assert stored_memory_data_dict["prompt"] == test_query
    assert stored_memory_data_dict["response"] == mock_chat_model.generate_response.return_value
    assert np.array_equal(stored_memory_data_dict["embedding"], mock_embedding_model.get_embedding.return_value)
    assert stored_memory_data_dict["concepts"] == mock_chat_model.extract_concepts.return_value
    assert stored_memory_data_dict["user_id"] == user_id # Ensure user_id is in memory_data
    assert stored_memory_data_dict["session_id"] == session_id # Ensure session_id is in memory_data
    
    assert store_kwargs["memory_type"] == memory_type
    assert store_kwargs["user_id"] == user_id
    assert store_kwargs["session_id"] == session_id


    # 7. Final response is from chat_model.generate_response
    assert final_response == mock_chat_model.generate_response.return_value

def test_process_input_memory_filtering_below_threshold(
    memory_manager, mock_chat_model, mock_embedding_model, mock_storage_engine, mock_app_config
):
    """Test that memories below similarity_threshold are filtered out."""
    # Modify storage engine mock to return a memory with low relevance score
    low_score_memory = {
        "id": "low_score_mem_1", "prompt": "Low score prompt", "response": "Low score response",
        "embedding": np.array([0.3] * TEST_EMBEDDING_DIM), "concepts": ["low_score"],
        "relevance_score": 0.5, # Below threshold of 0.7
        "memory_type": "short_term"
    }
    mock_storage_engine.retrieve_memories.return_value = [low_score_memory]

    test_query = "Another query"
    memory_manager.process_input(query=test_query)

    # Check context passed to generate_response
    args, kwargs = mock_chat_model.generate_response.call_args
    context_arg = args[0]
    assert context_arg['relevant_memories'] == [] # Should be empty after filtering

def test_clustering_triggered_after_threshold(
    memory_manager, mock_chat_model, mock_embedding_model, mock_storage_engine, mock_app_config
):
    """Test that clustering is triggered after interactions reach the threshold."""
    # Set threshold to a small number for testing
    mock_app_config.memory.clustering_trigger_threshold = 3
    
    # Re-initialize manager with this specific config if fixture doesn't pick it up
    # Or, ensure fixture uses the modified mock_app_config
    manager = MemoryManager( # Re-initialize for this test with specific threshold
        chat_model=mock_chat_model,
        embedding_model=mock_embedding_model,
        storage_engine=mock_storage_engine,
        config=mock_app_config # This config now has threshold = 3
    )

    # Access clustering mock through the manager instance
    mock_clustering_instance = manager.clustering = MagicMock() # Replace the actual clustering with a mock

    for i in range(mock_app_config.memory.clustering_trigger_threshold):
        manager.process_input(query=f"Query {i+1}")
        if i < mock_app_config.memory.clustering_trigger_threshold - 1:
            mock_clustering_instance.cluster_memories.assert_not_called()
        else:
            mock_clustering_instance.cluster_memories.assert_called_once()
    
    # Check if counter reset
    assert manager.interactions_since_last_clustering == 0

    # One more call to ensure it's not called again immediately
    mock_clustering_instance.cluster_memories.reset_mock()
    manager.process_input(query="Query after reset")
    mock_clustering_instance.cluster_memories.assert_not_called()
    assert manager.interactions_since_last_clustering == 1

# --- New tests for multi-level memory ---

def test_process_input_user_memory(
    memory_manager, mock_chat_model, mock_embedding_model, mock_storage_engine, mock_app_config
):
    """Test process_input with memory_type='user'."""
    test_query = "User specific query"
    user_id = "user123"
    session_id = "session_for_user_query" # Can be present or None
    memory_type = "user"

    # Expected embedding and concepts
    expected_embedding = mock_embedding_model.get_embedding.return_value
    expected_concepts = mock_chat_model.extract_concepts.return_value
    expected_response = mock_chat_model.generate_response.return_value

    memory_manager.process_input(
        query=test_query,
        user_id=user_id,
        session_id=session_id,
        memory_type=memory_type
    )

    # Assert retrieve_memories call
    mock_storage_engine.retrieve_memories.assert_called_once_with(
        query_embedding=expected_embedding,
        concepts=expected_concepts,
        memory_type=memory_type,
        user_id=user_id,
        session_id=session_id
    )

    # Assert store_memory call
    mock_storage_engine.store_memory.assert_called_once()
    args, kwargs = mock_storage_engine.store_memory.call_args
    
    memory_data_dict = args[0]
    assert memory_data_dict["prompt"] == test_query
    assert memory_data_dict["response"] == expected_response
    assert np.array_equal(memory_data_dict["embedding"], expected_embedding)
    assert memory_data_dict["concepts"] == expected_concepts
    assert memory_data_dict["user_id"] == user_id
    if session_id: # session_id is optional in memory_data if not defining characteristic of level
        assert memory_data_dict["session_id"] == session_id
    else:
        assert "session_id" not in memory_data_dict # Or assert it's None if it's always added

    assert kwargs["memory_type"] == memory_type
    assert kwargs["user_id"] == user_id
    assert kwargs["session_id"] == session_id


def test_process_input_session_memory(
    memory_manager, mock_chat_model, mock_embedding_model, mock_storage_engine, mock_app_config
):
    """Test process_input with memory_type='session'."""
    test_query = "Session specific query"
    user_id = "user_for_session_query" # Can be present or None
    session_id = "session789"
    memory_type = "session"

    expected_embedding = mock_embedding_model.get_embedding.return_value
    expected_concepts = mock_chat_model.extract_concepts.return_value
    expected_response = mock_chat_model.generate_response.return_value
    
    memory_manager.process_input(
        query=test_query,
        user_id=user_id,
        session_id=session_id,
        memory_type=memory_type
    )

    mock_storage_engine.retrieve_memories.assert_called_once_with(
        query_embedding=expected_embedding,
        concepts=expected_concepts,
        memory_type=memory_type,
        user_id=user_id,
        session_id=session_id
    )

    mock_storage_engine.store_memory.assert_called_once()
    args, kwargs = mock_storage_engine.store_memory.call_args
    
    memory_data_dict = args[0]
    assert memory_data_dict["prompt"] == test_query
    assert memory_data_dict["response"] == expected_response
    assert np.array_equal(memory_data_dict["embedding"], expected_embedding)
    assert memory_data_dict["concepts"] == expected_concepts
    assert memory_data_dict["session_id"] == session_id
    if user_id:
        assert memory_data_dict["user_id"] == user_id
    else:
        assert "user_id" not in memory_data_dict

    assert kwargs["memory_type"] == memory_type
    assert kwargs["user_id"] == user_id
    assert kwargs["session_id"] == session_id


def test_process_input_agent_memory(
    memory_manager, mock_chat_model, mock_embedding_model, mock_storage_engine, mock_app_config
):
    """Test process_input with memory_type='agent'."""
    test_query = "Agent specific query"
    user_id = None # Typically None for agent memory
    session_id = None # Typically None for agent memory
    memory_type = "agent"

    expected_embedding = mock_embedding_model.get_embedding.return_value
    expected_concepts = mock_chat_model.extract_concepts.return_value
    expected_response = mock_chat_model.generate_response.return_value

    memory_manager.process_input(
        query=test_query,
        user_id=user_id,
        session_id=session_id,
        memory_type=memory_type
    )

    mock_storage_engine.retrieve_memories.assert_called_once_with(
        query_embedding=expected_embedding,
        concepts=expected_concepts,
        memory_type=memory_type,
        user_id=user_id,
        session_id=session_id
    )

    mock_storage_engine.store_memory.assert_called_once()
    args, kwargs = mock_storage_engine.store_memory.call_args
    
    memory_data_dict = args[0]
    assert memory_data_dict["prompt"] == test_query
    assert memory_data_dict["response"] == expected_response
    assert np.array_equal(memory_data_dict["embedding"], expected_embedding)
    assert memory_data_dict["concepts"] == expected_concepts
    assert "user_id" not in memory_data_dict # Should not be present if None
    assert "session_id" not in memory_data_dict # Should not be present if None
    
    assert kwargs["memory_type"] == memory_type
    assert kwargs["user_id"] == user_id
    assert kwargs["session_id"] == session_id


# pytest.main() # For running from script, not needed if run via `pytest` command
