import pytest
import numpy as np
import faiss
from collections import defaultdict
from datetime import datetime
from typing import Dict, Any, List
import uuid # For generating unique IDs

from mindforge.core.memory_store import MemoryStore

TEST_EMBEDDING_DIM = 10 # Keep this small for tests

# --- Helper Functions ---

def create_sample_interaction(
    interaction_id: str,
    prompt: str = "Test prompt",
    response: str = "Test response",
    concepts: List[str] = None,
    embedding: np.ndarray = None,
    user_id: str = None,
    session_id: str = None,
    timestamp: float = None
) -> Dict[str, Any]:
    if concepts is None:
        concepts = ["test_concept"]
    if embedding is None:
        embedding = np.random.rand(TEST_EMBEDDING_DIM).astype(np.float32)
    
    interaction = {
        "id": interaction_id,
        "prompt": prompt,
        "response": response,
        "embedding": embedding, # Stored as numpy array in MemoryStore's internal lists
        "concepts": concepts,
        "timestamp": timestamp if timestamp else datetime.now().timestamp()
    }
    if user_id:
        interaction["user_id"] = user_id
    if session_id:
        interaction["session_id"] = session_id
    return interaction

# --- Pytest Fixtures ---

@pytest.fixture
def memory_store() -> MemoryStore:
    """Fixture to create a MemoryStore instance for testing."""
    return MemoryStore(dimension=TEST_EMBEDDING_DIM)

# --- Test Cases ---

# 1. Initialization Tests
def test_memory_store_initialization(memory_store: MemoryStore):
    """Test that MemoryStore initializes with expected empty structures."""
    assert memory_store.dimension == TEST_EMBEDDING_DIM
    assert isinstance(memory_store.short_term_memory, list)
    assert len(memory_store.short_term_memory) == 0
    
    assert isinstance(memory_store.long_term_memory, list)
    assert len(memory_store.long_term_memory) == 0
    
    assert isinstance(memory_store.embeddings, list)
    assert len(memory_store.embeddings) == 0
    
    assert isinstance(memory_store.timestamps, list)
    assert len(memory_store.timestamps) == 0
    
    assert isinstance(memory_store.access_counts, list)
    assert len(memory_store.access_counts) == 0
    
    assert isinstance(memory_store.concepts_list, list)
    assert len(memory_store.concepts_list) == 0

    assert isinstance(memory_store.user_memory, defaultdict)
    assert memory_store.user_memory.default_factory == list
    assert len(memory_store.user_memory) == 0
    
    assert isinstance(memory_store.session_memory, defaultdict)
    assert memory_store.session_memory.default_factory == list
    assert len(memory_store.session_memory) == 0
    
    assert isinstance(memory_store.agent_memory, list)
    assert len(memory_store.agent_memory) == 0
    
    assert isinstance(memory_store.index, faiss.IndexFlatL2)
    assert memory_store.index.d == TEST_EMBEDDING_DIM
    assert memory_store.index.ntotal == 0


# 2. add_interaction Tests
def test_add_interaction_short_term(memory_store: MemoryStore):
    """Test basic addition to short-term memory."""
    interaction_id = str(uuid.uuid4())
    sample_interaction = create_sample_interaction(interaction_id)
    
    memory_store.add_interaction(sample_interaction, memory_level="short_term")
    
    assert len(memory_store.short_term_memory) == 1
    assert memory_store.short_term_memory[0]["id"] == interaction_id
    assert memory_store.index.ntotal == 1
    assert len(memory_store.embeddings) == 1
    assert np.array_equal(memory_store.embeddings[0].reshape(1, -1), sample_interaction["embedding"].reshape(1, -1))
    assert len(memory_store.timestamps) == 1
    assert len(memory_store.access_counts) == 1
    assert memory_store.access_counts[0] == 1
    assert len(memory_store.concepts_list) == 1
    assert memory_store.concepts_list[0] == set(sample_interaction["concepts"])

def test_add_interaction_user_specific(memory_store: MemoryStore):
    """Test adding user-specific memory."""
    user_id = "user_test_01"
    interaction_id = str(uuid.uuid4())
    sample_interaction = create_sample_interaction(interaction_id, user_id=user_id)
    
    memory_store.add_interaction(sample_interaction, memory_level="user")
    
    assert len(memory_store.short_term_memory) == 1
    assert memory_store.short_term_memory[0]["id"] == interaction_id
    assert memory_store.index.ntotal == 1
    assert interaction_id in memory_store.user_memory[user_id]
    assert len(memory_store.user_memory[user_id]) == 1

def test_add_interaction_user_specific_missing_userid(memory_store: MemoryStore, capsys):
    """Test adding user-specific memory without user_id logs a warning."""
    interaction_id = str(uuid.uuid4())
    # user_id is deliberately missing from the interaction dict for this test
    sample_interaction = create_sample_interaction(interaction_id) 
    
    memory_store.add_interaction(sample_interaction, memory_level="user")
    
    captured = capsys.readouterr()
    assert f"Warning: user_id missing for user-level memory interaction {interaction_id}" in captured.out
    # Interaction should still be added to short_term_memory as per current implementation
    assert len(memory_store.short_term_memory) == 1 
    # But not to any specific user's list
    assert len(memory_store.user_memory) == 0 


def test_add_interaction_session_specific(memory_store: MemoryStore):
    """Test adding session-specific memory."""
    session_id = "session_test_01"
    interaction_id = str(uuid.uuid4())
    sample_interaction = create_sample_interaction(interaction_id, session_id=session_id)
    
    memory_store.add_interaction(sample_interaction, memory_level="session")
    
    assert len(memory_store.short_term_memory) == 1
    assert memory_store.short_term_memory[0]["id"] == interaction_id
    assert memory_store.index.ntotal == 1
    assert interaction_id in memory_store.session_memory[session_id]
    assert len(memory_store.session_memory[session_id]) == 1

def test_add_interaction_session_specific_missing_sessionid(memory_store: MemoryStore, capsys):
    """Test adding session-specific memory without session_id logs a warning."""
    interaction_id = str(uuid.uuid4())
    # session_id is deliberately missing
    sample_interaction = create_sample_interaction(interaction_id)
    
    memory_store.add_interaction(sample_interaction, memory_level="session")
    
    captured = capsys.readouterr()
    assert f"Warning: session_id missing for session-level memory interaction {interaction_id}" in captured.out
    assert len(memory_store.short_term_memory) == 1
    assert len(memory_store.session_memory) == 0


def test_add_interaction_agent_memory(memory_store: MemoryStore):
    """Test adding agent memory."""
    interaction_id = str(uuid.uuid4())
    sample_interaction = create_sample_interaction(interaction_id)
    
    memory_store.add_interaction(sample_interaction, memory_level="agent")
    
    assert len(memory_store.short_term_memory) == 1
    assert memory_store.short_term_memory[0]["id"] == interaction_id
    assert memory_store.index.ntotal == 1
    assert interaction_id in memory_store.agent_memory
    assert len(memory_store.agent_memory) == 1

def test_add_interaction_long_term_memory(memory_store: MemoryStore):
    """Test adding to long-term memory."""
    interaction_id = str(uuid.uuid4())
    sample_interaction = create_sample_interaction(interaction_id)
    
    memory_store.add_interaction(sample_interaction, memory_level="long_term")
    
    assert len(memory_store.short_term_memory) == 1 # Also in short-term for FAISS
    assert memory_store.short_term_memory[0]["id"] == interaction_id
    assert memory_store.index.ntotal == 1
    
    assert len(memory_store.long_term_memory) == 1
    assert memory_store.long_term_memory[0]["id"] == interaction_id
    assert memory_store.long_term_memory[0] == sample_interaction # Full object


# 3. retrieve Tests
@pytest.fixture
def populated_memory_store(memory_store: MemoryStore) -> MemoryStore:
    """MemoryStore populated with diverse interactions for retrieval tests."""
    # For consistent embeddings for easier similarity checks if needed
    emb1 = np.array([0.1] * TEST_EMBEDDING_DIM, dtype=np.float32)
    emb2 = np.array([0.2] * TEST_EMBEDDING_DIM, dtype=np.float32) # Close to emb1
    emb3 = np.array([0.8] * TEST_EMBEDDING_DIM, dtype=np.float32) # Different from emb1, emb2
    emb4 = np.array([0.15] * TEST_EMBEDDING_DIM, dtype=np.float32) # Very close to emb1

    # Interactions
    memory_store.add_interaction(create_sample_interaction("id_short_1", embedding=emb1.copy()), memory_level="short_term")
    memory_store.add_interaction(create_sample_interaction("id_user_u1_1", user_id="u1", embedding=emb2.copy()), memory_level="user")
    memory_store.add_interaction(create_sample_interaction("id_user_u1_2", user_id="u1", embedding=emb4.copy()), memory_level="user")
    memory_store.add_interaction(create_sample_interaction("id_user_u2_1", user_id="u2", embedding=emb3.copy()), memory_level="user")
    memory_store.add_interaction(create_sample_interaction("id_session_s1_1", session_id="s1", embedding=emb1.copy()), memory_level="session")
    memory_store.add_interaction(create_sample_interaction("id_agent_1", embedding=emb2.copy()), memory_level="agent")
    
    # Add one to long_term as well (this also adds to short_term_memory and FAISS)
    lt_interaction = create_sample_interaction("id_longterm_1", embedding=emb3.copy())
    memory_store.add_interaction(lt_interaction, memory_level="long_term") # This adds to long_term_memory AND short_term_memory

    return memory_store

def test_retrieve_no_level_filter(populated_memory_store: MemoryStore):
    """Retrieve with no memory_level filter, should get all similar from FAISS."""
    query_embedding = np.array([0.11] * TEST_EMBEDDING_DIM, dtype=np.float32) # Close to emb1, emb4, emb2
    results = populated_memory_store.retrieve(query_embedding, query_concepts=["test_concept"])
    
    # Expecting results based on vector similarity from all indexed items
    # Exact number depends on similarity_threshold (default 0.7) and actual distances
    # emb1 (id_short_1, id_session_s1_1), emb4 (id_user_u1_2), emb2 (id_user_u1_1, id_agent_1) are candidates
    # emb3 (id_user_u2_1, id_longterm_1) is less likely
    retrieved_ids = {r["id"] for r in results}
    assert "id_short_1" in retrieved_ids
    assert "id_session_s1_1" in retrieved_ids
    assert "id_user_u1_2" in retrieved_ids # from emb4
    # Depending on threshold and exact calculations, others might appear
    # This test mainly ensures it *can* retrieve across types if no filter.

def test_retrieve_short_term(populated_memory_store: MemoryStore):
    """Retrieve with memory_level='short_term'."""
    query_embedding = np.array([0.11] * TEST_EMBEDDING_DIM, dtype=np.float32)
    results = populated_memory_store.retrieve(query_embedding, query_concepts=["test_concept"], memory_level="short_term")
    
    # Should behave like no_level_filter as short_term is the default pool for FAISS
    retrieved_ids = {r["id"] for r in results}
    assert "id_short_1" in retrieved_ids 
    # All items are in short_term_memory for FAISS indexing, so this filter doesn't exclude by ID list.

def test_retrieve_long_term(populated_memory_store: MemoryStore):
    """Retrieve with memory_level='long_term'."""
    query_embedding = np.array([0.79] * TEST_EMBEDDING_DIM, dtype=np.float32) # Close to emb3
    results = populated_memory_store.retrieve(query_embedding, query_concepts=["test_concept"], memory_level="long_term")
    
    retrieved_ids = {r["id"] for r in results}
    assert "id_longterm_1" in retrieved_ids
    assert len(retrieved_ids) >= 1 # Might also pick up user_u2_1 if similarity is high enough and it's also in LT
    # Check that other items not in long_term_memory are excluded even if similar
    assert "id_short_1" not in retrieved_ids 

def test_retrieve_user_specific(populated_memory_store: MemoryStore):
    """Retrieve user-specific memories."""
    query_embedding = np.array([0.18] * TEST_EMBEDDING_DIM, dtype=np.float32) # Close to emb2 and emb4
    
    # User u1
    results_u1 = populated_memory_store.retrieve(query_embedding, query_concepts=["test_concept"], memory_level="user", user_id="u1")
    retrieved_ids_u1 = {r["id"] for r in results_u1}
    assert "id_user_u1_1" in retrieved_ids_u1 # emb2
    assert "id_user_u1_2" in retrieved_ids_u1 # emb4
    assert "id_user_u2_1" not in retrieved_ids_u1
    assert "id_short_1" not in retrieved_ids_u1

    # Non-existent user
    results_nonexistent = populated_memory_store.retrieve(query_embedding, query_concepts=["test_concept"], memory_level="user", user_id="nonexistent_user")
    assert len(results_nonexistent) == 0

def test_retrieve_session_specific(populated_memory_store: MemoryStore):
    """Retrieve session-specific memories."""
    query_embedding = np.array([0.11] * TEST_EMBEDDING_DIM, dtype=np.float32) # Close to emb1
    results = populated_memory_store.retrieve(query_embedding, query_concepts=["test_concept"], memory_level="session", session_id="s1")
    retrieved_ids = {r["id"] for r in results}
    assert "id_session_s1_1" in retrieved_ids
    assert "id_short_1" not in retrieved_ids # Even though emb1 is similar, it's not in session s1 list

def test_retrieve_agent_memory(populated_memory_store: MemoryStore):
    """Retrieve agent memories."""
    query_embedding = np.array([0.22] * TEST_EMBEDDING_DIM, dtype=np.float32) # Close to emb2
    results = populated_memory_store.retrieve(query_embedding, query_concepts=["test_concept"], memory_level="agent")
    retrieved_ids = {r["id"] for r in results}
    assert "id_agent_1" in retrieved_ids
    assert "id_user_u1_1" not in retrieved_ids # Even though emb2 is similar, not in agent list

def test_retrieve_similarity_threshold(populated_memory_store: MemoryStore):
    """Test that similarity_threshold is respected."""
    # emb1 is [0.1]*DIM. emb_query1 is very similar. emb_query2 is less similar.
    emb_query1 = np.array([0.101] * TEST_EMBEDDING_DIM, dtype=np.float32)
    emb_query2 = np.array([0.5] * TEST_EMBEDDING_DIM, dtype=np.float32) # Distance likely > threshold

    # High threshold, should only get very similar
    results_high_thresh = populated_memory_store.retrieve(emb_query1, query_concepts=["test_concept"], similarity_threshold=0.95)
    retrieved_ids_high = {r["id"] for r in results_high_thresh}
    assert "id_short_1" in retrieved_ids_high # Assuming this is very similar

    # Using emb_query2 which should be further away from emb1
    results_query2_default_thresh = populated_memory_store.retrieve(emb_query2, query_concepts=["test_concept"]) 
    # We'd expect id_short_1 (emb1) NOT to be here if 0.5 is too far for default 0.7 threshold
    # This depends heavily on L2 distance vs threshold conversion.
    # A more robust way is to check if a known dissimilar item is excluded.
    # emb3 ([0.8]) should be dissimilar to query_embedding_close_to_emb1 ([0.11])
    query_embedding_close_to_emb1 = np.array([0.11] * TEST_EMBEDDING_DIM, dtype=np.float32)
    results_default_thresh = populated_memory_store.retrieve(query_embedding_close_to_emb1, query_concepts=["test_concept"], similarity_threshold=0.7)
    retrieved_ids_default = {r["id"] for r in results_default_thresh}
    assert "id_user_u2_1" not in retrieved_ids_default # Based on emb3 being far from query

    results_low_thresh = populated_memory_store.retrieve(query_embedding_close_to_emb1, query_concepts=["test_concept"], similarity_threshold=0.1)
    retrieved_ids_low = {r["id"] for r in results_low_thresh}
    assert "id_user_u2_1" in retrieved_ids_low # emb3 should now be included with low threshold

# 4. Promotion Methods Tests
def test_promote_to_long_term(memory_store: MemoryStore):
    interaction_id = str(uuid.uuid4())
    sample_interaction = create_sample_interaction(interaction_id)
    memory_store.add_interaction(sample_interaction, memory_level="short_term")

    assert memory_store.promote_to_long_term(interaction_id) is True
    assert len(memory_store.long_term_memory) == 1
    assert memory_store.long_term_memory[0]["id"] == interaction_id
    assert memory_store.long_term_memory[0] == sample_interaction

    # Idempotency
    assert memory_store.promote_to_long_term(interaction_id) is False
    assert len(memory_store.long_term_memory) == 1 # No duplicates

    # Non-existent ID
    assert memory_store.promote_to_long_term("non_existent_id") is False

def test_promote_session_to_user(memory_store: MemoryStore):
    user_id = "user_prom_1"
    session_id = "session_prom_1"
    interaction_id = str(uuid.uuid4())
    
    sample_interaction = create_sample_interaction(interaction_id, session_id=session_id)
    memory_store.add_interaction(sample_interaction, memory_level="session")

    assert memory_store.promote_session_to_user(interaction_id, user_id) is True
    assert interaction_id in memory_store.user_memory[user_id]

    # Idempotency
    assert memory_store.promote_session_to_user(interaction_id, user_id) is False
    assert len(memory_store.user_memory[user_id]) == 1

    # ID not in any session
    other_interaction_id = str(uuid.uuid4())
    memory_store.add_interaction(create_sample_interaction(other_interaction_id), memory_level="short_term")
    assert memory_store.promote_session_to_user(other_interaction_id, user_id) is False

    # Invalid user_id
    assert memory_store.promote_session_to_user(interaction_id, "") is False # Empty user_id

def test_mark_as_agent_knowledge(memory_store: MemoryStore):
    interaction_id = str(uuid.uuid4())
    sample_interaction = create_sample_interaction(interaction_id)
    memory_store.add_interaction(sample_interaction, memory_level="short_term")

    assert memory_store.mark_as_agent_knowledge(interaction_id) is True
    assert interaction_id in memory_store.agent_memory

    # Idempotency
    assert memory_store.mark_as_agent_knowledge(interaction_id) is False
    assert len(memory_store.agent_memory) == 1
    
    # Optional: Test with non-existent ID (current implementation doesn't check if ID is in short_term_memory)
    # Depending on desired strictness, this could be a valid case or an error.
    # For now, the method allows adding any string ID.
    # assert memory_store.mark_as_agent_knowledge("non_existent_id_for_agent") is True 


# if __name__ == "__main__":
# pytest.main()
