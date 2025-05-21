import pytest
import numpy as np
import time
from mindforge.storage.sqlite_engine import SQLiteEngine
from mindforge.config import AppConfig # Needed for default embedding_dim if not overridden

# Define a consistent embedding dimension for tests
TEST_EMBEDDING_DIM = 10

@pytest.fixture
def memory_engine():
    """Fixture to create an in-memory SQLiteEngine for testing."""
    # Using AppConfig to potentially get default embedding_dim, though we override for clarity
    # config = AppConfig()
    engine = SQLiteEngine(db_path=":memory:", embedding_dim=TEST_EMBEDDING_DIM)
    # engine._initialize_db() # Initialization is called in __init__
    yield engine
    # No explicit close needed for :memory: as it's wiped, but good practice if it were a file
    if hasattr(engine, 'conn') and engine.conn:
        engine.conn.close()

@pytest.fixture
def sample_memory_data():
    """Fixture to provide sample memory data."""
    return {
        "id": "test_memory_1",
        "prompt": "Test prompt",
        "response": "Test response",
        "embedding": np.array([0.1] * TEST_EMBEDDING_DIM).astype(float), # Consistent with TEST_EMBEDDING_DIM
        "concepts": ["test", "memory"],
        # For type-specific data, add them in the test itself
    }

def test_store_and_retrieve_short_term_memory(memory_engine, sample_memory_data):
    """Test storing and retrieving a short-term memory."""
    engine = memory_engine
    data = sample_memory_data

    engine.store_memory(memory_data=data, memory_type="short_term")

    retrieved_memories = engine.retrieve_memories(
        query_embedding=data["embedding"],
        concepts=data["concepts"],
        memory_type="short_term",
        limit=1
    )

    assert len(retrieved_memories) == 1
    retrieved = retrieved_memories[0]
    assert retrieved["id"] == data["id"]
    assert retrieved["prompt"] == data["prompt"]
    assert retrieved["response"] == data["response"]
    assert retrieved["memory_type"] == "short_term"
    # Concepts are stored in a separate table and joined, check if they are present
    # The retrieved concepts might be a string or list depending on GROUP_CONCAT, adjust as needed
    # Current retrieve_memories returns concepts as a list
    assert all(c in retrieved["concepts"] for c in data["concepts"])
    assert len(retrieved["concepts"]) == len(data["concepts"])


def test_store_and_retrieve_long_term_memory(memory_engine, sample_memory_data):
    engine = memory_engine
    data = {**sample_memory_data, "id": "long_term_1"}

    engine.store_memory(memory_data=data, memory_type="long_term")
    retrieved = engine.retrieve_memories(data["embedding"], data["concepts"], "long_term", limit=1)[0]

    assert retrieved["id"] == data["id"]
    assert retrieved["memory_type"] == "long_term"

def test_store_and_retrieve_user_memory(memory_engine, sample_memory_data):
    engine = memory_engine
    user_id = "user123"
    data = {**sample_memory_data, "id": "user_mem_1", "preference": 0.8, "history": 0.5}

    engine.store_memory(memory_data=data, memory_type="user", user_id=user_id)
    
    # Direct check on user_memories table
    with engine._get_db_connection() as conn:
        cursor = conn.execute("SELECT * FROM user_memories WHERE user_id = ? AND memory_id = ?", (user_id, data["id"]))
        user_mem_row = cursor.fetchone()
        assert user_mem_row is not None
        assert user_mem_row["preference"] == data["preference"]

    retrieved_memories = engine.retrieve_memories(
        query_embedding=data["embedding"],
        concepts=data["concepts"],
        memory_type="user",
        user_id=user_id,
        limit=1
    )
    assert len(retrieved_memories) == 1
    retrieved = retrieved_memories[0]
    assert retrieved["id"] == data["id"]
    assert retrieved["memory_type"] == "user"
    assert retrieved["user_preference"] == data["preference"] # Check joined data

def test_store_and_retrieve_session_memory(memory_engine, sample_memory_data):
    engine = memory_engine
    session_id = "session456"
    data = {**sample_memory_data, "id": "session_mem_1", "recent_activity": 0.9, "context": 0.7}

    engine.store_memory(memory_data=data, memory_type="session", session_id=session_id)

    # Direct check on session_memories table
    with engine._get_db_connection() as conn:
        cursor = conn.execute("SELECT * FROM session_memories WHERE session_id = ? AND memory_id = ?", (session_id, data["id"]))
        session_mem_row = cursor.fetchone()
        assert session_mem_row is not None
        assert session_mem_row["recent_activity"] == data["recent_activity"]

    retrieved_memories = engine.retrieve_memories(
        query_embedding=data["embedding"],
        concepts=data["concepts"],
        memory_type="session",
        session_id=session_id,
        limit=1
    )
    assert len(retrieved_memories) == 1
    retrieved = retrieved_memories[0]
    assert retrieved["id"] == data["id"]
    assert retrieved["memory_type"] == "session"
    assert retrieved["session_activity"] == data["recent_activity"]

def test_store_and_retrieve_agent_memory(memory_engine, sample_memory_data):
    engine = memory_engine
    data = {**sample_memory_data, "id": "agent_mem_1", "knowledge": 0.95, "adaptability": 0.6}

    engine.store_memory(memory_data=data, memory_type="agent")

    # Direct check on agent_memories table
    with engine._get_db_connection() as conn:
        cursor = conn.execute("SELECT * FROM agent_memories WHERE memory_id = ?", (data["id"],))
        agent_mem_row = cursor.fetchone()
        assert agent_mem_row is not None
        assert agent_mem_row["knowledge"] == data["knowledge"]


    retrieved_memories = engine.retrieve_memories(
        query_embedding=data["embedding"],
        concepts=data["concepts"],
        memory_type="agent",
        limit=1
    )
    assert len(retrieved_memories) == 1
    retrieved = retrieved_memories[0]
    assert retrieved["id"] == data["id"]
    assert retrieved["memory_type"] == "agent"
    assert retrieved["agent_knowledge"] == data["knowledge"]


def test_recency_boost_short_term_memory(memory_engine, sample_memory_data):
    """Test that recency_boost increases for short-term memory on access."""
    engine = memory_engine
    data = {**sample_memory_data, "id": "recency_test_1"}

    engine.store_memory(memory_data=data, memory_type="short_term")

    # Retrieve initial memory to get its recency_boost
    retrieved_initial = engine.retrieve_memories(data["embedding"], data["concepts"], "short_term", limit=1)[0]
    initial_boost = retrieved_initial["recency_boost"]
    assert initial_boost == 1.0 # Default value

    # Access the memory multiple times
    # retrieve_memories calls _update_access_patterns internally
    for _ in range(3):
        time.sleep(0.01) # Ensure last_access time changes if DB resolution is low
        engine.retrieve_memories(data["embedding"], data["concepts"], "short_term", limit=1)

    # Retrieve again and check recency_boost
    # Need to fetch directly from DB as retrieve_memories might return a cached/calculated value
    # or its own copy that doesn't reflect the update immediately for this test's purpose.
    with engine._get_db_connection() as conn:
        cursor = conn.execute("SELECT recency_boost FROM memories WHERE id = ?", (data["id"],))
        updated_boost_row = cursor.fetchone()
    
    assert updated_boost_row is not None
    updated_boost = updated_boost_row["recency_boost"]
    
    # Expected boost: 1.0 * 1.1 * 1.1 * 1.1 (for the 3 retrieves after initial store)
    # Plus one more 1.1 from the initial retrieve_initial
    # So, 1.0 * (1.1 ** 4) if we count the retrieve_initial call as an access that boosts.
    # The logic is: initial store sets it to 1.0. Each access via _update_access_patterns for short_term multiplies by 1.1.
    # The fixture retrieve_initial is one access. The 3 loops are 3 more accesses.
    expected_boost = 1.0 * (1.1**4)
    assert updated_boost == pytest.approx(expected_boost)

def test_recency_boost_long_term_memory(memory_engine, sample_memory_data):
    """Test that recency_boost does NOT change for long-term memory on access."""
    engine = memory_engine
    data = {**sample_memory_data, "id": "recency_test_long_term_1"}

    engine.store_memory(memory_data=data, memory_type="long_term")

    retrieved_initial = engine.retrieve_memories(data["embedding"], data["concepts"], "long_term", limit=1)[0]
    initial_boost = retrieved_initial["recency_boost"]
    assert initial_boost == 1.0

    for _ in range(3):
        time.sleep(0.01)
        engine.retrieve_memories(data["embedding"], data["concepts"], "long_term", limit=1)

    with engine._get_db_connection() as conn:
        cursor = conn.execute("SELECT recency_boost FROM memories WHERE id = ?", (data["id"],))
        updated_boost_row = cursor.fetchone()
    
    assert updated_boost_row is not None
    updated_boost = updated_boost_row["recency_boost"]
    assert updated_boost == pytest.approx(initial_boost) # Should remain 1.0
    
import sqlite3 # Added for helper functions
import uuid # For generating unique IDs

# Helper to get DB connection, if not exposed by SQLiteEngine
# Add this to SQLiteEngine or make _get_db_connection public if needed for tests
# For now, assuming _get_db_connection can be used for testing purposes.
# SQLiteEngine in the provided code does not have a public _get_db_connection.
# For testing, we'll connect directly or add a helper if it were a real scenario.
# However, memory_engine fixture provides the engine instance, and we can use its db_path.

def get_db_connection(db_path: str):
    """Helper to get a new SQLite connection for verification."""
    return sqlite3.connect(db_path)

def get_memory_details(conn: sqlite3.Connection, mem_id: str) -> sqlite3.Row:
    """Fetches a memory's details from the memories table."""
    return conn.execute("SELECT * FROM memories WHERE id = ?", (mem_id,)).fetchone()

def get_user_memory_link(conn: sqlite3.Connection, mem_id: str, user_id: str) -> sqlite3.Row:
    """Checks if a memory_id is linked to a user_id in user_memories."""
    return conn.execute("SELECT * FROM user_memories WHERE memory_id = ? AND user_id = ?", (mem_id, user_id)).fetchone()

def get_session_memory_link(conn: sqlite3.Connection, mem_id: str, session_id: str) -> sqlite3.Row:
    """Checks if a memory_id is linked to a session_id in session_memories."""
    return conn.execute("SELECT * FROM session_memories WHERE memory_id = ? AND session_id = ?", (mem_id, session_id)).fetchone()

def get_agent_memory_link(conn: sqlite3.Connection, mem_id: str) -> sqlite3.Row:
    """Checks if a memory_id is in agent_memories."""
    return conn.execute("SELECT * FROM agent_memories WHERE memory_id = ?", (mem_id,)).fetchone()


# Add a test for concept graph updates (basic)
def test_concept_graph_updated_on_store(memory_engine, sample_memory_data):
    engine = memory_engine
    data = {**sample_memory_data, "id": "concept_graph_test_1", "concepts": ["apple", "banana", "cherry"]}
    engine.store_memory(memory_data=data, memory_type="short_term")

    with engine._get_db_connection() as conn:
        # Check for ("apple", "banana") or ("banana", "apple")
        cursor = conn.execute(
            "SELECT COUNT(*) FROM concept_graph WHERE (source = 'apple' AND target = 'banana') OR (source = 'banana' AND target = 'apple')"
        )
        assert cursor.fetchone()[0] >= 1 # Should be 1, but >=1 if multiple stores happen with same pair
        
        cursor = conn.execute(
            "SELECT COUNT(*) FROM concept_graph WHERE (source = 'apple' AND target = 'cherry') OR (source = 'cherry' AND target = 'apple')"
        )
        assert cursor.fetchone()[0] >= 1
        
        cursor = conn.execute(
            "SELECT COUNT(*) FROM concept_graph WHERE (source = 'banana' AND target = 'cherry') OR (source = 'cherry' AND target = 'banana')"
        )
        assert cursor.fetchone()[0] >= 1

        # Check weights (optional, more detailed)
        cursor = conn.execute(
            "SELECT weight FROM concept_graph WHERE (source = 'apple' AND target = 'banana') OR (source = 'banana' AND target = 'apple')"
        )
        weight = cursor.fetchone()[0]
        assert weight == pytest.approx(1.0) # Initial weight

        # Store another memory with overlapping concepts to test weight update
        data2 = {**sample_memory_data, "id": "concept_graph_test_2", "concepts": ["apple", "banana", "date"]}
        engine.store_memory(memory_data=data2, memory_type="short_term")
        
        cursor = conn.execute(
            "SELECT weight FROM concept_graph WHERE (source = 'apple' AND target = 'banana') OR (source = 'banana' AND target = 'apple')"
        )
        updated_weight = cursor.fetchone()[0]
        assert updated_weight == pytest.approx(1.0 + 0.1) # Weight should increase by 0.1

# --- Tests for update_memory_level ---

def test_update_memory_level_short_to_user(memory_engine: SQLiteEngine, sample_memory_data):
    mem_id = str(uuid.uuid4())
    user_id = "user_update_1"
    data = {**sample_memory_data, "id": mem_id}
    memory_engine.store_memory(data, memory_type="short_term")

    assert memory_engine.update_memory_level(mem_id, "user", user_id=user_id) is True

    with get_db_connection(memory_engine.db_path) as conn:
        mem_details = get_memory_details(conn, mem_id)
        assert mem_details["memory_type"] == "user"
        user_link = get_user_memory_link(conn, mem_id, user_id)
        assert user_link is not None
        assert user_link["preference"] == 0.0 # Check default metadata
        assert get_session_memory_link(conn, mem_id, "any_session") is None
        assert get_agent_memory_link(conn, mem_id) is None

def test_update_memory_level_session_to_user(memory_engine: SQLiteEngine, sample_memory_data):
    mem_id = str(uuid.uuid4())
    user_id = "user_update_2"
    session_id = "session_update_1"
    data = {**sample_memory_data, "id": mem_id}
    memory_engine.store_memory(data, memory_type="session", session_id=session_id)

    assert memory_engine.update_memory_level(mem_id, "user", user_id=user_id) is True

    with get_db_connection(memory_engine.db_path) as conn:
        mem_details = get_memory_details(conn, mem_id)
        assert mem_details["memory_type"] == "user"
        user_link = get_user_memory_link(conn, mem_id, user_id)
        assert user_link is not None
        assert user_link["preference"] == 0.0
        assert get_session_memory_link(conn, mem_id, session_id) is None # Should be removed

def test_update_memory_level_short_to_agent(memory_engine: SQLiteEngine, sample_memory_data):
    mem_id = str(uuid.uuid4())
    data = {**sample_memory_data, "id": mem_id}
    memory_engine.store_memory(data, memory_type="short_term")

    assert memory_engine.update_memory_level(mem_id, "agent") is True
    
    with get_db_connection(memory_engine.db_path) as conn:
        mem_details = get_memory_details(conn, mem_id)
        assert mem_details["memory_type"] == "agent"
        agent_link = get_agent_memory_link(conn, mem_id)
        assert agent_link is not None
        assert agent_link["knowledge"] == 0.0 # Check default metadata

def test_update_memory_level_short_to_long_term(memory_engine: SQLiteEngine, sample_memory_data):
    mem_id = str(uuid.uuid4())
    data = {**sample_memory_data, "id": mem_id}
    memory_engine.store_memory(data, memory_type="short_term")

    assert memory_engine.update_memory_level(mem_id, "long_term") is True

    with get_db_connection(memory_engine.db_path) as conn:
        mem_details = get_memory_details(conn, mem_id)
        assert mem_details["memory_type"] == "long_term"
        # No specific association table for long_term

def test_update_memory_level_user_to_short_term(memory_engine: SQLiteEngine, sample_memory_data):
    mem_id = str(uuid.uuid4())
    user_id = "user_update_3"
    data = {**sample_memory_data, "id": mem_id}
    memory_engine.store_memory(data, memory_type="user", user_id=user_id)

    assert memory_engine.update_memory_level(mem_id, "short_term") is True

    with get_db_connection(memory_engine.db_path) as conn:
        mem_details = get_memory_details(conn, mem_id)
        assert mem_details["memory_type"] == "short_term"
        assert get_user_memory_link(conn, mem_id, user_id) is None # Should be removed

def test_update_memory_level_missing_user_id(memory_engine: SQLiteEngine, sample_memory_data):
    mem_id = str(uuid.uuid4())
    data = {**sample_memory_data, "id": mem_id}
    memory_engine.store_memory(data, memory_type="short_term")
    with pytest.raises(ValueError, match="user_id is required for new_memory_level 'user'"):
        memory_engine.update_memory_level(mem_id, "user") # No user_id

def test_update_memory_level_missing_session_id(memory_engine: SQLiteEngine, sample_memory_data):
    mem_id = str(uuid.uuid4())
    data = {**sample_memory_data, "id": mem_id}
    memory_engine.store_memory(data, memory_type="short_term")
    with pytest.raises(ValueError, match="session_id is required for new_memory_level 'session'"):
        memory_engine.update_memory_level(mem_id, "session") # No session_id

def test_update_memory_level_non_existent_id(memory_engine: SQLiteEngine):
    assert memory_engine.update_memory_level("non_existent_mem", "user", user_id="u1") is False

def test_update_memory_level_no_change(memory_engine: SQLiteEngine, sample_memory_data):
    mem_id = str(uuid.uuid4())
    user_id = "user_update_4"
    data = {**sample_memory_data, "id": mem_id}
    memory_engine.store_memory(data, memory_type="user", user_id=user_id)
    
    # Attempting to "update" to the same level
    assert memory_engine.update_memory_level(mem_id, "user", user_id=user_id) is False

    # Verify no unintended changes occurred
    with get_db_connection(memory_engine.db_path) as conn:
        mem_details = get_memory_details(conn, mem_id)
        assert mem_details["memory_type"] == "user"
        assert get_user_memory_link(conn, mem_id, user_id) is not None

def test_update_memory_level_default_metadata_values(memory_engine: SQLiteEngine, sample_memory_data):
    """Test that new associations get default metadata values."""
    mem_id_user = str(uuid.uuid4())
    user_id = "user_meta_1"
    data_user = {**sample_memory_data, "id": mem_id_user}
    memory_engine.store_memory(data_user, memory_type="short_term")
    memory_engine.update_memory_level(mem_id_user, "user", user_id=user_id)
    
    mem_id_session = str(uuid.uuid4())
    session_id = "session_meta_1"
    data_session = {**sample_memory_data, "id": mem_id_session}
    memory_engine.store_memory(data_session, memory_type="short_term")
    memory_engine.update_memory_level(mem_id_session, "session", session_id=session_id)

    mem_id_agent = str(uuid.uuid4())
    data_agent = {**sample_memory_data, "id": mem_id_agent}
    memory_engine.store_memory(data_agent, memory_type="short_term")
    memory_engine.update_memory_level(mem_id_agent, "agent")

    with get_db_connection(memory_engine.db_path) as conn:
        user_link = get_user_memory_link(conn, mem_id_user, user_id)
        assert user_link["preference"] == 0.0
        assert user_link["history"] == 0.0

        session_link = get_session_memory_link(conn, mem_id_session, session_id)
        assert session_link["recent_activity"] == 0.0
        assert session_link["context"] == 0.0

        agent_link = get_agent_memory_link(conn, mem_id_agent)
        assert agent_link["knowledge"] == 0.0
        assert agent_link["adaptability"] == 0.0


# pytest.main() # For running from script, not needed if run via `pytest` command
