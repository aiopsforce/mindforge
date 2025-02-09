import os
import numpy as np
from mindforge.config import AppConfig
from mindforge.models.chat import OllamaChatModel
from mindforge.models.embedding import OllamaEmbeddingModel
from mindforge import MemoryManager
from mindforge.storage.sqlite_engine import SQLiteEngine

# --- Setup: Ollama Models and Configuration ---

# Use a consistent configuration for all examples.
config = AppConfig()
config.model.use_model = "ollama"
config.model.chat_model_name = "llama2"  # Or your preferred Ollama model
config.model.embedding_model_name = "llama2"
config.model.ollama_base_url = "http://localhost:11434"  # Default Ollama URL
config.storage.db_path = "ollama_memory_test.db"  # Use a separate DB for testing

# Initialize Ollama models (make sure Ollama is running)
chat_model = OllamaChatModel(
    model_name=config.model.chat_model_name, base_url=config.model.ollama_base_url
)
embedding_model = OllamaEmbeddingModel(
    model_name=config.model.embedding_model_name,
    base_url=config.model.ollama_base_url,
)

# Initialize MemoryManager
manager = MemoryManager(
    chat_model=chat_model, embedding_model=embedding_model, config=config
)

# --- Scenario 1: Short-Term vs. Long-Term Memory ---
#   - Demonstrate storing and retrieving from short-term and long-term memory.

print("\n--- Scenario 1: Short-Term vs. Long-Term Memory ---")

# Initial interaction (short-term)
response1 = manager.process_input(query="What's the weather like today?")
print(f"Short-Term Response 1: {response1}")

# Another short-term interaction
response2 = manager.process_input(query="And what about tomorrow?")
print(f"Short-Term Response 2: {response2}")


# Store a fact in long-term memory
# We simulate getting an embedding.  In a real application, process_input already does this.
long_term_embedding = embedding_model.get_embedding("The capital of France is Paris.")
manager._store_interaction(
    query="What's the capital of France?",
    response="Paris",
    embedding=long_term_embedding,
    concepts=["capital", "France", "Paris"],
    memory_type="long_term",
)

# Retrieve from long-term memory
# Simulate a query that should retrieve the long-term memory.
query_embedding = embedding_model.get_embedding("What city is the capital of France?")
long_term_memories = manager.storage.retrieve_memories(
    query_embedding=query_embedding, concepts=["capital", "France"], memory_type="long_term"
)
print(f"Long-Term Memories Retrieved: {long_term_memories}")


# --- Scenario 2: User-Specific Memory ---
#   - Show how to store and retrieve memories associated with a particular user.

print("\n--- Scenario 2: User-Specific Memory ---")

# Interaction with user1
response_user1_1 = manager.process_input(
    query="I like cats.", user_id="user1", memory_type="user"
)
print(f"User1 Response 1: {response_user1_1}")

# Another interaction with user1
response_user1_2 = manager.process_input(
    query="What's my favorite animal?", user_id="user1", memory_type="user"
)
print(f"User1 Response 2: {response_user1_2}")

# Interaction with user2
response_user2_1 = manager.process_input(
    query="I prefer dogs.", user_id="user2", memory_type="user"
)
print(f"User2 Response 1: {response_user2_1}")

# Retrieve user1's preferences.  Simulate a query and retrieval.
query_embedding_user1 = embedding_model.get_embedding("What does user1 like?")
user1_memories = manager.storage.retrieve_memories(
    query_embedding=query_embedding_user1,
    concepts=["likes", "preference"],
    user_id="user1",
    memory_type="user",
)
print(f"User1 Memories Retrieved: {user1_memories}")

# --- Scenario 3: Session-Specific Memory ---
#   - Demonstrate using session-specific context.

print("\n--- Scenario 3: Session-Specific Memory ---")

# Start a session (session1)
response_session1_1 = manager.process_input(
    query="Let's talk about movies.", session_id="session1", memory_type="session"
)
print(f"Session1 Response 1: {response_session1_1}")

# Continue the session
response_session1_2 = manager.process_input(
    query="What are some good action movies?",
    session_id="session1",
    memory_type="session",
)
print(f"Session1 Response 2: {response_session1_2}")


# Start a new session (session2)
response_session2_1 = manager.process_input(
    query="Now let's discuss books.", session_id="session2", memory_type="session"
)
print(f"Session2 Response 1: {response_session2_1}")

# Retrieve session1's context.  Simulate a query and retrieval.
query_embedding_session1 = embedding_model.get_embedding(
    "What were we discussing in session1?"
)
session1_memories = manager.storage.retrieve_memories(
    query_embedding=query_embedding_session1,
    concepts=["discussion", "session"],
    session_id="session1",
    memory_type="session",
)
print(f"Session1 Memories Retrieved: {session1_memories}")

# --- Scenario 4: Agent-Specific Memory ---
#  - Show how agent memory (knowledge, adaptability) is handled.
print("\n--- Scenario 4: Agent-Specific Memory ---")

# Store agent knowledge
agent_knowledge_embedding = embedding_model.get_embedding("The speed of light is approximately 300,000 km/s.")
manager._store_interaction(query="What's the speed of light", response="approximately 300,000 km/s", embedding=agent_knowledge_embedding, concepts=["speed", "light"], memory_type="agent")


# Simulate a query that uses agent memory
response_agent = manager.process_input(
    query="How fast does light travel?",
)  # No memory_type specified, so it checks all.
print(f"Agent Response (using knowledge): {response_agent}")

# --- Scenario 5: Concept Graph & Spreading Activation ---

print("\n--- Scenario 5: Concept Graph & Spreading Activation ---")

# Add some interactions that will create concept relationships
manager.process_input(query="Machine learning involves algorithms and data.")
manager.process_input(query="Deep learning is a subset of machine learning.")
manager.process_input(query="Algorithms use data for training.")

# Get related concepts for "machine learning"
related_concepts = manager.concept_graph.get_related_concepts("machine_learning")
print(f"Concepts related to 'machine_learning': {related_concepts}")

# --- Scenario 6:  Semantic Clustering ---
# (This is usually done automatically, but we can trigger it manually)

print("\n--- Scenario 6: Semantic Clustering ---")
manager.clustering.cluster_memories()
# The clustering results are now stored as concepts in the database.  You can see this through standard retrievals.

# --- Scenario 7: Vector-Based Similarity Search ---
print("\n--- Scenario 7: Vector-Based Similarity Search ---")

# Example interactions and embeddings
interactions = [
    ("What is the capital of France?", "Paris"),
    ("What is the largest country in the world?", "Russia"),
    ("What is the chemical symbol for gold?", "Au"),
]
for question, answer in interactions:
    embedding = embedding_model.get_embedding(question)
    manager._store_interaction(
        query=question,
        response=answer,
        embedding=embedding,
        concepts=question.split(),
        memory_type="short_term",
    )

# Similar query
similar_query = "What's the capital city of France?"
similar_query_embedding = embedding_model.get_embedding(similar_query)

#Retrieve memories
retrieved_memories = manager.storage.retrieve_memories(
query_embedding = similar_query_embedding,
concepts = similar_query.split(),
)
print(f"Retrieved memories for '{similar_query}': {retrieved_memories}")

# --- Cleanup (Optional) ---
# Delete the test database file.  Important for clean testing.
if os.path.exists(config.storage.db_path):
    os.remove(config.storage.db_path)
print("Test database cleaned up.")