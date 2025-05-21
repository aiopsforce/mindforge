import os
import numpy as np
import sys # For sys.exit

from mindforge.config import AppConfig
from mindforge.main import initialize_models # Using directly from main
from mindforge import MemoryManager
from mindforge.storage.sqlite_engine import SQLiteEngine
from mindforge.utils.errors import ConfigurationError, ModelError # Import ConfigurationError
from mindforge.utils.logging import LogManager # Optional: for logging

# --- Configuration and Initialization ---

# Use a consistent configuration for all examples.
config = AppConfig()
# --- IMPORTANT ---
# You can override the model provider here for testing if needed, e.g.:
# config.model.use_model = "ollama" # or "openai", "azure"
# config.model.chat_model_name = "llama2" # Adjust as per provider
# config.model.embedding_model_name = "llama2" # Adjust as per provider
# By default, it will use what's in AppConfig defaults (likely OpenAI)
# Ensure OLLAMA_BASE_URL is set if using Ollama, etc.

config.storage.db_path = "full_example_memory.db"  # Renamed database

# Initialize logging (optional, but good practice for examples)
log_manager = LogManager(log_dir=config.log_dir, log_level=config.log_level)
logger = log_manager.get_logger("mindforge.full_example")

try:
    logger.info(f"Starting MindForge Full Example with provider: {config.model.use_model}")

    # Handle API key for OpenAI directly from environment variable, similar to main.py
    if config.model.use_model == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API key not found in environment variables (OPENAI_API_KEY).")
            sys.exit(1) # Exit if key is required and not found
        config.model.chat_api_key = api_key
        config.model.embedding_api_key = api_key
        logger.info("OpenAI API key configured.")
    elif config.model.use_model == "azure":
        if (
            not os.getenv("AZURE_OPENAI_KEY")
            or not os.getenv("AZURE_OPENAI_ENDPOINT")
        ):
            logger.error("Azure API key or endpoint not found in environment variables (AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT).")
            sys.exit(1)
        config.model.chat_api_key = os.getenv("AZURE_OPENAI_KEY")
        config.model.embedding_api_key = os.getenv("AZURE_OPENAI_KEY")
        config.model.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not config.model.azure_api_version: 
            logger.warning("Azure API version not set in config. Using default or expecting it to be pre-configured.")
        logger.info("Azure OpenAI API key and endpoint configured.")
    elif config.model.use_model == "ollama":
        if not config.model.ollama_base_url: 
             default_ollama_url = "http://localhost:11434"
             logger.warning(f"Ollama base URL not set in config. Using default: {default_ollama_url}")
             config.model.ollama_base_url = default_ollama_url
        logger.info(f"Ollama provider selected. Base URL: {config.model.ollama_base_url}")

    # Initialize models using the function from main
    chat_model, embedding_model = initialize_models(config)
    logger.info("AI models initialized successfully.")

    # Validate embedding dimension (as done in main.py)
    if embedding_model.dimension != config.vector.embedding_dim:
        error_msg = (
            f"Embedding dimension mismatch: Model dimension is {embedding_model.dimension}, "
            f"config.vector.embedding_dim is {config.vector.embedding_dim}. "
            "Please ensure the AppConfig.vector.embedding_dim matches the embedding model."
        )
        logger.error(error_msg)
        raise ConfigurationError(error_msg)
    logger.info(f"Embedding dimension validated: {embedding_model.dimension}")

    # Initialize Storage Engine
    storage_engine = SQLiteEngine(
        db_path=config.storage.db_path, embedding_dim=embedding_model.dimension
    )
    logger.info(f"Storage engine initialized with DB: {config.storage.db_path}")

    # Initialize MemoryManager
    manager = MemoryManager(
        chat_model=chat_model,
        embedding_model=embedding_model,
        storage_engine=storage_engine,
        config=config,
    )
    logger.info("MemoryManager initialized.")

except ConfigurationError as e:
    logger.error(f"Configuration error during setup: {e}")
    sys.exit(1) 
except Exception as e:
    logger.error(f"An unexpected error occurred during setup: {e}")
    sys.exit(1)

# --- Scenario 1: Short-Term vs. Long-Term Memory ---
#   - Demonstrate storing and retrieving from short-term and long-term memory.

print("\n--- Scenario 1: Short-Term vs. Long-Term Memory ---")

# Initial interaction (short-term)
response1 = manager.process_input(query="What's the weather like today?")
print(f"Short-Term Response 1: {response1}")

# Another short-term interaction
response2 = manager.process_input(query="And what about tomorrow?")
print(f"Short-Term Response 2: {response2}")


# Store a fact in long-term memory using process_input
print("\nStoring a fact into long-term memory via process_input...")
# The query itself is the "fact" we want to make memorable. The LLM will respond to it.
# This interaction (query + LLM response + concepts) gets stored.
fact_query_paris = "An important fact: The capital of France is Paris."
manager.process_input(
    query=fact_query_paris,
    memory_type="long_term" # This ensures the interaction is stored as long-term
)
print(f"Fact-storing query processed for long-term memory: \"{fact_query_paris}\"")

# Retrieve and verify (adapt verification)
print("\nAttempting to retrieve long-term memory about France's capital...")
# We use a query that should be similar to the stored fact's embedding
query_embedding_france = embedding_model.get_embedding("What is the capital of France?")
# Concepts for retrieval could also be dynamically extracted from the query
long_term_memories = manager.storage.retrieve_memories(
    query_embedding=query_embedding_france,
    concepts=["capital", "France", "Paris", "fact"], # Using relevant concepts
    memory_type="long_term", # Filter for long-term
    limit=5
)
print(f"Retrieved Long-Term Memories (first 5 matching concepts/vector):")
found_paris_fact = False
for mem in long_term_memories:
    print(f"  - ID: {mem['id']}, Prompt: {mem['prompt'][:60]}..., Score: {mem.get('relevance_score', 'N/A')}")
    if "capital of france is paris" in mem['prompt'].lower():
        found_paris_fact = True
if found_paris_fact:
    print("Found a relevant memory matching 'capital of France is Paris'.")
else:
    print("Could not find the specific Paris fact in top retrieved long-term memories. Retrieval depends on LLM behavior, embedding similarity, and concept matching.")


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

# Retrieve user1's preferences.
# The LLM should use the stored memory as context.
print(f"User1 LLM Response 2 (asking fav color): {response_user1_2}")
if "blue" in response_user1_2.lower():
    print("User1's preference for 'blue' was likely recalled by the LLM based on memory context.")
else:
    print("User1's preference was not explicitly in the LLM's response, but the interaction was stored and used as context.")


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

# Session context is implicitly used by the LLM in process_input for the same session_id.
# Verification here is that the conversation flows.
print(f"Session1 LLM Response 2 (Python benefits): {response_session1_2}")
# To explicitly check stored memories for session1:
session1_query_embedding = embedding_model.get_embedding("Tell me about Python")
session1_memories = manager.storage.retrieve_memories(
    query_embedding=session1_query_embedding,
    concepts=["python", "programming", "benefits"], # example concepts
    session_id="session1",
    memory_type="session",
    limit=3
)
print(f"Retrieved Session1 Memories (first 3):")
for mem in session1_memories:
    print(f"  - ID: {mem['id']}, Prompt: {mem['prompt'][:60]}...")

# --- Scenario 4: Agent-Specific Memory ---
#  - Show how agent memory (knowledge, adaptability) is handled.
print("\n--- Scenario 4: Agent-Specific Memory ---")

# Store agent knowledge using process_input
print("\nStoring a fact into agent memory via process_input...")
agent_fact_query = "Agent Information: My primary directive is to be helpful and assist users."
manager.process_input(
    query=agent_fact_query,
    memory_type="agent"
)
print(f"Agent fact-storing query processed: \"{agent_fact_query}\"")

# Simulate a query that might use agent memory
response_agent = manager.process_input(
    query="What is your primary directive as an agent?", # More natural query
)
print(f"Agent LLM Response (directive query): {response_agent}")
if "helpful" in response_agent.lower() or "assist users" in response_agent.lower():
    print("Agent's directive was likely recalled by the LLM based on memory context.")
else:
    print("Agent's directive was not explicitly in the LLM's response, but interaction was stored and used as context.")

# --- Scenario 5: Concept Graph & Spreading Activation ---
print("\n--- Scenario 5: Concept Graph & Spreading Activation ---")

print("\nStoring interactions to build concept graph...")
manager.process_input(query="Artificial intelligence is a broad field that encompasses many disciplines.")
manager.process_input(query="Machine learning is a subfield of artificial intelligence focusing on algorithms that learn from data.")
manager.process_input(query="Deep learning is a specialized type of machine learning using neural networks with many layers.")
print("Interactions stored.")

# Get related concepts for "artificial intelligence"
# Note: The quality of concepts depends on the LLM's `extract_concepts` method.
print("\nRetrieving related concepts for 'artificial_intelligence' (if extracted by LLM and graph built):")
# Assuming 'artificial_intelligence' or similar concepts are extracted by the LLM and stored by SQLiteEngine
related_concepts = manager.concept_graph.get_related_concepts("artificial_intelligence", depth=2)
if related_concepts:
    print(f"Concepts related to 'artificial_intelligence' (or similar): {related_concepts}")
else:
    print("No direct concepts named 'artificial_intelligence' found or no relations. This depends on LLM's concept extraction and graph construction.")


# --- Scenario 6: Semantic Clustering ---
# Clustering is now triggered by a threshold in MemoryManager.
# We can add more interactions to potentially trigger it if the threshold is high.
print("\n--- Scenario 6: Semantic Clustering ---")
print(f"Clustering will be triggered automatically every {config.memory.clustering_trigger_threshold} interactions.")
print("Adding a few more interactions to potentially trigger clustering (if threshold is low):")
for i in range(5):
    manager.process_input(f"This is interaction number {i+1} for semantic clustering test purposes.")
print(f"Total interactions processed in this example run: {manager.interactions_since_last_clustering} (approx, as it resets).")
print("If clustering threshold was met, it would have run. Check logs if enabled for clustering messages.")

# --- Scenario 7: Vector-Based Similarity Search ---
print("\n--- Scenario 7: Vector-Based Similarity Search ---")

print("\nStoring diverse facts for similarity search using process_input...")
manager.process_input(query="The currency of Japan is the Yen.", memory_type="short_term")
manager.process_input(query="The highest mountain in the world is Mount Everest.", memory_type="short_term")
manager.process_input(query="The chemical symbol for water is H2O.", memory_type="short_term")
print("Facts stored via process_input.")

# Similar query
similar_query = "What is the currency used in Japan?"
print(f"\nPerforming similarity search for: \"{similar_query}\"")

# Retrieve memories based on this similar query
# Concepts for retrieval should ideally be extracted from the similar_query by the LLM.
similar_query_embedding = embedding_model.get_embedding(similar_query)
similar_query_concepts = chat_model.extract_concepts(similar_query) # Let LLM extract concepts
logger.info(f"Concepts extracted for similarity search query '{similar_query}': {similar_query_concepts}")

retrieved_memories_vector = manager.storage.retrieve_memories(
    query_embedding=similar_query_embedding,
    concepts=similar_query_concepts,
    limit=5 # Retrieve top 5 relevant
)
print(f"Retrieved Memories for '{similar_query}' (first 5, based on vector + concept score):")
found_yen_fact = False
for mem in retrieved_memories_vector:
    print(f"  - ID: {mem['id']}, Prompt: {mem['prompt'][:70]}..., Score: {mem.get('relevance_score', 'N/A')}")
    if "yen" in mem['prompt'].lower() and "japan" in mem['prompt'].lower():
        found_yen_fact = True
if found_yen_fact:
    print("Found a relevant memory regarding Japan's currency.")
elif retrieved_memories_vector:
    print("Relevant memory not definitively found in top results, but some memories were retrieved.")
else:
    print("No memories retrieved. This could be due to various factors including LLM concept extraction, embedding differences, or relevance scoring.")

# --- Cleanup ---
print("\n--- Cleanup ---")
# Delete the test database file.
if os.path.exists(config.storage.db_path):
    try:
        # Attempt to close the database connection if SQLiteEngine has a close method
        if hasattr(storage_engine, 'close'):
             storage_engine.close() # Assuming a close method exists or will be added
        elif hasattr(storage_engine, '_close_db'): # For older versions if any
             storage_engine._close_db()

        os.remove(config.storage.db_path)
        logger.info(f"Test database '{config.storage.db_path}' cleaned up successfully.")
        print(f"Test database '{config.storage.db_path}' cleaned up.")
    except Exception as e:
        logger.error(f"Error cleaning up test database: {e}")
        print(f"Error cleaning up test database: {e} (It might be in use. Manual deletion might be needed.)")
else:
    logger.info("Test database file not found, no cleanup needed.")
    print("Test database file not found, no cleanup needed.")

logger.info("MindForge Full Example finished.")