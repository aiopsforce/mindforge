# MindForge
MindForge is a Python library designed to provide sophisticated memory management capabilities for AI agents and models. It combines vector-based similarity search, concept graphs, and multi-level memory structures (short-term, long-term, user-specific, session-specific, and agent-specific) to enable more context-aware and adaptive AI responses.

### *** Work In Progress ***

## Features

*   **Multi-Level Memory:**  Organizes memories into different levels, including:
    *   **Short-Term Memory:**  For recent interactions.
    *   **Long-Term Memory:**  For persistent knowledge.
    *   **User-Specific Memory:**  Tailored to individual users.
    *   **Session-Specific Memory:**  Contextual information for a single session.
    *   **Agent-Specific Memory:**  Knowledge and adaptability specific to the AI agent.
*   **Vector-Based Similarity Search:**  Uses `sqlite-vec` (and optionally FAISS) for fast and efficient retrieval of memories based on semantic similarity.
*   **Concept Graph:**  Builds and maintains a graph of relationships between concepts, enabling spreading activation for enhanced retrieval.
*   **Semantic Clustering:**  Groups memories based on their embeddings, improving retrieval efficiency and identifying related concepts.
*   **Flexible Storage:**  Provides `SQLiteEngine` and `SQLiteVecEngine` for persistent storage, with options for optimizing performance.
*   **Model Agnostic:**  Supports OpenAI, Azure OpenAI, and Ollama models for chat and embedding generation, with an easily extensible interface for adding other models.
*   **Built-in Utilities:**  Includes tools for logging, monitoring, vector optimization, profiling, and input validation.
*   **Configurable:** Uses dataclasses for easy configuration of memory, vector, model, and storage parameters.

## Installation

```bash
pip install mindforge
```

Or install from source:

```bash
git clone https://github.com/yourusername/mindforge.git
cd mindforge
pip install -e .
```

## Dependencies:

```bash
openai (for OpenAI models)
azure-openai (for Azure OpenAI models)
requests (for Ollama models)
sqlite-vec (for vector search)
faiss-cpu (for FAISS-based MemoryStore)
scikit-learn (for clustering and vector optimization)
scipy (for vector optimization)

pip install openai azure-openai requests sqlite-vec faiss-cpu scikit-learn scipy
```

## Quick Start

```python
from mindforge import MemoryManager
from mindforge.models.chat import OpenAIChatModel
from mindforge.models.embedding import OpenAIEmbeddingModel

# Initialize models
chat_model = OpenAIChatModel(api_key="your-openai-key")
embedding_model = OpenAIEmbeddingModel(api_key="your-openai-key")

# Create memory manager
manager = MemoryManager(
    chat_model=chat_model,
    embedding_model=embedding_model,
    db_path="mindforge.db"
)

# Process a query
response = manager.process_input(
    query="What is machine learning?",
    user_id="user123",
    session_id="session456"
)
print(response)
```

## Advanced Usage
```python
import os
from mindforge import MemoryManager
from mindforge.models.chat import OpenAIChatModel
from mindforge.models.embedding import OpenAIEmbeddingModel
from mindforge.config import AppConfig

# --- Example 1: Using OpenAI Models (requires OPENAI_API_KEY) ---

# Set your OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

# Initialize the configuration (using default settings)
config = AppConfig()

# Initialize the chat and embedding models
chat_model = OpenAIChatModel(api_key=config.model.chat_api_key)
embedding_model = OpenAIEmbeddingModel(api_key=config.model.embedding_api_key)

# Initialize the MemoryManager
manager = MemoryManager(
    chat_model=chat_model, embedding_model=embedding_model, config=config
)

# Process a query
response = manager.process_input(query="What is the capital of France?")
print(f"Response: {response}")

# --- Example 2: Using Azure OpenAI Models ---

# Set your Azure OpenAI API key and endpoint as environment variables (or in config)
# os.environ["AZURE_OPENAI_API_KEY"] = "your_azure_openai_api_key"
# os.environ["AZURE_OPENAI_ENDPOINT"] = "your_azure_openai_endpoint"

config = AppConfig()
config.model.use_model = "azure"
config.model.chat_api_key = "your_azure_openai_api_key"  # Replace with your key
config.model.embedding_api_key = "your_azure_openai_api_key"  # Replace with your key
config.model.azure_endpoint = "your_azure_openai_endpoint"  # Replace with endpoint
config.model.azure_api_version = "2024-02-15-preview"
config.model.chat_model_name = "your-chat-deployment-name"  # Replace
config.model.embedding_model_name = "your-embedding-deployment-name"  # Replace

# Initialize Azure models (assuming you've set the environment variables)
from mindforge.models.chat import AzureChatModel
from mindforge.models.embedding import AzureEmbeddingModel

chat_model = AzureChatModel(
    api_key=config.model.chat_api_key,
    endpoint=config.model.azure_endpoint,
    deployment_name=config.model.chat_model_name,
    api_version=config.model.azure_api_version,
)
embedding_model = AzureEmbeddingModel(
    api_key=config.model.embedding_api_key,
    endpoint=config.model.azure_endpoint,
    deployment_name=config.model.embedding_model_name,
    api_version=config.model.azure_api_version,
)

manager = MemoryManager(
    chat_model=chat_model, embedding_model=embedding_model, config=config
)

response = manager.process_input(query="What is machine learning?")
print(f"Response: {response}")


# --- Example 3: Using Ollama Models ---

config = AppConfig()
config.model.use_model = "ollama"
config.model.chat_model_name = "llama2"  # Or your preferred Ollama model
config.model.embedding_model_name = "llama2"
config.model.ollama_base_url = "http://localhost:11434"  # Default Ollama URL

from mindforge.models.chat import OllamaChatModel
from mindforge.models.embedding import OllamaEmbeddingModel

chat_model = OllamaChatModel(
    model_name=config.model.chat_model_name, base_url=config.model.ollama_base_url
)
embedding_model = OllamaEmbeddingModel(
    model_name=config.model.embedding_model_name,
    base_url=config.model.ollama_base_url,
)

manager = MemoryManager(
    chat_model=chat_model, embedding_model=embedding_model, config=config
)

response = manager.process_input(query="Explain quantum physics.")
print(f"Response: {response}")

# --- Example 4:  Using MemoryStore (FAISS) ---
from mindforge.core.memory_store import MemoryStore
import numpy as np

# Initialize the MemoryStore
memory_store = MemoryStore(dimension=1536)  # Match your embedding dimension

# Add some interactions
interaction1 = {
    "id": "1",
    "embedding": np.random.rand(1536),
    "text": "This is the first interaction.",
    "concepts": ["interaction", "first"],
    "timestamp": 1678886400.0
}
interaction2 = {
    "id": "2",
    "embedding": np.random.rand(1536),
    "text": "This is the second interaction, about cats.",
    "concepts": ["interaction", "second", "cats"],
    "timestamp": 1678886460.0
}

memory_store.add_interaction(interaction1)
memory_store.add_interaction(interaction2, memory_level="user") # Add to user memory

# Retrieve relevant interactions
query_embedding = np.random.rand(1536)
query_concepts = ["interaction"]
retrieved = memory_store.retrieve(query_embedding, query_concepts, memory_level="user") # Retrieve from user memory
print(f"Retrieved interactions: {retrieved}")
```

## MemoryManager
The MemoryManager is the central class for processing user input and managing memories.

# Initialize MemoryManager (see Quick Start for model initialization)
manager = MemoryManager(chat_model, embedding_model, config)

# Process user input
```python
response = manager.process_input(
    query="What is the meaning of life?",
    user_id="user123",      # Optional:  Associate with a specific user
    session_id="session456", # Optional:  Associate with a specific session
    memory_type="short_term" # Optional: Specify memory type (short_term, long_term, user, session, agent)
)
print(response)
```

Generates an embedding of the query using the embedding_model.
Extracts key concepts from the query using the chat_model.
Retrieves relevant memories from the storage (using SQLiteEngine or SQLiteVecEngine). You can filter by memory_type, user_id, and session_id.
Builds a context dictionary from the retrieved memories.
Generates a response using the chat_model, passing the context and query.
Stores the interaction (query, response, embedding, concepts) in the storage.
Updates the concept graph and semantic clusters.

## MemoryStore
The MemoryStore class provides a more direct way to interact with the memory storage, using FAISS for vector indexing.

from mindforge.core.memory_store import MemoryStore
import numpy as np

# Initialize (dimension must match your embedding model)
memory_store = MemoryStore(dimension=1536)

# Add interactions
interaction1 = {"id": "1", "embedding": np.random.rand(1536), "text": "...", "concepts": ["..."]}
interaction2 = {"id": "2", "embedding": np.random.rand(1536), "text": "...", "concepts": ["..."]}
memory_store.add_interaction(interaction1)
memory_store.add_interaction(interaction2, memory_level="user")  # Add to user-specific memory

# Retrieve memories
query_embedding = np.random.rand(1536)
query_concepts = ["concept1", "concept2"]
results = memory_store.retrieve(query_embedding, query_concepts, memory_level="user", similarity_threshold=0.8)
print(results)


## SQLiteEngine and SQLiteVecEngine
These classes provide persistent storage for memories, using SQLite and sqlite-vec for vector search. SQLiteEngine offers more features (user, session, agent memories, concept graph updates), while SQLiteVecEngine is optimized for vector search performance. You typically interact with them through the MemoryManager. However, you can use them directly:

from mindforge.storage.sqlite_engine import SQLiteEngine
import numpy as np

# Initialize
engine = SQLiteEngine(db_path="my_memories.db", embedding_dim=1536)

# Store a memory
memory_data = {
    "id": "unique_id",
    "prompt": "What is the capital of Australia?",
    "response": "Canberra",
    "embedding": np.random.rand(1536).tolist(),  # Store as list
    "concepts": ["capital", "Australia"],
}
engine.store_memory(memory_data, memory_type="long_term", user_id="user42")

# Retrieve memories
query_embedding = np.random.rand(1536)
memories = engine.retrieve_memories(
    query_embedding, concepts=["capital"], memory_type="long_term", user_id="user42"
)
print(memories)


## Configuration
Configuration
MindForge uses dataclasses for configuration:

from mindforge.config import AppConfig, MemoryConfig, ModelConfig

# Use default configuration
config = AppConfig()

# Customize specific settings
config.memory.similarity_threshold = 0.8
config.model.chat_model_name = "gpt-3.5-turbo"
config.storage.db_path = "custom_database.db"

# Create a completely custom configuration
custom_memory_config = MemoryConfig(short_term_limit=500, decay_rate=0.05)
custom_model_config = ModelConfig(chat_model_name="llama2", use_model="ollama")
custom_config = AppConfig(memory=custom_memory_config, model=custom_model_config)

from mindforge.utils.logging import LogManager

# Initialize (usually done in main.py)
log_manager = LogManager(log_dir="my_logs", log_level="DEBUG")
logger = log_manager.get_logger("my_module")

logger.info("This is an informational message.")
logger.debug("This is a debug message.")
logger.error("This is an error message.")

# You can also configure the log level through AppConfig:
config = AppConfig(log_level="WARNING")


## Error Handling

```bash
MindForge defines custom exception classes:
MindForgeError: Base exception.
ConfigurationError: For configuration issues.
ModelError: For errors related to AI models.
StorageError: For storage-related errors.
ValidationError: For input validation errors.
MemoryError: For memory-related errors.
```

These exceptions are used throughout the library to provide more specific error information.


## Advanced Usage
Concept Graph: The ConceptGraph class manages relationships between concepts. The MemoryManager automatically updates the graph. You can access it directly for more advanced analysis:
```python
from mindforge.utils.graph import ConceptGraph

graph = ConceptGraph(engine)  # Pass your storage engine
related_concepts = graph.get_related_concepts("machine_learning")
print(related_concepts)
```


## Clustering: The MemoryClustering class clusters memories based on their embeddings. The MemoryManager periodically updates the clusters. You can use this class to perform clustering manually:

```python
from mindforge.utils.clustering import MemoryClustering

clustering = MemoryClustering(engine)
clustering.cluster_memories(n_clusters=20)  # Specify the number of clusters
```

## Vector Optimization: The VectorOptimizer class provides utilities for compressing and quantizing embeddings, which can improve storage efficiency and retrieval speed.

```python
from mindforge.utils.optimization import VectorOptimizer
import numpy as np

embeddings = np.random.rand(100, 1536)
compressed = VectorOptimizer.compress_embeddings(embeddings, target_dim=256)
quantized = VectorOptimizer.quantize_vectors(embeddings, bits=8)
```


## Custom Models: You can easily add support for other chat and embedding models by creating classes that implement the BaseChatModel and BaseEmbeddingModel interfaces.

```python
Profiling: Use the profile decorator to profile specific functions:

from mindforge.utils.profiling import profile

@profile(output_file="my_function_profile.txt")
def my_function():
    # ... your code ...
    pass
```

## Contributing
Contributions are welcome! Please see the project's GitHub repository for guidelines. This includes bug reports, feature requests, and code contributions.