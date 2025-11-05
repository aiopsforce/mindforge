# Getting Started with MindForge

## What is MindForge?

MindForge is a sophisticated memory management system for AI agents and language models. It provides:

- **Multi-level memory management**: Short-term, long-term, user-specific, session-specific, and agent-specific memory
- **Vector-based similarity search**: Fast retrieval of relevant memories using embeddings
- **Concept graphs**: Semantic relationships between concepts for enhanced context
- **Multi-provider support**: Works with OpenAI, Azure OpenAI, Ollama, and any LiteLLM-supported provider
- **Flexible storage backends**: SQLite, PostgreSQL, Redis, and ChromaDB support

## Installation

### Using pip

```bash
pip install mindforge
```

### From source

```bash
git clone https://github.com/yourusername/mindforge.git
cd mindforge
pip install -e .
```

## Quick Start

### 1. Set up your environment

```bash
export OPENAI_API_KEY='your-openai-api-key'
```

### 2. Create a simple memory-enabled AI agent

```python
import os
from mindforge import MemoryManager
from mindforge.models.chat import OpenAIChatModel
from mindforge.models.embedding import OpenAIEmbeddingModel
from mindforge.storage.sqlite_engine import SQLiteEngine
from mindforge.config import AppConfig

# Initialize configuration
config = AppConfig()

# Get API key
api_key = os.getenv("OPENAI_API_KEY")

# Initialize models
chat_model = OpenAIChatModel(api_key=api_key, model_name="gpt-3.5-turbo")
embedding_model = OpenAIEmbeddingModel(api_key=api_key)

# Initialize storage
storage = SQLiteEngine(db_path="my_agent.db", embedding_dim=1536)

# Create memory manager
manager = MemoryManager(
    chat_model=chat_model,
    embedding_model=embedding_model,
    storage_engine=storage,
    config=config
)

# Use it!
response = manager.process_input("Hello! My name is Alice.")
print(response)

response = manager.process_input("What's my name?")
print(response)  # Should remember "Alice"
```

## Core Concepts

### Memory Types

MindForge supports multiple memory types:

#### 1. **Short-term Memory** (default)
Stores recent interactions temporarily. Great for maintaining context in a conversation.

```python
manager.process_input("I just had coffee.", memory_type="short_term")
```

#### 2. **Long-term Memory**
Persists important information indefinitely. Use for facts and knowledge.

```python
manager.process_input(
    "The project deadline is December 31st.",
    memory_type="long_term"
)
```

#### 3. **User-specific Memory**
Maintains personalized information for different users.

```python
manager.process_input(
    "My favorite color is blue.",
    user_id="user123",
    memory_type="user"
)
```

#### 4. **Session-specific Memory**
Keeps track of context within a single conversation session.

```python
manager.process_input(
    "Let's discuss Python programming.",
    session_id="session456",
    memory_type="session"
)
```

#### 5. **Agent-specific Memory**
Stores knowledge about the AI agent itself.

```python
manager.process_input(
    "You are an expert in data science.",
    memory_type="agent"
)
```

### Configuration

MindForge uses dataclasses for easy configuration:

```python
from mindforge.config import AppConfig, MemoryConfig, ModelConfig

# Use defaults
config = AppConfig()

# Customize memory settings
config.memory.similarity_threshold = 0.8
config.memory.short_term_limit = 500

# Customize model settings
config.model.chat_model_name = "gpt-4"
config.model.embedding_model_name = "text-embedding-3-large"

# Customize storage
config.storage.db_path = "custom_database.db"
```

### Supported Model Providers

#### OpenAI

```python
from mindforge.models.chat import OpenAIChatModel
from mindforge.models.embedding import OpenAIEmbeddingModel

chat_model = OpenAIChatModel(api_key="your-key", model_name="gpt-4")
embedding_model = OpenAIEmbeddingModel(api_key="your-key")
```

#### Azure OpenAI

```python
from mindforge.models.chat import AzureChatModel
from mindforge.models.embedding import AzureEmbeddingModel

chat_model = AzureChatModel(
    api_key="your-key",
    endpoint="https://your-resource.openai.azure.com/",
    deployment_name="gpt-35-turbo",
    api_version="2024-02-15-preview"
)

embedding_model = AzureEmbeddingModel(
    api_key="your-key",
    endpoint="https://your-resource.openai.azure.com/",
    deployment_name="text-embedding-ada-002",
    api_version="2024-02-15-preview"
)
```

#### Ollama (Local Models)

```python
from mindforge.models.chat import OllamaChatModel
from mindforge.models.embedding import OllamaEmbeddingModel

chat_model = OllamaChatModel(
    model_name="llama2",
    base_url="http://localhost:11434"
)

embedding_model = OllamaEmbeddingModel(
    model_name="llama2",
    base_url="http://localhost:11434"
)
```

#### LiteLLM (Universal)

```python
from mindforge.models.chat import LiteLLMChatModel
from mindforge.models.embedding import LiteLLMEmbeddingModel

chat_model = LiteLLMChatModel(
    model_name="claude-3-opus-20240229",
    api_key="your-anthropic-key"
)

embedding_model = LiteLLMEmbeddingModel(
    model_name="text-embedding-3-small",
    api_key="your-openai-key",
    dimension=1536
)
```

### Storage Backends

#### SQLite (Default)

```python
from mindforge.storage.sqlite_engine import SQLiteEngine

storage = SQLiteEngine(db_path="mindforge.db", embedding_dim=1536)
```

#### SQLite with vec0 extension

```python
from mindforge.storage.sqlite_vec_engine import SQLiteVecEngine

storage = SQLiteVecEngine(db_path="mindforge.db", embedding_dim=1536)
```

#### PostgreSQL with pgvector

```python
from mindforge.storage.postgres_vec_engine import PostgresVectorEngine

storage = PostgresVectorEngine(
    host="localhost",
    port=5432,
    database="mindforge",
    user="postgres",
    password="your-password",
    embedding_dim=1536
)
```

#### Redis with vector search

```python
from mindforge.storage.redis_vec_engine import RedisVectorEngine

storage = RedisVectorEngine(
    host="localhost",
    port=6379,
    embedding_dim=1536
)
```

#### ChromaDB

```python
from mindforge.storage.chroma_engine import ChromaDBEngine

storage = ChromaDBEngine(
    persist_directory="./chroma_db",
    collection_name="mindforge",
    embedding_dim=1536
)
```

## Next Steps

- Check out the [examples](../examples) directory for more detailed use cases
- Read the [Architecture Guide](./ARCHITECTURE.md) to understand how MindForge works
- Explore the [API Reference](./API_REFERENCE.md) for detailed documentation

## Common Patterns

### Building a Conversational AI

```python
# Initialize once
manager = MemoryManager(...)

# In your conversation loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit']:
        break

    response = manager.process_input(
        query=user_input,
        user_id="user123",
        session_id="current_session",
        memory_type="session"
    )

    print(f"AI: {response}")
```

### Multi-user Application

```python
def handle_user_request(user_id: str, message: str):
    response = manager.process_input(
        query=message,
        user_id=user_id,
        memory_type="user"
    )
    return response

# Different users get personalized responses
response1 = handle_user_request("alice", "What's my favorite food?")
response2 = handle_user_request("bob", "What's my favorite food?")
```

### Knowledge Base

```python
# Store knowledge
manager.process_input(
    "The company was founded in 2020.",
    memory_type="long_term"
)

manager.process_input(
    "Our main product is an AI assistant.",
    memory_type="long_term"
)

# Query knowledge
response = manager.process_input(
    "When was the company founded?",
    memory_type="long_term"
)
```

## Troubleshooting

### Import Errors

Make sure you've installed all dependencies:

```bash
pip install -e .
```

### API Key Issues

Verify your API key is set correctly:

```python
import os
print(os.getenv("OPENAI_API_KEY"))  # Should print your key
```

### Database Locked

If using SQLite, make sure only one process is accessing the database at a time. Consider using PostgreSQL or Redis for multi-process applications.

### Embedding Dimension Mismatch

Ensure your storage dimension matches your embedding model:

```python
# For text-embedding-3-small (1536 dimensions)
storage = SQLiteEngine(db_path="mindforge.db", embedding_dim=1536)

# For other models, check their documentation
```

## Support

- Report issues on [GitHub](https://github.com/yourusername/mindforge/issues)
- Check the [examples](../examples) for common use cases
- Read the documentation for detailed information
