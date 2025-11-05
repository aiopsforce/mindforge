# MindForge Architecture

## Overview

MindForge is designed with a modular, extensible architecture that separates concerns between memory management, AI model interaction, and storage.

```
┌─────────────────────────────────────────────────────────┐
│                    User Application                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────┐
│                   MemoryManager                          │
│  • Coordinates memory operations                        │
│  • Manages memory retrieval and storage                 │
│  • Integrates chat and embedding models                 │
└──────┬─────────────┬──────────────┬─────────────────────┘
       │             │              │
       ↓             ↓              ↓
┌──────────┐  ┌──────────┐  ┌──────────────┐
│   Chat   │  │Embedding │  │   Storage    │
│  Model   │  │  Model   │  │   Engine     │
└──────────┘  └──────────┘  └──────┬───────┘
                                    │
                     ┌──────────────┼──────────────┐
                     ↓              ↓              ↓
              ┌──────────┐   ┌──────────┐  ┌──────────┐
              │ SQLite   │   │PostgreSQL│  │  Redis   │
              │  Vector  │   │  pgvector│  │  Vector  │
              └──────────┘   └──────────┘  └──────────┘
```

## Core Components

### 1. MemoryManager

The `MemoryManager` is the central orchestrator that coordinates all memory operations.

**Responsibilities:**
- Process user inputs and generate responses
- Extract concepts from queries using the chat model
- Generate embeddings using the embedding model
- Retrieve relevant memories from storage
- Store new interactions
- Trigger clustering and graph updates

**Key Methods:**
- `process_input()`: Main entry point for processing queries
- `_build_context()`: Constructs context from retrieved memories
- `_store_interaction()`: Persists new interactions

### 2. Model Layer

The model layer provides abstractions for different LLM providers.

#### Base Classes

**BaseChatModel**
- `generate_response(context, query)`: Generate AI responses
- `extract_concepts(text)`: Extract key concepts from text

**BaseEmbeddingModel**
- `get_embedding(text)`: Generate vector embeddings
- `dimension`: Property returning embedding dimensions

#### Implementations

1. **OpenAI Models**
   - Uses OpenAI's latest API (v1.0+)
   - Supports GPT-3.5, GPT-4, and embedding models

2. **Azure OpenAI Models**
   - Compatible with Azure's OpenAI service
   - Supports deployment-based model access

3. **Ollama Models**
   - Enables local model usage
   - Great for privacy-sensitive applications

4. **LiteLLM Models**
   - Universal adapter for any LLM provider
   - Supports 100+ models from various providers

### 3. Storage Layer

The storage layer handles persistence and retrieval of memories.

#### BaseStorage Interface

```python
class BaseStorage(ABC):
    def store_memory(memory_data, memory_type, user_id, session_id)
    def retrieve_memories(query_embedding, concepts, memory_type, user_id, session_id, limit)
    def update_memory_level(memory_id, new_memory_level, user_id, session_id)
```

#### SQLiteEngine

The default storage implementation using SQLite with vector search.

**Features:**
- FAISS-based or sqlite-vec vector indexing
- Multi-level memory types
- Concept graph storage
- Session and user memory tracking

**Schema:**
```sql
memories (
    id, prompt, response, timestamp,
    access_count, last_access, memory_type, recency_boost
)

memory_vectors (
    rowid, embedding
)

concepts (
    id, memory_id, concept, weight
)

user_memories (
    user_id, memory_id, preference, history
)

session_memories (
    session_id, memory_id, recent_activity, context
)

agent_memories (
    memory_id, knowledge, adaptability
)
```

#### Alternative Storage Engines

**SQLiteVecEngine**
- Optimized for vector search performance
- Uses sqlite-vec extension

**PostgresVectorEngine**
- Production-ready for multi-process applications
- Uses pgvector extension
- Supports concurrent access

**RedisVectorEngine**
- In-memory storage for ultra-fast retrieval
- Great for high-throughput applications
- Supports distributed deployments

**ChromaDBEngine**
- Document-oriented vector storage
- Built-in embedding support
- Easy integration with ML workflows

### 4. Memory Types

MindForge implements a sophisticated multi-level memory hierarchy:

#### Short-term Memory
- **Purpose**: Recent context in conversations
- **Characteristics**: Limited retention, fast decay
- **Use cases**: Chatbots, conversational AI

#### Long-term Memory
- **Purpose**: Persistent knowledge and facts
- **Characteristics**: Permanent storage, no decay
- **Use cases**: Knowledge bases, fact retention

#### User-specific Memory
- **Purpose**: Personalization per user
- **Characteristics**: Isolated per user_id
- **Use cases**: Multi-user applications, personalization

#### Session-specific Memory
- **Purpose**: Context within a conversation
- **Characteristics**: Scoped to session_id
- **Use cases**: Conversation threads, task continuity

#### Agent-specific Memory
- **Purpose**: Agent's self-knowledge
- **Characteristics**: Agent's capabilities and identity
- **Use cases**: Agent introspection, capability awareness

### 5. Concept Graph

The concept graph tracks relationships between concepts for enhanced retrieval.

**Features:**
- Tracks co-occurrence of concepts
- Weighted edges based on frequency
- Spreading activation for semantic search
- Temporal decay of relationships

**Benefits:**
- Improves context relevance
- Discovers related memories
- Enhances semantic understanding

### 6. Semantic Clustering

Automatic clustering of memories based on semantic similarity.

**Process:**
1. Periodically triggered after N interactions
2. Groups memories by embedding similarity
3. Identifies memory themes and topics
4. Enables topic-based retrieval

**Algorithms:**
- K-means clustering
- Hierarchical clustering
- DBSCAN for outlier detection

## Data Flow

### Memory Storage Flow

```
User Input
    ↓
Generate Embedding (Embedding Model)
    ↓
Extract Concepts (Chat Model)
    ↓
Create Memory Data
    ↓
Store in Database
    ↓
Update Concept Graph
    ↓
Check Clustering Threshold
    ↓
[Optional] Trigger Clustering
```

### Memory Retrieval Flow

```
Query Input
    ↓
Generate Query Embedding
    ↓
Extract Query Concepts
    ↓
Vector Similarity Search
    ↓
Concept Graph Expansion
    ↓
Apply Filters (type, user, session)
    ↓
Score and Rank Results
    ↓
Return Top K Memories
```

### Response Generation Flow

```
Query + Retrieved Memories
    ↓
Build Context Dictionary
    ↓
Format Context for LLM
    ↓
Generate Response (Chat Model)
    ↓
Return Response to User
```

## Configuration System

MindForge uses a hierarchical configuration system:

```python
AppConfig
├── MemoryConfig
│   ├── similarity_threshold
│   ├── short_term_limit
│   ├── long_term_limit
│   ├── decay_rate
│   └── clustering_trigger_threshold
├── VectorConfig
│   ├── embedding_dim
│   ├── index_type
│   └── max_neighbors
├── ModelConfig
│   ├── chat_model_name
│   ├── embedding_model_name
│   ├── use_model (provider)
│   └── provider-specific settings
└── StorageConfig
    ├── db_path
    ├── wal_mode
    └── cache_size
```

## Extension Points

### Adding New Model Providers

```python
from mindforge.core.base_model import BaseChatModel, BaseEmbeddingModel

class MyCustomChatModel(BaseChatModel):
    def generate_response(self, context, query):
        # Your implementation
        pass

    def extract_concepts(self, text):
        # Your implementation
        pass
```

### Adding New Storage Backends

```python
from mindforge.storage.base_storage import BaseStorage

class MyCustomStorage(BaseStorage):
    def store_memory(self, memory_data, memory_type, user_id, session_id):
        # Your implementation
        pass

    def retrieve_memories(self, query_embedding, concepts, ...):
        # Your implementation
        pass
```

### Custom Memory Types

Memory types are extensible through the database schema. Add new types by:

1. Updating the CHECK constraint in the memories table
2. Creating corresponding junction tables
3. Implementing storage/retrieval logic

## Performance Considerations

### Vector Search Optimization

- **Indexing**: Use FAISS or specialized vector databases for large datasets
- **Dimensionality Reduction**: Consider PCA or quantization for reduced storage
- **Caching**: Implement embedding caching for frequently queried items

### Scaling Strategies

1. **Horizontal Scaling**: Use Redis or PostgreSQL for distributed deployments
2. **Vertical Scaling**: Increase vector index size and cache
3. **Sharding**: Partition memories by user_id or time ranges

### Memory Management

- **Cleanup**: Implement periodic cleanup of old short-term memories
- **Archival**: Move old memories to cold storage
- **Compression**: Use vector quantization for storage efficiency

## Security Considerations

- **API Keys**: Never hardcode; use environment variables
- **User Isolation**: Ensure user_id filtering is enforced
- **SQL Injection**: Use parameterized queries (already implemented)
- **Rate Limiting**: Implement at application level
- **Data Privacy**: Consider encryption at rest for sensitive memories

## Testing Strategy

MindForge includes comprehensive tests:

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflows
- **Storage Tests**: Database operations
- **Model Tests**: Mock LLM responses for deterministic testing

Run tests with:
```bash
pytest tests/
```

## Future Enhancements

Potential areas for extension:

1. **Multi-modal Memory**: Support for images, audio
2. **Federated Learning**: Distributed memory updates
3. **Memory Compression**: Summarization of old memories
4. **Active Learning**: Smart memory prioritization
5. **Memory Visualization**: Graph visualization tools
6. **Cross-lingual Memory**: Multi-language support
