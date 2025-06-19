
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MemoryConfig:
    """Configuration for memory management."""

    short_term_limit: int = 1000  # Max number of entries in short-term memory (Note: Not strictly enforced by current SQLiteEngine; primarily for alternative storage or future eviction policies).
    long_term_limit: int = 10000  # Max number of entries in long-term memory (Note: Not strictly enforced by current SQLiteEngine; primarily for alternative storage or future eviction policies).
    similarity_threshold: float = 0.7
    decay_rate: float = 0.1  # Rate at which memory relevance decays over time (Note: Current SQLiteEngine uses recency_boost on access; true time-based decay is a future enhancement).
    reinforcement_rate: float = 0.1  # Rate at which memory relevance is reinforced (Note: Partially addressed by recency_boost; more explicit reinforcement is a future enhancement).
    min_access_for_long_term: int = 5  # Minimum access count to promote a memory to long-term (Note: Memory promotion not yet implemented).
    clustering_trigger_threshold: int = 100


@dataclass
class VectorConfig:
    """Configuration for vector operations."""

    embedding_dim: int = 1536
    index_type: str = "l2"  # Type of index for vector search (e.g., "l2", "cosine"). (Note: sqlite-vec handles this internally based on embedding type; more relevant for other vector DBs like FAISS).
    batch_size: int = 32  # Batch size for embedding generation or batch database operations (Note: Current implementation processes interactively; relevant for future batch processing features).
    max_neighbors: int = 100  # Maximum number of neighbors to retrieve in vector search (Note: Retrieval limit is currently set per query in MemoryManager/SQLiteEngine; this could be a default for index configuration in other vector DBs).


@dataclass
class ModelConfig:
    """Configuration for AI models."""

    chat_api_key: Optional[str] = None
    embedding_api_key: Optional[str] = None
    chat_model_name: str = "gpt-4"
    embedding_model_name: str = "text-embedding-3-small"
    azure_endpoint: Optional[str] = None
    azure_api_version: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    litellm_base_url: Optional[str] = None
    use_model: str = "openai"  # Added:  Specify which provider to use (openai, azure, ollama, litellm)


@dataclass
class StorageConfig:
    """Configuration for storage."""

    db_path: str = "mindforge.db"
    wal_mode: bool = True
    journal_mode: str = "WAL"
    cache_size: int = -2000  # 2GB in memory
    temp_store: int = 2  # Memory temp store


@dataclass
class AppConfig:
    """Overall application configuration."""

    memory: MemoryConfig = field(default_factory=MemoryConfig)
    vector: VectorConfig = field(default_factory=VectorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    log_level: str = "INFO"  # Added log level
    log_dir: str = "logs"  # Added log directory
