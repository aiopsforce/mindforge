
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MemoryConfig:
    """Configuration for memory management."""

    short_term_limit: int = 1000
    long_term_limit: int = 10000
    similarity_threshold: float = 0.7
    decay_rate: float = 0.1
    reinforcement_rate: float = 0.1
    min_access_for_long_term: int = 5


@dataclass
class VectorConfig:
    """Configuration for vector operations."""

    embedding_dim: int = 1536
    index_type: str = "l2"  # Consider using "cosine" for cosine similarity
    batch_size: int = 32
    max_neighbors: int = 100


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
    use_model: str = "openai"  # Added:  Specify which provider to use (openai, azure, ollama)


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
