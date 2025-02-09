"""
MindForge: A memory management and retrieval system for AI models.

This package provides tools for storing, retrieving, and managing memories 
for AI agents, including short-term and long-term memory, concept graphs, 
and vector-based similarity search.
"""

from .core.memory_manager import MemoryManager
from .core.memory_store import MemoryStore
from .storage.sqlite_engine import SQLiteEngine
from .storage.sqlite_vec_engine import SQLiteVecEngine

__version__ = "0.0.1"

__all__ = [
    "MemoryManager",
    "MemoryStore",
    "SQLiteEngine",
    "SQLiteVecEngine",
]