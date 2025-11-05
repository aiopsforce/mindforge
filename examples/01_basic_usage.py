"""
Basic Usage Example for MindForge
==================================
This example demonstrates the simplest way to use MindForge for AI memory management.
"""

import os
from mindforge import MemoryManager
from mindforge.models.chat import OpenAIChatModel
from mindforge.models.embedding import OpenAIEmbeddingModel
from mindforge.storage.sqlite_engine import SQLiteEngine
from mindforge.config import AppConfig

def main():
    print("=" * 60)
    print("MindForge - Basic Usage Example")
    print("=" * 60)

    # Step 1: Set up configuration
    config = AppConfig()

    # Step 2: Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nError: OPENAI_API_KEY environment variable not set!")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return

    # Step 3: Initialize models
    print("\n1. Initializing AI models...")
    chat_model = OpenAIChatModel(api_key=api_key, model_name="gpt-3.5-turbo")
    embedding_model = OpenAIEmbeddingModel(api_key=api_key)
    print("   ✓ Models initialized")

    # Step 4: Initialize storage
    print("\n2. Initializing storage...")
    storage = SQLiteEngine(db_path="basic_example.db", embedding_dim=1536)
    print("   ✓ Storage initialized")

    # Step 5: Create memory manager
    print("\n3. Creating MemoryManager...")
    manager = MemoryManager(
        chat_model=chat_model,
        embedding_model=embedding_model,
        storage_engine=storage,
        config=config
    )
    print("   ✓ MemoryManager ready")

    # Step 6: Process some queries
    print("\n4. Processing queries with memory...")
    print("-" * 60)

    queries = [
        "Hello! My name is Alex and I love programming.",
        "What's my name?",
        "What do I love doing?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        response = manager.process_input(query)
        print(f"Response: {response}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)

    # Cleanup
    import os
    if os.path.exists("basic_example.db"):
        os.remove("basic_example.db")
        print("\nCleaned up database file.")

if __name__ == "__main__":
    main()
