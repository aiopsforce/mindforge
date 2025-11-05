"""
Memory Types Example for MindForge
===================================
This example demonstrates different memory types:
- Short-term memory (recent context)
- Long-term memory (persistent knowledge)
- User-specific memory (personalization)
- Session-specific memory (conversation context)
- Agent-specific memory (agent knowledge)
"""

import os
from mindforge import MemoryManager
from mindforge.models.chat import OpenAIChatModel
from mindforge.models.embedding import OpenAIEmbeddingModel
from mindforge.storage.sqlite_engine import SQLiteEngine
from mindforge.config import AppConfig

def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def main():
    print_section("MindForge - Memory Types Demo")

    # Setup
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nError: OPENAI_API_KEY not set!")
        return

    config = AppConfig()
    chat_model = OpenAIChatModel(api_key=api_key, model_name="gpt-3.5-turbo")
    embedding_model = OpenAIEmbeddingModel(api_key=api_key)
    storage = SQLiteEngine(db_path="memory_types.db", embedding_dim=1536)

    manager = MemoryManager(
        chat_model=chat_model,
        embedding_model=embedding_model,
        storage_engine=storage,
        config=config
    )

    # Demo 1: Short-term Memory (default)
    print_section("1. Short-term Memory")
    print("Short-term memory stores recent interactions temporarily.\n")

    response1 = manager.process_input(
        "I just ate a delicious pizza for lunch.",
        memory_type="short_term"
    )
    print(f"Input: I just ate a delicious pizza for lunch.")
    print(f"Response: {response1}\n")

    response2 = manager.process_input(
        "What did I eat for lunch?",
        memory_type="short_term"
    )
    print(f"Input: What did I eat for lunch?")
    print(f"Response: {response2}")

    # Demo 2: Long-term Memory
    print_section("2. Long-term Memory")
    print("Long-term memory persists important information.\n")

    response1 = manager.process_input(
        "Remember that the project deadline is December 31st, 2025.",
        memory_type="long_term"
    )
    print(f"Input: Remember that the project deadline is December 31st, 2025.")
    print(f"Response: {response1}\n")

    response2 = manager.process_input(
        "When is the project deadline?",
        memory_type="long_term"
    )
    print(f"Input: When is the project deadline?")
    print(f"Response: {response2}")

    # Demo 3: User-specific Memory
    print_section("3. User-specific Memory")
    print("User memory enables personalization for different users.\n")

    # User 1
    response1 = manager.process_input(
        "My favorite color is blue.",
        user_id="user_alice",
        memory_type="user"
    )
    print(f"[Alice] Input: My favorite color is blue.")
    print(f"[Alice] Response: {response1}\n")

    # User 2
    response2 = manager.process_input(
        "My favorite color is red.",
        user_id="user_bob",
        memory_type="user"
    )
    print(f"[Bob] Input: My favorite color is red.")
    print(f"[Bob] Response: {response2}\n")

    # Retrieve for each user
    response3 = manager.process_input(
        "What's my favorite color?",
        user_id="user_alice",
        memory_type="user"
    )
    print(f"[Alice] Input: What's my favorite color?")
    print(f"[Alice] Response: {response3}\n")

    response4 = manager.process_input(
        "What's my favorite color?",
        user_id="user_bob",
        memory_type="user"
    )
    print(f"[Bob] Input: What's my favorite color?")
    print(f"[Bob] Response: {response4}")

    # Demo 4: Session-specific Memory
    print_section("4. Session-specific Memory")
    print("Session memory maintains context within a conversation.\n")

    # Session 1
    response1 = manager.process_input(
        "Let's discuss Python programming.",
        session_id="session_001",
        memory_type="session"
    )
    print(f"[Session 001] Input: Let's discuss Python programming.")
    print(f"[Session 001] Response: {response1}\n")

    response2 = manager.process_input(
        "What are we discussing?",
        session_id="session_001",
        memory_type="session"
    )
    print(f"[Session 001] Input: What are we discussing?")
    print(f"[Session 001] Response: {response2}")

    # Demo 5: Agent-specific Memory
    print_section("5. Agent-specific Memory")
    print("Agent memory stores knowledge about the AI agent itself.\n")

    response1 = manager.process_input(
        "You are an expert in machine learning and data science.",
        memory_type="agent"
    )
    print(f"Input: You are an expert in machine learning and data science.")
    print(f"Response: {response1}\n")

    response2 = manager.process_input(
        "What are you an expert in?",
        memory_type="agent"
    )
    print(f"Input: What are you an expert in?")
    print(f"Response: {response2}")

    print_section("Demo Complete!")

    # Cleanup
    import os
    if os.path.exists("memory_types.db"):
        os.remove("memory_types.db")
        print("\nCleaned up database file.")

if __name__ == "__main__":
    main()
