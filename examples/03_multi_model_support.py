"""
Multi-Model Support Example for MindForge
=========================================
This example demonstrates how to use MindForge with different LLM providers:
- OpenAI
- Azure OpenAI
- Ollama (local models)
- LiteLLM (any provider)
"""

import os
from mindforge import MemoryManager
from mindforge.storage.sqlite_engine import SQLiteEngine
from mindforge.config import AppConfig

def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def example_openai():
    """Example using OpenAI models"""
    print_section("OpenAI Example")

    from mindforge.models.chat import OpenAIChatModel
    from mindforge.models.embedding import OpenAIEmbeddingModel

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set. Skipping OpenAI example.")
        return

    print("✓ Using OpenAI models")

    config = AppConfig()
    chat_model = OpenAIChatModel(api_key=api_key, model_name="gpt-3.5-turbo")
    embedding_model = OpenAIEmbeddingModel(api_key=api_key)
    storage = SQLiteEngine(db_path="openai_example.db", embedding_dim=1536)

    manager = MemoryManager(
        chat_model=chat_model,
        embedding_model=embedding_model,
        storage_engine=storage,
        config=config
    )

    response = manager.process_input("Hello! This is a test with OpenAI.")
    print(f"\nQuery: Hello! This is a test with OpenAI.")
    print(f"Response: {response}")

    # Cleanup
    import os as os_cleanup
    if os_cleanup.path.exists("openai_example.db"):
        os_cleanup.remove("openai_example.db")

def example_azure():
    """Example using Azure OpenAI models"""
    print_section("Azure OpenAI Example")

    from mindforge.models.chat import AzureChatModel
    from mindforge.models.embedding import AzureEmbeddingModel

    api_key = os.getenv("AZURE_OPENAI_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    if not api_key or not endpoint:
        print("❌ AZURE_OPENAI_KEY or AZURE_OPENAI_ENDPOINT not set.")
        print("   Skipping Azure example.")
        return

    print("✓ Using Azure OpenAI models")

    config = AppConfig()
    chat_model = AzureChatModel(
        api_key=api_key,
        endpoint=endpoint,
        deployment_name="gpt-35-turbo",  # Your deployment name
        api_version="2024-02-15-preview"
    )
    embedding_model = AzureEmbeddingModel(
        api_key=api_key,
        endpoint=endpoint,
        deployment_name="text-embedding-ada-002",  # Your deployment name
        api_version="2024-02-15-preview"
    )
    storage = SQLiteEngine(db_path="azure_example.db", embedding_dim=1536)

    manager = MemoryManager(
        chat_model=chat_model,
        embedding_model=embedding_model,
        storage_engine=storage,
        config=config
    )

    response = manager.process_input("Hello! This is a test with Azure OpenAI.")
    print(f"\nQuery: Hello! This is a test with Azure OpenAI.")
    print(f"Response: {response}")

    # Cleanup
    import os as os_cleanup
    if os_cleanup.path.exists("azure_example.db"):
        os_cleanup.remove("azure_example.db")

def example_ollama():
    """Example using Ollama for local models"""
    print_section("Ollama Example (Local Models)")

    from mindforge.models.chat import OllamaChatModel
    from mindforge.models.embedding import OllamaEmbeddingModel

    print("ℹ️  Make sure Ollama is running locally: ollama serve")
    print("   And you have a model installed: ollama pull llama2")

    try:
        config = AppConfig()
        chat_model = OllamaChatModel(
            model_name="llama2",
            base_url="http://localhost:11434"
        )
        embedding_model = OllamaEmbeddingModel(
            model_name="llama2",
            base_url="http://localhost:11434"
        )
        storage = SQLiteEngine(db_path="ollama_example.db", embedding_dim=4096)

        manager = MemoryManager(
            chat_model=chat_model,
            embedding_model=embedding_model,
            storage_engine=storage,
            config=config
        )

        response = manager.process_input("Hello! This is a test with Ollama.")
        print(f"\nQuery: Hello! This is a test with Ollama.")
        print(f"Response: {response}")

        # Cleanup
        import os as os_cleanup
        if os_cleanup.path.exists("ollama_example.db"):
            os_cleanup.remove("ollama_example.db")

    except Exception as e:
        print(f"❌ Ollama not available: {e}")
        print("   Skipping Ollama example.")

def example_litellm():
    """Example using LiteLLM for any provider"""
    print_section("LiteLLM Example (Universal Provider)")

    from mindforge.models.chat import LiteLLMChatModel
    from mindforge.models.embedding import LiteLLMEmbeddingModel

    api_key = os.getenv("OPENAI_API_KEY")  # Can be any provider's key
    if not api_key:
        print("❌ API key not set. Skipping LiteLLM example.")
        return

    print("✓ Using LiteLLM with OpenAI backend")

    config = AppConfig()
    chat_model = LiteLLMChatModel(
        model_name="gpt-3.5-turbo",
        api_key=api_key
    )
    embedding_model = LiteLLMEmbeddingModel(
        model_name="text-embedding-3-small",
        api_key=api_key,
        dimension=1536
    )
    storage = SQLiteEngine(db_path="litellm_example.db", embedding_dim=1536)

    manager = MemoryManager(
        chat_model=chat_model,
        embedding_model=embedding_model,
        storage_engine=storage,
        config=config
    )

    response = manager.process_input("Hello! This is a test with LiteLLM.")
    print(f"\nQuery: Hello! This is a test with LiteLLM.")
    print(f"Response: {response}")

    # Cleanup
    import os as os_cleanup
    if os_cleanup.path.exists("litellm_example.db"):
        os_cleanup.remove("litellm_example.db")

def main():
    print_section("MindForge - Multi-Model Support Demo")

    print("\nThis example demonstrates MindForge's support for multiple LLM providers.")
    print("Each provider example will run if the required environment variables are set.\n")

    # Run examples
    example_openai()
    example_azure()
    example_ollama()
    example_litellm()

    print_section("All Examples Complete!")

if __name__ == "__main__":
    main()
