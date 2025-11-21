import sys
import os

# Add the parent directory to sys.path to allow imports from mindforge
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mindforge.core.memory_manager import MemoryManager
from mindforge.core.memory_types import MemoryType
from mindforge.models.chat import OllamaChatModel
from mindforge.models.embedding import OllamaEmbeddingModel
from mindforge.storage.sqlite_engine import SQLiteEngine
from mindforge.config import AppConfig

def main():
    # Initialize components
    # Note: This example assumes Ollama is running locally.
    # If not, you might need to mock the models or use OpenAI ones with API keys.
    # For the purpose of this example, we'll show how to instantiate and use the memory types.

    config = AppConfig()

    # Use dummy models if real ones are not available/configured,
    # but for the example structure we will instantiate standard ones.
    # Ideally, you'd replace these with valid model instances.
    chat_model = OllamaChatModel(model_name="llama3")
    embedding_model = OllamaEmbeddingModel(model_name="nomic-embed-text")

    # Use SQLite storage for simplicity
    storage_engine = SQLiteEngine(config=config)

    memory_manager = MemoryManager(
        chat_model=chat_model,
        embedding_model=embedding_model,
        storage_engine=storage_engine,
        config=config
    )

    print("MindForge Multi-Level Memory Example")
    print("====================================")

    scenarios = [
        (MemoryType.SHORT_TERM, "What is the weather today?", "Checking weather..."),
        (MemoryType.LONG_TERM, "The capital of France is Paris.", "Fact stored."),
        (MemoryType.USER_SPECIFIC, "My favorite color is blue.", "User preference updated."),
        (MemoryType.SESSION_SPECIFIC, "I am currently debugging the login page.", "Context set."),
        (MemoryType.AGENT_SPECIFIC, "I am a coding assistant.", "Role defined."),
        (MemoryType.PERSONA, "We are good friends.", "Relationship deepened."),
        (MemoryType.TOOLBOX, '{"name": "calculator", "schema": "..."}', "Tool registered."),
        (MemoryType.CONVERSATION, "Hello, how are you?", "Greeting received."),
        (MemoryType.WORKFLOW, "Build process failed.", "Workflow outcome recorded."),
        (MemoryType.EPISODIC, "The user solved the difficult bug on Friday.", "Episode logged."),
        (MemoryType.AGENT_REGISTRY, "Agent 'Coder' is available.", "Registry updated."),
        (MemoryType.ENTITY, "Elon Musk is the CEO of Tesla.", "Entity info stored."),
    ]

    for mem_type, query, response in scenarios:
        print(f"\nProcessing {mem_type.value} memory...")
        try:
            # We mock the embedding and chat response generation for the sake of the example
            # if the models fail (e.g. Ollama not running).
            # In a real scenario, process_input would call the models.

            # Here we are just demonstrating the API call structure
            # To actually run this without a running LLM, we'd need to mock the models.
            # But this file serves as a usage example.

            print(f"  Query: {query}")
            print(f"  Type: {mem_type.name}")

            # Note: This will fail if Ollama is not running or configured.
            # Ideally, we would catch exceptions or mock the manager's internal calls
            # if we wanted this script to be runnable without deps.
            # However, an example should show real usage.

            # Let's assume the user might run this.
            # We will call process_input.
            result = memory_manager.process_input(
                query=query,
                memory_type=mem_type
            )
            print(f"  Result: {result}")

        except Exception as e:
            print(f"  (Simulation): stored interaction for {mem_type.value} (Error calling model: {e})")
            # Fallback: Manually store to demonstrate the memory type works in storage
            # Mocking what process_input does internally
            embedding = [0.1] * 1536 # Dummy embedding
            concepts = ["example"]
            memory_manager._store_interaction(
                query=query,
                response=response,
                embedding=embedding,
                concepts=concepts,
                memory_type=mem_type.value
            )
            print("  (Fallback): Manually stored interaction successfully.")

    print("\nAll memory types processed.")

if __name__ == "__main__":
    main()
