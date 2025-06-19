
import sys
import os
from mindforge.utils.logging import LogManager
from mindforge.utils.errors import ConfigurationError, ModelError
from mindforge import MemoryManager
from mindforge.models.chat import (
    OpenAIChatModel,
    AzureChatModel,
    OllamaChatModel,
    LiteLLMChatModel,
)
from mindforge.models.embedding import (
    OpenAIEmbeddingModel,
    AzureEmbeddingModel,
    OllamaEmbeddingModel,
    LiteLLMEmbeddingModel,
)
from mindforge.storage.sqlite_engine import SQLiteEngine # Import SQLiteEngine
from mindforge.config import AppConfig  # Import AppConfig
from typing import Optional


def initialize_models(config: AppConfig):
    """Initialize chat and embedding models based on configuration."""
    try:
        if config.model.use_model == "openai":
            if not config.model.chat_api_key or not config.model.embedding_api_key:
                raise ConfigurationError("API keys for OpenAI are missing.")
            chat_model = OpenAIChatModel(
                api_key=config.model.chat_api_key, model_name=config.model.chat_model_name
            )
            embedding_model = OpenAIEmbeddingModel(
                api_key=config.model.embedding_api_key,
                model_name=config.model.embedding_model_name,
            )

        elif config.model.use_model == "azure":
            if (
                not config.model.chat_api_key
                or not config.model.embedding_api_key
                or not config.model.azure_endpoint
                or not config.model.azure_api_version
            ):
                raise ConfigurationError("Azure configuration parameters are missing.")

            chat_model = AzureChatModel(
                api_key=config.model.chat_api_key,
                endpoint=config.model.azure_endpoint,
                deployment_name=config.model.chat_model_name,  # Assuming deployment name
                api_version=config.model.azure_api_version,
            )
            embedding_model = AzureEmbeddingModel(
                api_key=config.model.embedding_api_key,
                endpoint=config.model.azure_endpoint,
                deployment_name=config.model.embedding_model_name,  # Assuming deployment name
                api_version=config.model.azure_api_version,
            )
        elif config.model.use_model == "ollama":
            chat_model = OllamaChatModel(
                model_name=config.model.chat_model_name,
                base_url=config.model.ollama_base_url,
            )
            embedding_model = OllamaEmbeddingModel(
                model_name=config.model.embedding_model_name,
                base_url=config.model.ollama_base_url,
            )
        elif config.model.use_model == "litellm":
            chat_model = LiteLLMChatModel(
                model_name=config.model.chat_model_name,
                api_key=config.model.chat_api_key,
                base_url=config.model.litellm_base_url,
            )
            embedding_model = LiteLLMEmbeddingModel(
                model_name=config.model.embedding_model_name,
                api_key=config.model.embedding_api_key,
                base_url=config.model.litellm_base_url,
                dimension=config.vector.embedding_dim,
            )
        else:
            raise ConfigurationError(f"Unsupported model provider: {config.model.use_model}")


        return chat_model, embedding_model
    except Exception as e:
        raise ConfigurationError(f"Failed to initialize models: {str(e)}")


def process_query(
    manager: MemoryManager,
    query: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """Process a single query with error handling."""
    try:
        response = manager.process_input(query=query, user_id=user_id, session_id=session_id)
        return response
    except Exception as e:
        raise ModelError(f"Failed to process query: {str(e)}")


def main():
    # Initialize logging
    config = AppConfig()  # Load configuration
    log_manager = LogManager(log_dir=config.log_dir, log_level=config.log_level)
    logger = log_manager.get_logger("mindforge.main")

    try:
        logger.info("Starting MindForge")

        #  Handle API key for OpenAI directly from environment variable
        if config.model.use_model == "openai":
             api_key = os.getenv("OPENAI_API_KEY")
             if not api_key:
                 logger.error("OpenAI API key not found in environment variables")
                 sys.exit(1)
             config.model.chat_api_key = api_key  # Set in config for consistency
             config.model.embedding_api_key = api_key

        # Initialize components with proper error handling
        try:
            # Initialize models
            chat_model, embedding_model = initialize_models(config)
            logger.info("AI models initialized successfully")

            # Validate embedding dimension
            if embedding_model.dimension != config.vector.embedding_dim:
                error_msg = (
                    f"Embedding dimension mismatch: Model dimension is {embedding_model.dimension}, "
                    f"config.vector.embedding_dim is {config.vector.embedding_dim}. "
                    "Please ensure the configuration matches the embedding model."
                )
                logger.error(error_msg)
                raise ConfigurationError(error_msg)
            logger.info(f"Embedding dimension validated: {embedding_model.dimension}")

            # Initialize storage engine
            storage_engine = SQLiteEngine(
                db_path=config.storage.db_path,
                embedding_dim=embedding_model.dimension # Uses the validated dimension
            )
            logger.info("Storage engine initialized successfully")

            # Initialize memory manager
            manager = MemoryManager(
                chat_model=chat_model,
                embedding_model=embedding_model,
                storage_engine=storage_engine,
                config=config
            )
            logger.info("Memory manager initialized successfully")

        except ConfigurationError as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            sys.exit(1)

        # Process example queries with proper logging and error handling
        example_queries = [
            {
                "query": "What is machine learning?",
                "user_id": "user123",
                "session_id": "session456",
            },
            {
                "query": "How does deep learning work?",
                "user_id": "user123",
                "session_id": "session456",
            },
        ]

        for query_data in example_queries:
            try:
                logger.info(f"Processing query: {query_data['query']}")

                response = process_query(
                    manager,
                    query=query_data["query"],
                    user_id=query_data["user_id"],
                    session_id=query_data["session_id"],
                )

                logger.info("Query processed successfully")
                print(f"\nQuery: {query_data['query']}")
                print(f"Response: {response}\n")

            except ModelError as e:
                logger.error(f"Error processing query: {str(e)}")
                continue
            except Exception as e:
                logger.exception("Unexpected error occurred while processing query")
                continue

        logger.info("MindForge execution completed successfully")

    except Exception as e:
        logger.exception("Fatal error occurred")
        sys.exit(1)
    finally:
        # Cleanup code here if needed
        logger.info("MindForge shutting down")


if __name__ == "__main__":
    main()