
import logging


class MindForgeError(Exception):
    """Base exception class for MindForge."""

    pass


class ConfigurationError(MindForgeError):
    """Raised when there's a configuration error."""

    pass


class ModelError(MindForgeError):
    """Raised when there's an error with AI models."""

    pass


class StorageError(MindForgeError):
    """Raised when there's a storage-related error."""

    pass


class ValidationError(MindForgeError):
    """Raised when validation fails."""

    pass


class MemoryError(MindForgeError):
    """Raised when there's a memory-related error."""

    pass


def handle_exceptions(logger: logging.Logger):
    """Decorator for exception handling."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except MindForgeError as e:
                logger.error(f"MindForge error in {func.__name__}: {str(e)}")
                raise
            except Exception as e:
                logger.exception(f"Unexpected error in {func.__name__}: {str(e)}")
                raise MindForgeError(f"Unexpected error: {str(e)}")

        return wrapper

    return decorator
