
import time
import logging
from functools import wraps
from typing import Any, Callable, Dict


class PerformanceMonitor:
    """Monitor system performance."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics = {}

    def timer(self, operation: str) -> Callable:
        """Decorator to time operations."""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                start = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start

                if operation not in self.metrics:
                    self.metrics[operation] = []
                self.metrics[operation].append(duration)

                self.logger.debug(f"{operation} took {duration:.4f} seconds")
                return result

            return wrapper

        return decorator

    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation."""
        if operation not in self.metrics:
            return {}

        times = self.metrics[operation]
        return {
            "count": len(times),
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
        }