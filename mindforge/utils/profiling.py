
import cProfile
import pstats
import io
from typing import Callable, Any
from functools import wraps


def profile(output_file: str = None):
    """Decorator for profiling functions."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            pr = cProfile.Profile()
            pr.enable()

            result = func(*args, **kwargs)

            pr.disable()

            if output_file:
                with open(output_file, "w") as f:
                    stats = pstats.Stats(pr, stream=f)
                    stats.sort_stats("cumulative")
                    stats.print_stats()
            else:  # Print to stdout if no output file
                s = io.StringIO()
                stats = pstats.Stats(pr, stream=s)
                stats.sort_stats("cumulative")
                stats.print_stats()
                print(s.getvalue())

            return result

        return wrapper

    return decorator
