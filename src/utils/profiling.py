"""Performance profiling utilities."""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


def profile(name: str | None = None, level: int = logging.INFO):
    """
    Decorator to log execution time of functions.

    Args:
        name: Optional custom name for the profiled function. If None, uses module.function_name
        level: Logging level (default: INFO)

    Example:
        @profile("judge_scoring")
        def score_batch(prompts, responses):
            ...

        # Logs: "⏱️  judge_scoring took 2.34s"
    """

    def decorator(func: Callable) -> Callable:
        nonlocal name
        if name is None:
            name = f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start
                logger.log(level, f"⏱️  {name} took {elapsed:.2f}s")

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start
                logger.log(level, f"⏱️  {name} took {elapsed:.2f}s")

        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# Import asyncio only when needed to avoid circular import issues
import asyncio
