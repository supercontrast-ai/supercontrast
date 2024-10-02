import logging
import time

from functools import wraps
from typing import Any, Callable


def track_latency(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        latency = end_time - start_time
        logging.info(
            f"Function '{func.__name__}' took {latency:.4f} seconds to execute."
        )
        return result

    return wrapper
