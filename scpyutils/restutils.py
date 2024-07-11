"""
Utilities for REST API calls.

Author: Sam Cohan
"""

from typing import Any

import requests
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from scpyutils.logutils import setup_logger

LOGGER = setup_logger(__name__)


def should_retry(exception: BaseException) -> bool:
    """Custom function to signal retry on all exceptions except 400 errors.

    Args:
        exception: The exception raised by the function.

    Returns:
        Boolean of whether to retry the function.
    """
    if not isinstance(exception, requests.exceptions.RequestException):
        return False
    if isinstance(exception, requests.exceptions.HTTPError):
        # Fail immediately for 400 errors
        if 400 <= exception.response.status_code < 500:
            LOGGER.error(f"Immediate failure on 400 error: {exception}")
            return False
    LOGGER.warning("Retrying on error:", exception)
    return True


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception(should_retry),
)
def requests_get(*args: Any, **kwargs: Any) -> dict:
    """requests.get() that retries on all exceptions except 400 errors."""
    response = requests.get(*args, **kwargs)
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response.json()
