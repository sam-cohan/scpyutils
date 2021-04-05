"""
Utilities for multi-processing and multi-threading.

Author: Sam Cohan
"""
import multiprocessing as mpc
from typing import Any, Callable, Dict, List

from joblib import Parallel, delayed


def parallel_map(
    func: Callable[[Any], Any],
    kwargs_list: List[Dict[str, Any]],
    n_jobs: int = -1,
    backend: str = "loky",
) -> List[Any]:
    """Apply a function to a list of inputs in a pool of jobs.

    Args:
        func: Function which accepts keyword arguments.
        kwargs_list: List of kwargs used to call `func`.
        n_jobs: Pool size.
        backend: `backend` used by joblib (default: 'loky')
            Pick from {'threading', 'multiprocessing', 'loky'}

    Returns:
        List of results from calling func with each input.
    """
    if n_jobs == -1:
        n_jobs = mpc.cpu_count()
        print(f"parallel_map will use n_jobs={n_jobs}")
    results = []
    with Parallel(backend=backend, n_jobs=n_jobs) as parallel:
        for i in range(0, len(kwargs_list), n_jobs):
            results.extend(
                parallel(
                    delayed(func)(**kwargs) for kwargs in kwargs_list[i : i + n_jobs]
                )
            )
    return results
