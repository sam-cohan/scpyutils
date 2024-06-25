"""
Utilities for multiprocessing.

Author: Sam Cohan
"""

import multiprocessing as mpc
import multiprocessing.context as mpc_ctx
from multiprocessing.pool import ThreadPool
from typing import Any, Callable, Dict, List

from joblib import Parallel, delayed

from scpyutils.logutils import setup_logger

LOGGER = setup_logger(__name__)


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


class RunInThreadWithTimeout:
    def __init__(self, timeout=2):
        self._timeout = timeout
        self._func = None

    def __call__(self, func):
        self._func = func
        return self.__threaded_call__

    def __threaded_call__(self, *args, **kwargs):
        _pool = ThreadPool(processes=1)

        def _target():
            return self._func(*args, **kwargs)

        try:
            _res = _pool.apply_async(_target).get(timeout=self._timeout)
        except mpc_ctx.TimeoutError as e:
            LOGGER.error(
                "TimeoutError trying to run %s(*%s, **%s)... %s",
                self._func,
                args,
                kwargs,
                e,
            )
            raise e
        return _res
