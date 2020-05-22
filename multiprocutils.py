import multiprocessing.context as mpc_ctx
from multiprocessing.pool import ThreadPool

from .logutils import setup_logger

LOGGER = setup_logger(__name__)


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
                "TimeoutError trying to run %s(*%s, **%s)... %s", self._func, args, kwargs, e)
            raise e
        return _res
