import pdb
import sys
import time
from typing import Any, Callable


from .cacheutils import get_func_name
from .logutils import setup_logger

LOGGER = setup_logger(__name__)


def time_me(func, *args, **kwargs):
    func_name = get_func_name(func)
    logger = kwargs.pop("___logger", kwargs.get("__logger", LOGGER))
    getattr(logger, "debug", logger)(f"calling {func_name}...")
    t1 = time.time()
    res = func(*args, **kwargs)
    t2 = time.time()
    getattr(logger, "debug", logger)(f"{func_name} took {t2-t1:,.6f} seconds.")
    return res


def print_len(x, logger=LOGGER):
    getattr(logger, "debug", logger)("length is {0:,.0f}.".format(len(x)))
    return x


def patch_instance(inst: Any, func_name: str, new_func: Callable):
    """Allows to patch instance member functions with arbitrary ones.

    Arguments:
        inst (Object): instance you want to monkey-patch
        func_name (str): name of the function on the instance you want to
            monkey-patch
        new_func (Callable): new function to replace the old function
    """
    import types

    try:
        # in python 2.7 this was .im_self
        self = getattr(inst, func_name).__self__
    except AttributeError:
        print("WARNING: %s does not exist... will add it!" % func_name)
        # try to get the self from any arbitrary method on the instance
        self = [
            getattr(inst, x).__self__
            for x in dir(inst)
            if isinstance(getattr(inst, x), types.MethodType)
        ][0]
    inst.__dict__[func_name] = types.MethodType(
        new_func, self
    )  # in python 2.7 this needed inst.__class__
