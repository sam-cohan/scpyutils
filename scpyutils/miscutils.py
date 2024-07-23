"""
Miscellaneous Utilities.

Author: Sam Cohan
"""

import importlib
import time
from functools import wraps
from itertools import zip_longest
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

from scpyutils.cacheutils import get_func_name
from scpyutils.logutils import setup_logger

LOGGER = setup_logger(__name__)


def diff_rec(  # noqa: C901
    left: Union[Dict, Iterable],
    right: Union[Dict, Iterable],
    ignore_keys: Optional[set] = None,
    comp_mode: str = "dp",
    precision: int = 6,
) -> Union[Dict[str, Any], Dict[int, Any], Tuple[Any, Any], Tuple[Any, Any, Any]]:
    """
    Returns diff of two dictionaries (or non-string iterables).
    It uses recursive drill down, so do not use on structures which have big depth!

    Args:
        left: Left dict or non-string iterable.
        right: Right dict or non-string iterable.
        ignore_keys: Optional set of keys to be ignored during comparison.
        comp_mode: Numerical comparison modes. Must be one of
            - sf : significant figures
            - dp : decimal places
            - pc : percentage change
        precision: Precision depending on the mode:
            if comp_mode='sf' then significant figures to round to before comparison
            if comp_mode='dp' then decimal places to round to before comparison
            if comp_mode='pc' then percentage change required to trigger diff.

    Returns:
        Diff of left and right. If left and right were dictionaries that are not
        identical, then one or more of the following keys will exist:
            - "ValDiff": if there are keys which have value differences, then this
                will contain a dictionary of those keys with values being tuple of
                (left_value, right_value, diff_value) if numerical or simply
                (left_value, right_value) if not numerical.
            - "OnlyInLeft": if there are some keys that only exist in the left, then
                those keys and their values will appear here.
            - "OnlyInRight": if there are some keys that only exist in the right, then
                those keys and their values will appear here.
    """
    if ignore_keys is None:
        ignore_keys = set()
    ignore_keys = set(ignore_keys)

    def diff_rec_(
        left: Any, right: Any
    ) -> Union[Dict[str, Any], Dict[int, Any], Tuple[Any, Any], Tuple[Any, Any, Any]]:
        diffs: Union[
            Dict[str, Any], Dict[int, Any], Tuple[Any, Any], Tuple[Any, Any, Any]
        ] = {}
        left_is_dict = isinstance(left, dict)
        right_is_dict = isinstance(right, dict)
        if left_is_dict and right_is_dict:
            left_keys = set(left)
            right_keys = set(right)
            shared_keys = left_keys & right_keys - ignore_keys
            only_in_left_keys = left_keys - right_keys - ignore_keys
            only_in_right_keys = right_keys - left_keys - ignore_keys
            map_diffs: Dict[str, Any] = {}
            for key in shared_keys:
                res = diff_rec_(left[key], right[key])
                if res:
                    if "ValDiff" not in map_diffs:
                        map_diffs["ValDiff"] = {}
                    map_diffs["ValDiff"][key] = res
            if only_in_left_keys:
                tmp_map = {k: left[k] for k in only_in_left_keys}
                map_diffs["OnlyInLeft"] = tmp_map
            if only_in_right_keys:
                tmp_map = {k: right[k] for k in only_in_right_keys}
                map_diffs["OnlyInRight"] = tmp_map
            if map_diffs:
                diffs = map_diffs
        elif not left_is_dict and not right_is_dict:
            left_is_iterable = hasattr(left, "__iter__") and not isinstance(left, str)
            right_is_iterable = hasattr(right, "__iter__") and not isinstance(
                right, str
            )
            if left_is_iterable and right_is_iterable:
                iterable_diffs: Dict[int, Any] = {}
                idx = -1
                for item1, item2 in zip_longest(left, right):
                    idx += 1
                    res = diff_rec_(item1, item2)
                    if res:
                        iterable_diffs[idx] = res
                if iterable_diffs:
                    diffs = iterable_diffs
            elif not left_is_iterable and not right_is_iterable:
                if left != right:
                    if comp_mode == "sf":
                        try:
                            left_num = ("%%.%dg" % (int(precision))) % (left)
                            right_num = ("%%.%dg" % (int(precision))) % (right)
                            if left_num != right_num:
                                diffs = (left, right, right - left)
                        except:  # noqa: E722
                            diffs = (left, right)
                    elif comp_mode == "dp":
                        try:
                            left_num = ("{0:.%sf}" % (int(precision))).format(
                                float(left)
                            )
                            right_num = ("{0:.%sf}" % (int(precision))).format(
                                float(right)
                            )
                            if left_num != right_num:
                                diffs = (left, right, right - left)
                        except:  # noqa: E722
                            diffs = (left, right)
                    elif comp_mode == "pc":
                        try:
                            if abs(right - left) / float(abs(left)) >= precision:
                                diffs = (left, right, right - left)
                        except:  # noqa: E722
                            diffs = (left, right)
                    else:
                        raise Exception("comp_mode='%s' not supported" % (comp_mode))
            else:
                diffs = (left, right)
        else:
            diffs = (left, right)
        return diffs

    return diff_rec_(left, right)


def patch_instance(inst: Any, func_name: str, new_func: Callable) -> None:
    """
    Allows to patch instance member functions with arbitrary ones.

    Args:
        inst: Instance you want to monkey-patch.
        func_name: Name of the function on the instance you want to monkey-patch.
        new_func: New function to replace the old function.

    Returns:
        None
    """
    import types

    try:
        # in python 2.7 this was .im_self
        self = getattr(inst, func_name).__self__
    except AttributeError:
        LOGGER.warning(f"WARNING: {func_name} does not exist... will add it!")
        # try to get the self from any arbitrary method on the instance
        self = [
            getattr(inst, x).__self__
            for x in dir(inst)
            if isinstance(getattr(inst, x), types.MethodType)
        ][0]
    inst.__dict__[func_name] = types.MethodType(
        new_func, self
    )  # in python 2.7 this needed inst.__class__


def reload_module(module_name: str) -> Any:
    """
    Reloads a specified module.

    Args:
        module_name: The name of the module to reload.

    Returns:
        The reloaded module.
    """
    module = importlib.import_module(module_name)
    importlib.reload(module)
    return module


def retry(
    max_tries: int = 3,
    wait_secs: int = 5,
    raise_on_fail: bool = True,
    res_on_fail: Any = None,
) -> Callable:
    """
    Decorator for allowing a function to retry.

    Args:
        max_tries: Maximum number of attempts to call the function. Defaults to 3.
        wait_secs: Time to wait between failures in seconds. Defaults to 5.
        raise_on_fail: Whether to raise if all attempts fail. Defaults to True.
        res_on_fail: Value to return in case of failure. Defaults to None. This only
            makes sense when raise_on_fail is set to False.

    Returns:
        The decorated function.
    """

    def retry_(func: Callable) -> Callable:
        @wraps(func)
        def retry_func(*args: Any, **kwargs: Any) -> Any:
            tries = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    tries += 1
                    LOGGER.exception(
                        f"ERROR: attempt={tries} failed in calling {func}: {e}"
                    )
                    if tries >= max_tries:
                        LOGGER.error(f"Max tries={max_tries} reached for {func}.")
                        if raise_on_fail:
                            raise e
                        else:
                            break
                    time.sleep(wait_secs)
                    LOGGER.warning(f"Retrying to call {func}")
            return res_on_fail

        return retry_func

    return retry_


def time_me(func: Callable, *args: Any, **kwargs: Any) -> Any:
    """
    Measures the execution time of a function.

    Args:
        func: The function to measure.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The result of the function execution.
    """
    func_name = get_func_name(func)
    logger = kwargs.pop("___logger", kwargs.get("__logger", LOGGER))
    getattr(logger, "debug", logger)(f"calling {func_name}...")
    start = time.time()
    res = func(*args, **kwargs)
    end = time.time()
    getattr(logger, "debug", logger)(f"{func_name} took {end-start:,.03f} seconds.")
    return res
