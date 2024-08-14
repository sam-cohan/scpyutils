"""
Miscellaneous Utilities.

Author: Sam Cohan
"""

import difflib
import importlib
import re
import time
from functools import wraps
from itertools import zip_longest
from typing import Any, Callable, Dict, Iterable, Literal, Optional, Tuple, Union

from scpyutils.cacheutils import get_func_name
from scpyutils.logutils import setup_logger

LOGGER = setup_logger(__name__)


def diff_str(  # noqa: C901
    text1: str,
    text2: str,
    diff_level: Literal["char", "word"] = "char",
    html_output: bool = False,
    mark_spaces: bool = False,
) -> str:
    """
    Generate a diff between two strings at the specified diff level.

    Args:
        text1: The first string to compare.
        text2: The second string to compare.
        diff_level: The level of comparison ('char' or 'word'). Default is 'char'.
        html_output: If True, returns the diff in HTML format. Default is False.
        mark_spaces: Whether to mark spaces with '␣' for emphasis. Default is False.

    Returns:
        A string representing the differences between text1 and text2.
    """
    space_placeholder = "␣" if mark_spaces else " "
    if diff_level == "word":
        text1_seq = re.findall(r"\S+|\s+", text1)
        text2_seq = re.findall(r"\S+|\s+", text2)
    else:  # 'char' level diff
        text1_seq = list(text1)
        text2_seq = list(text2)

    matcher = difflib.SequenceMatcher(None, text1_seq, text2_seq)
    diff_output = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            text1_part = "".join(text1_seq[i1:i2]).replace(" ", space_placeholder)
            text2_part = "".join(text2_seq[j1:j2]).replace(" ", space_placeholder)

            # Adjust leading space
            if i1 > 0 and text1_seq[i1 - 1].isspace():
                diff_output.append(" ")

            diff_output.append(
                (
                    f"<span style='background-color:#ffdddd;'>[-{text1_part}]</span>"
                    f"<span style='background-color:#ddffdd;'>[+{text2_part}]</span>"
                )
                if html_output
                else f"[-{text1_part}][+{text2_part}]"
            )

            # Adjust trailing space
            if i2 < len(text1_seq) and text1_seq[i2 : i2 + 1] == [" "]:
                diff_output.append(" ")
        elif tag == "delete":
            if i1 > 0 and text1_seq[i1 - 1].isspace():
                diff_output.append(" ")
            diff_output.append(
                f"<span style='background-color:#ffdddd;'>"
                f"[-{''.join(text1_seq[i1:i2]).replace(' ', space_placeholder)}]</span>"
                if html_output
                else f"[-{''.join(text1_seq[i1:i2]).replace(' ', space_placeholder)}]"
            )
        elif tag == "insert":
            if j1 > 0 and text2_seq[j1 - 1].isspace():
                diff_output.append(" ")
            diff_output.append(
                f"<span style='background-color:#ddffdd;'>"
                f"[+{''.join(text2_seq[j1:j2]).replace(' ', space_placeholder)}]</span>"
                if html_output
                else f"[+{''.join(text2_seq[j1:j2]).replace(' ', space_placeholder)}]"
            )
        elif tag == "equal":
            diff_output.append("".join(text1_seq[i1:i2]))

    diff_output_str = "".join(diff_output)
    if html_output:
        diff_output_str = diff_output_str.replace("\n", "<br>")

    return diff_output_str


def display_diff_str(
    text1: str,
    text2: str,
    diff_level: Literal["char", "word"] = "char",
    mark_spaces: bool = False,
) -> None:
    """
    Display a diff between two strings using HTML.

    Args:
        text1: The first string to compare.
        text2: The second string to compare.
        diff_level: The level of comparison ('char' or 'word'). Default is 'char'.
        mark_spaces: Whether to mark spaces with '␣' for emphasis. Default is False.
    """
    from IPython.display import HTML, display  # type: ignore

    diff_html = diff_str(
        text1,
        text2,
        diff_level=diff_level,
        html_output=True,
        mark_spaces=mark_spaces,
    )
    display(HTML(diff_html))


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
