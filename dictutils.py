from itertools import product, zip_longest
from typing import Any, Dict, List


class PathMissingException(Exception):
    pass


def read_dict(obj, path, err=False):
    if path:
        try:
            return read_dict(obj[path[0]], path[1:])
        except:
            if err:
                raise Exception(
                    "Could not identify path %s in dictionary %s" % (path, obj)
                )
            return {}
    else:
        return obj


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


def update_paths(
    dct: Dict,
    paths: Dict,
    update_if_exists: bool = True,
    raise_if_missing: bool = False,
):
    _ = [
        update_path(
            dct=dct,
            path=path,
            val=val,
            update_if_exists=update_if_exists,
            raise_if_missing=raise_if_missing,
        )
        for path, val in paths.items()
    ]


def update_path(
    dct: Dict,
    path: str,
    val: Any,
    update_if_exists: bool = True,
    raise_if_missing: bool = False,
):
    path_parts = path.split("~")
    ensure_entry(
        dct=dct,
        path_parts=path_parts,
        val=val,
        update_if_exists=update_if_exists,
        raise_if_missing=raise_if_missing,
    )


def ensure_entry(
    dct: Dict,
    path_parts: List,
    val: Any = None,
    update_if_exists: bool = False,
    raise_if_missing: bool = False,
) -> Any:
    """Make sure an entry exists at the path without failing.

    Note that this is a recursive function does not update the entry if it
    already exists.

    Arguments:
        dct (Dict): possibly nested dictionary object to check
        path_parths (List): list of keys
        val (Any): value to be used only if the entry was missing.
            (defaults to None).
        update_if_exists (bool): whether we should update the value with
            `val` even if it exists (defaults to False)
        raise_if_missing (bool): whether an exception should be thrown
            if the path is missing. Note that if this is on, effectively
            the `val` parameter is not useful. (defaults to False).

    """
    fld = path_parts[0]
    next_path_parts = path_parts[1:]
    if next_path_parts:
        next_dct = dct.get(fld)
        if not isinstance(next_dct, dict):
            if raise_if_missing:
                raise PathMissingException()
            next_dct = {}
        dct[fld] = next_dct
        ensure_entry(
            dct=next_dct,
            path_parts=next_path_parts,
            val=val,
            update_if_exists=update_if_exists,
            raise_if_missing=raise_if_missing,
        )
    else:
        if fld not in dct:
            if raise_if_missing:
                raise PathMissingException()
            dct[fld] = val
        elif update_if_exists:
            dct[fld] = val


def deep_update(d1, d2):
    if isinstance(d1, dict) and isinstance(d2, dict):
        union = set(d1) | set(d2)
        return {k: find_deep_val(k, d1, d2) for k in union}
    else:
        return d2


def find_deep_val(k, d1, d2):
    if k in d2.keys():
        return deep_update(d1.get(k, None), d2[k])
    else:
        return d1[k]


def soft_update(d1, d2, raise_error=False):
    if raise_error:
        undesired_keys = set(d2) - set(d1)
        if undesired_keys:
            raise Exception(
                "The following keys should not be present: %s" % undesired_keys
            )
    return {k: d2.get(k, d1[k]) for k in d1}


def diff_rec(left, right, ignore_keys=None, comp_mode="dp", precision=6):
    """
    Given two generic structures left and right, this will return
    a map showing their diff. It uses recursive drill down, so do
    not use on structures which have big depth!

    ignore_keys:
        set of keys to be ignored during comparison

    Global variables used by diff_rec:

    comp_mode:
        Numerical comparison modes. Must be one of
            - sf : significant figures
            - dp : decimal places
            - pc : percentage change
    precision:
        precision depending on the mode:
        if comp_mode='sf' then significant figures to round to before comparison
        if comp_mode='dp' then decimal places to round to before comparison
        if comp_mode='pc' then percentage change required to trigger diff
    """
    if ignore_keys is None:
        ignore_keys = {}
    ignore_keys = set(ignore_keys)

    def diff_rec_(left, right):
        diffs = {}
        left_is_dict = isinstance(left, dict)
        right_is_dict = isinstance(right, dict)
        if left_is_dict and right_is_dict:
            left_keys = set(left)
            right_keys = set(right)
            shared_keys = left_keys & right_keys - ignore_keys
            only_in_left_keys = left_keys - right_keys - ignore_keys
            only_in_right_keys = right_keys - left_keys - ignore_keys
            map_diffs = {}
            for key in shared_keys:
                res = diff_rec_(left[key], right[key])
                if res:
                    if "ValDiff" not in map_diffs:
                        map_diffs["ValDiff"] = {}
                    map_diffs["ValDiff"][key] = res
            if only_in_left_keys:
                tmp_map = {}
                for k in only_in_left_keys:
                    tmp_map[k] = left[k]
                map_diffs["OnlyInLeft"] = tmp_map
            if only_in_right_keys:
                tmp_map = {}
                for k in only_in_right_keys:
                    tmp_map[k] = right[k]
                map_diffs["OnlyInRight"] = tmp_map
            if map_diffs:
                diffs = map_diffs
        elif not left_is_dict and not right_is_dict:
            left_is_iterable = hasattr(left, "__iter__") and not isinstance(left, str)
            right_is_iterable = hasattr(right, "__iter__") and not isinstance(
                right, str
            )
            if left_is_iterable and right_is_iterable:
                iterable_diffs = {}
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
                        except:
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
                        except:
                            diffs = (left, right)
                    elif comp_mode == "pc":
                        try:
                            if abs(right - left) / float(abs(left)) >= precision:
                                diffs = (left, right, right - left)
                        except:
                            diffs = (left, right)
                    else:
                        raise Exception("comp_mode='%s' not supported" % (comp_mode))
            else:
                diffs = (left, right)
        else:
            diffs = (left, right)
        return diffs

    return diff_rec_(left, right)


def get_first_item(d, l, f):
    for i in l:
        if d.get(f(i), None):
            return i
    return


def dict_product(d):
    keys = list(d.keys())
    return [
        {x[0]: x[1] for x in zip(keys, item)} for item in list(product(*d.values()))
    ]


def flatten_dict(d, pre_key=""):
    if isinstance(d, dict):
        this_d = {}
        for k, v in d.items():
            this_key = f"{pre_key}__{k}" if pre_key else k
            if not isinstance(v, dict):
                this_d[this_key] = v
            else:
                this_d.update(flatten_dict(v, pre_key=this_key))
        return this_d
    return d
