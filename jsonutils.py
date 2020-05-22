import collections
from decimal import Decimal
import json

import pandas as pd
import simplejson


def json_default(obj):
    try:
        return obj.to_dict()
    except Exception:
        try:
            return obj.to_json()
        except:
            if pd.api.types.is_integer(obj):
                return int(obj)
            elif pd.api.types.is_number(obj):
                return float(obj)
            elif isinstance(obj, collections.deque):
                return list(obj)
    raise TypeError(str(type(obj)) + ": " + str(obj))


class DefaultJSONEncoder(json.JSONEncoder):

    def default(self, obj):
        try:
            return json_default(obj)
        except:
            pass
        return json.JSONEncoder.default(self, obj)


def stringify(obj):
    if isinstance(obj, list):
        return [stringify(x) for x in obj]
    elif isinstance(obj, dict):
        return {
            str(k): stringify(v)
            for k, v in obj.items()}
    return obj


def json_dumps(obj, indent=4, outfile=None, encode=False):
    if outfile:
        return simplejson.dump(stringify(obj),
                               outfile,
                               default=json_default)
    else:
        val = simplejson.dumps(stringify(obj),
                               default=json_default,
                               indent=indent,
                               sort_keys=True)
        if encode:
            return val.encode('ascii')
        return val


def json_loads(obj, decode=True):
    if decode:
        return simplejson.loads(obj.decode('utf-8'))
    else:
        return simplejson.loads(obj)


def json_load(path):
    with open(path) as jsonfile:
        data = simplejson.load(jsonfile)
    return data


def get_dict(val):
    if isinstance(val, dict):
        return val
    return json_loads(val, decode=False)
