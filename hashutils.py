import hashlib
import inspect
import copy


class VersionManager():
    _versions = {}

    def get_class_version(self, inst):
        cls = inst.__class__
        try:
            return self._versions[cls]
        except KeyError:
            version = hashlib.sha1(VersionManager.get_source(inst)).hexdigest()
            self._versions[cls] = version
            return version

    @staticmethod
    def get_source(inst):
        return '\n'.join(
            [''.join(inspect.getsourcelines(f[1])[0])
                for f in inspect.getmembers(inst, predicate=inspect.ismethod)]
        )

VERSION_MANAGER = VersionManager()


def _make_hash(obj, debug=False):
    # memory optimized http://stackoverflow.com/a/8714242
    """
    make_hash({'x': 1, 'b': [3,1,2,'b'], 'c': {1:2}})
    make_hash({'x': 1, 'b': [3,1,2,'b'], 'c': {1:2}})
    make_hash({'x': 1, 'b': [3,1,2,'b']})
    make_hash({'x': 1, 'b': [3,1,2,'b']})
    make_hash({'x': 1, 'b': [3,1,2,'b']})
    """
    try:
        if isinstance(obj, (set, tuple, list)):
            if isinstance(obj, (set,)):
                obj = sorted(list(obj))
            return tuple(map(make_hash, obj))
        elif not isinstance(obj, dict):
            return hash(obj)
        try:
            new_o = copy.deepcopy(obj)
        except:
            if debug:
                import pickle
                pickle.dump(obj, open("deepcopy_failure.p", "wb"))
            raise
        nitems = sorted(new_o.items())
        for k, v in nitems:
            if v is not None:
                new_o[k] = make_hash(v)
            else:
                new_o[k] = make_hash("None")
        new_oo = sorted(new_o.items())
        return hash(tuple(frozenset(new_oo)))
    except Exception as e:
        if not debug:
            _make_hash(obj, debug=True)
        raise Exception(e)


def make_hash(params):
    """
    Calculate sha1 of a dictionary

    :param params: dict - a dictionary
    :return: str - calculated hash id of params
    """
    return hashlib.sha1(str(_make_hash(params)).encode('utf-8')).hexdigest()
