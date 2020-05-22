import types


def hasmethod(obj, name):
    return hasattr(obj, name) and type(getattr(obj, name)) == types.MethodType


def get_attribute_or_method_result(obj, att_name):
    if hasmethod(obj, att_name):
        return getattr(obj, att_name)()
    return getattr(obj, att_name, None)
