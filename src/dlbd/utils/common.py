import collections.abc
from itertools import product


def deep_dict_update(original, update, copy=False):
    """Recursively update a dict.

    Subdict's won't be overwritten but also updated.
    """
    if copy:
        original = dict(original)
    if not isinstance(original, collections.Mapping):
        if copy:
            update = dict(update)
        return update
    for key, value in update.items():
        if isinstance(value, collections.Mapping):
            original[key] = deep_dict_update(original.get(key, {}), value, copy)
        else:
            original[key] = value
    return original


def to_range(opts):
    return list(range(opts["start"], opts["end"], opts.get("step", 1)))


def expand_options_dict(options):
    """Function to expand the options found in an options dict. If an option is a dict,
    two possibilities arise:
        - if the key "start" and "end" are present, then the dict is treated as a range and is
        replaced by a list with all values from the range. A "step" key can be found to define
        the step of the range. Default step: 1
        - Otherwise, the dict is expanded in a recursive manner.

    Args:
        options (dict): dict containing the options

    Returns:
        list: A list of dicts each containing a set of options
    """
    res = []
    tmp = []
    for val in options.values():
        if isinstance(val, dict):
            if "start" in val and "end" in val:
                tmp.append(to_range(val))
            else:
                tmp.append(expand_options_dict(val))
        else:
            if not isinstance(val, list):
                val = [val]
            tmp.append(val)
    for v in product(*tmp):
        d = dict(zip(options.keys(), v))
        res.append(d)
    return res


def get_dict_path(dict_obj, path, default=None, sep="--"):
    if not isinstance(path, list):
        path = path.split(sep)
    if not dict_obj:
        print("Warning! Empty dict provided. Returning default value")
        return default
    if not path:
        print("Warning! Empty path provided. Returning default value")
        return default

    key = path.pop(0)
    value = dict_obj.get(key, default)

    if path:
        if isinstance(value, dict):
            return get_dict_path(value, path, default)
        else:
            print(
                (
                    "Warning! Path does not exists as the value for key '{}' is not a dict."
                    + " Returning default value."
                ).format(key)
            )
            return default

    return value

