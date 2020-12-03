import collections.abc


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
