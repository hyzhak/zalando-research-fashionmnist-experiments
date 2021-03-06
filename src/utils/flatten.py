import collections


def flatten(d, parent_key='', sep='.'):
    """
    based on https://stackoverflow.com/a/6027615/1324730
    :param d:
    :param parent_key:
    :param sep:
    :return:
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten(d, sep='.'):
    """
    unflat dictionary to the deep dictionary

    :param d:
    :param sep:
    :return:
    """
    res = {}
    for k, value in d.items():
        set_deep_value(res, k.split(sep), value)
    return res


def set_deep_value(dd, keys, value):
    latest = keys.pop()
    for k in keys:
        dd = dd.setdefault(k, {})
    dd.setdefault(latest, value)
