def should_minimize(name):
    """
    should we minimize or optimize?
    :param name:
    :return:
    """
    return name != 'accuracy'


def is_better_score(name, old, new):
    """
    Is new better then old
    :param name: metric
    :param old:
    :param new:
    :return:
    """
    if should_minimize(name):
        return old > new
    else:
        return old < new
