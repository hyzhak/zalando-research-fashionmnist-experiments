import re

_reg = re.compile(r'(?!^)(?<!_)([A-Z])')


def get_class_name_as_snake(instance):
    """
    get name of class as snake_string
    """
    s = type(instance).__name__
    return _reg.sub(r'_\1', s).lower()
