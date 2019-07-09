import os
import sys


def get_project_name() -> str:
    if not hasattr(sys.modules['__main__'], '__file__'):
        return ''

    abs_path = os.path.dirname(os.path.abspath(__file__))
    rel_path = os.path.dirname(sys.modules['__main__'].__file__)
    root_path = abs_path[:-len(rel_path) - 1]
    root_path_split = os.path.split(root_path)
    return root_path_split[-1]
