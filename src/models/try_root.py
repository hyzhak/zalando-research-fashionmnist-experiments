import os
import sys

abs_path = os.path.dirname(os.path.abspath(__file__))
rel_path = os.path.dirname(sys.modules['__main__'].__file__)
root_path = abs_path[:-len(rel_path) - 1]
print('abs_path', abs_path)
print('rel_path', rel_path)
print('root_path', root_path)
root_path_split = os.path.split(root_path)
print('split', root_path_split)
print('project directory', root_path_split[-1])

#
# print()
# print('sys.modules dir', dir(sys.modules))
# print()
# print("dir sys.modules['__main__']", dir(sys.modules['__main__']))
# print()
# print("sys.modules['__main__']", sys.modules['__main__'])
# print()
# print('sys.modules', sys.modules)
