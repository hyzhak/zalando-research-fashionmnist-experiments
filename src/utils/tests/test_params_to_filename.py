import os
from unittest.mock import Mock

from src.utils.params_to_filename import get_params_of_task, params_to_filename


def test_params_to_filename_with_simple_params():
    assert params_to_filename({
        'num': 1,
        'str': 'value'
    }) == os.path.join('num=1', 'str=value')


def test_params_to_filename_with_nested_params():
    assert params_to_filename({
        'params': {
            'a': 1,
            'b': 'hello',
            'c': {
                'd': 2,
                'e': 'world'
            }
        }
    }) == os.path.join(
        'params.a=1', 'params.b=hello', 'params.c.d=2', 'params.c.e=world'
    )


def test_params_to_filename_with_non_params():
    assert params_to_filename({}) is None


def test_get_params_of_task():
    task = Mock()
    task.to_str_params.return_value = {
        'batch_size': '16',
        'metrics': 'accuracy',
        'loss': 'sparse_categorical_crossentropy',
        'epoch': '5',
        'optimizer': 'adam',
        'optimizer_props':
            '{"lr": 1.4772678135469222e-06, "beta_1": 0.7221158963084221}',
        'valid_size': '0.1',
        'train_size': '',
        'random_seed': '12345'
    }
    assert get_params_of_task(task) == {
        'batch_size': 16,
        'metrics': 'accuracy',
        'loss': 'sparse_categorical_crossentropy',
        'epoch': 5,
        'optimizer': 'adam',
        'optimizer_props':
            {
                "lr": 1.4772678135469222e-06,
                "beta_1": 0.7221158963084221
            },
        'valid_size': 0.1,
        'train_size': '',
        'random_seed': 12345
    }
