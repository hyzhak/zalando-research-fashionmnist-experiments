from unittest.mock import Mock, MagicMock

from src.utils.params_to_filename import get_params_of_task, params_to_filename


def test_params_to_filename_with_simple_params():
    assert params_to_filename({
        'num': 1,
        'str': 'value'
    }) == 'num=1__str=value'


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
    }) == 'params.a=1__params.b=hello__params.c.d=2__params.c.e=world'


def test_params_to_filename_with_non_params():
    assert params_to_filename({}) is ''


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
