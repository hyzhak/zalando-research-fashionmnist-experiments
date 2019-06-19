from src.utils.params_to_filename import params_to_filename


def test_params_to_filename_with_simple_params():
    assert params_to_filename({
        'num': 1,
        'str': 'value'
    }) == 'num=1__str=value'


def test_params_to_filename_with_non_params():
    assert params_to_filename({}) == 'default'
