import luigi
import json


def encode_value(v):
    if v is dict:
        return v
    else:
        return str(v)


def gen_deep_key_and_value(d):
    for key, value in d.items():
        if type(value) is dict:
            for nested_key, nested_value in gen_deep_key_and_value(value):
                yield [key] + nested_key, nested_value
        else:
            yield [key], value


def params_to_filename(d):
    return '__'.join(
        [f'{".".join(key_path)}={value}' for key_path, value in gen_deep_key_and_value(d)]
    )


def get_params_of_task(task, exclude=[]):
    """
    for some reasons luigi encode dict params to json. we revert it here
    :param task:
    :param exclude:
    :return:
    """
    # undo encoding of dicts to string
    params = task.to_str_params(only_significant=True, only_public=True)
    # each value is encoded to string, event when it is dict
    # I'm trying to unwrap dict, because default dict encoding isn't good enough for filename
    for param_name, param_value in params.items():
        if param_name in exclude:
            continue
        if isinstance(param_value, str):
            try:
                param_value = json.loads(param_value)
                params[param_name] = param_value
            except:
                pass
    return params


def encode_task_to_filename(task, exclude=[]):
    # family = task.get_task_family()
    # encoded_params = params_to_filename(
    #     task.to_str_params(only_significant=True, only_public=True)) or 'default'

    # return '__'.join((family, encoded_params))

    # undo encoding of dicts to string
    return params_to_filename(
        get_params_of_task(task, exclude)
    ) or 'default'
