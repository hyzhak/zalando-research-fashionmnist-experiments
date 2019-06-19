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


def encode_task_to_filename(task):
    family = task.get_task_family()
    encoded_params = params_to_filename(
        task.to_str_params(only_significant=True, only_public=True)) or 'default'

    return '__'.join((family, encoded_params))
