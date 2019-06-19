def params_to_filename(d):
    return '__'.join(f'{key}={value}' for key, value in d.items()) or 'default'


def encode_task_to_filename(task):
    family = task.get_task_family()
    encoded_params = params_to_filename(
        task.to_str_params(only_significant=True, only_public=True))

    return '__'.join((family, encoded_params))
