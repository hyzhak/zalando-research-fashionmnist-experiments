def get_model_task_by_name(name):
    try:
        m = __import__(f'src.models.{name}', fromlist=[''])
    except ModuleNotFoundError:
        raise Exception(f'There is no such model as {name}')

    print('dir(m)', dir(m))

    if 'Model' not in dir(m):
        raise Exception(f'Model {name} should define Model class')

    return m.Model
