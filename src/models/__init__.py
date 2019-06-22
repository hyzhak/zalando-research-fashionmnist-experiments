from . import simple_cnn


def get_model_task_by_name(name):
    if name == 'simple_cnn':
        return simple_cnn.SimpleCNN
    else:
        raise NotImplementedError()
