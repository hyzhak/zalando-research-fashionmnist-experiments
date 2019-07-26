import luigi
from tensorflow.keras.callbacks import Callback


class LuigiTaskCallback(Callback):
    def __init__(self, task: luigi.Task, name, num_of_samples):
        self._task = task
        self._name = name
        self._num_of_samples = num_of_samples

    def on_predict_batch_end(self, batch, logs={}):
        self._task.set_status_message(
            f'Predict ({self._name}): {batch} / {self._num_of_samples}'
        )
        self._task.set_progress_percentage(round(100 * batch / self._num_of_samples, 1))
