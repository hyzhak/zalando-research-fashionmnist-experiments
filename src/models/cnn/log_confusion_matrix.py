import mlflow
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import Callback
import time


class LogConfusionMatrix(Callback):
    """
    Log confusion matrix to mlflow
    """

    def __init__(self, valid_x, valid_y, labels, normalize=True):
        self._valid_x = valid_x
        self._valid_y = valid_y
        self._labels = labels
        self._size = len(labels)
        self._normalize = normalize

    def on_epoch_end(self, epoch, logs=None):
        start_time = time.time()
        y_preds = np.argmax(self.model.predict(self._valid_x), axis=1)
        cm = confusion_matrix(self._valid_y, y_preds)
        if self._normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        cm = cm.ravel()
        mlflow.log_metrics({
            f'cm_{self._labels[idx // self._size]}_{self._labels[idx % self._size]}': cm[idx]
            for idx in range(self._size * self._size)
        }, step=epoch)
        delta_time = time.time() - start_time
        mlflow.log_metric(f'train_time.confusion_matrix', delta_time, step=epoch)
