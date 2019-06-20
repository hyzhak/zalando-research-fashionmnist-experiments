import math
import mlflow
import mlflow.keras
import numpy as np
from sklearn import metrics
from tensorflow import keras
from tensorflow.keras.callbacks import Callback


def eval_and_log_metrics(prefix, y_true, y_pred, epoch):
    print('# eval_and_log_metrics')
    metric_name = 'accuracy'
    print('(y_true, y_pred)', (y_true.shape, y_pred.shape))
    metric = metrics.accuracy_score(y_true, y_pred)
    mlflow.log_metric(f'{metric_name} {prefix}', metric, step=epoch)
    return metric


class MLflowCheckpoint(Callback):
    """
    based on:
    https://github.com/mlflow/mlflow/blob/master/examples/hyperparam/train.py

    Example of Keras MLflow logger.
    Logs training metrics and final model with MLflow.
    We log metrics provided by Keras during training and keep track of the best model (best loss
    on validation dataset). Every improvement of the best model is also evaluated on the test set.
    At the end of the training, log the best model with MLflow.
    """

    def __init__(self, test_x, test_y, loss='accuracy'):
        self._test_x = test_x
        self._test_y = test_y
        self._target_loss = loss

        self.train_loss = f'{loss} train'
        self.val_loss = f'{loss} val'
        self.test_loss = f'{loss} test'

        self._best_train_loss = math.inf
        self._best_val_loss = math.inf
        self._best_model = None
        self._next_step = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Log the best model at the end of the training run.
        """
        if not self._best_model:
            raise Exception('Failed to build any model')

        mlflow.log_metrics({
            f'{self._target_loss} loss train': self._best_train_loss,
            f'{self._target_loss} loss val': self._best_val_loss,
        }, step=self._next_step)

        mlflow.keras.log_model(self._best_model, 'model')

    def on_epoch_end(self, epoch, logs=None):
        """
        Log Keras metrics with MLflow. If model improved on the validation data, evaluate it on
        a test set and store it as the best model.
        """
        if not logs:
            return
        self._next_step = epoch + 1
        train_loss = logs['loss']
        val_loss = logs['val_loss']

        mlflow.log_metrics({
            f'{self._target_loss} loss train': train_loss,
            f'{self._target_loss} loss val': val_loss,
            f'{self._target_loss} acc train': logs['acc'],
            f'{self._target_loss} acc val': logs['val_acc'],
        }, step=epoch)

        if val_loss < self._best_val_loss:
            # The result improved in the validation set.
            # Log the model with mlflow and also evaluate and log on test set.
            self._best_train_loss = train_loss
            self._best_val_loss = val_loss
            self._best_model = keras.models.clone_model(self.model)
            self._best_model.set_weights([x.copy() for x in self.model.get_weights()])
            preds = self._best_model.predict(self._test_x)
            eval_and_log_metrics('test',
                                 self._test_y,
                                 np.argmax(preds, axis=1), epoch)
