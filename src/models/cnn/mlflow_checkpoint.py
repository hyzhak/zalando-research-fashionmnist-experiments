import math
import mlflow
import mlflow.keras
import numpy as np
from sklearn import metrics
from tensorflow import keras
from tensorflow.keras.callbacks import Callback
import time


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

        self._best_train_loss = math.inf
        self._best_val_loss = math.inf
        self._best_train_acc = -math.inf
        self._best_val_acc = -math.inf
        self._best_test_acc = -math.inf

        self._best_model = None
        self._epoch_train_time_sum = 0
        self._next_step = 0
        self._epoch_start_at = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Log the best model at the end of the training run.
        """
        if not self._best_model:
            raise Exception('Failed to build any model')

        # do we really need it?
        # mlflow.log_metrics({
        #     f'loss.train': self._best_train_loss,
        #     f'loss.val': self._best_val_loss,
        # }, step=self._next_step)

        # log the best model
        mlflow.keras.log_model(self._best_model, 'model')

    def get_best_metrics(self):
        return {
            'loss': {
                'train': float(self._best_train_loss),
                'val': float(self._best_val_loss),
            },
            'accuracy': {
                'train': float(self._best_train_acc),
                'val': float(self._best_val_acc),
                'test': float(self._best_test_acc),
            },
            'train_time': {
                'epoch': self._epoch_train_time_sum / self._next_step,
                'total': self._epoch_train_time_sum
            }
        }

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start_at = time.time()

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
        train_acc = logs['acc']
        val_acc = logs['val_acc']
        epoch_train_time = time.time() - self._epoch_start_at

        self._epoch_train_time_sum += epoch_train_time

        mlflow.log_metrics({
            f'loss.train': train_loss,
            f'loss.val': val_loss,
            f'accuracy.train': logs['acc'],
            f'accuracy.val': logs['val_acc'],
            f'train_time.epoch': epoch_train_time
        }, step=epoch)

        if val_loss < self._best_val_loss:
            # The result improved in the validation set.
            # Log the model with mlflow and also evaluate and log on test set.
            self._best_train_loss = train_loss
            self._best_val_loss = val_loss
            self._best_train_acc = train_acc
            self._best_val_acc = val_acc
            self._best_model = keras.models.clone_model(self.model)
            self._best_model.set_weights([x.copy() for x in self.model.get_weights()])
            preds = self._best_model.predict(self._test_x)

            # evaluate model on test set
            self._best_test_acc = metrics.accuracy_score(self._test_y,
                                                         np.argmax(preds, axis=1))
            mlflow.log_metric(f'accuracy.test',
                              self._best_test_acc, step=epoch)
