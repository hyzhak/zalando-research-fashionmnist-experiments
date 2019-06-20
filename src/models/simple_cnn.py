import luigi
import mlflow
import numpy as np
import random as rn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import optimizers
import time

from src.data.external_test_set import ExternalTestSet
from src.data.external_train_set import ExternalTrainSet
from src.models.mlflow_checkpoint import MLflowCheckpoint
from src.utils.extract_x_y import extract_x_and_y, reshape_X_to_2d
from src.utils.params_to_filename import encode_task_to_filename


class SimpleCNN(luigi.Task):
    model_name = 'simple_cnn'

    experiment_id = luigi.Parameter(
        default='',
        significant=False,
    )
    verbose = luigi.IntParameter(
        default=1,
        significant=False
    )
    batch_size = luigi.IntParameter(
        default=16,
        description='Batch size passed to the learning algo.'
    )
    metrics = luigi.Parameter(
        default='accuracy'
    )
    loss = luigi.Parameter(
        default='sparse_categorical_crossentropy'
    )
    epoch = luigi.IntParameter(
        default=5
    )
    optimizer = luigi.Parameter(
        default='adam',
        description='optimizer (SGD, Adam, etc)'
    )
    optimizer_props = luigi.DictParameter(
        default={}
    )
    valid_size = luigi.FloatParameter(
        default=0.1
    )
    random_seed = luigi.IntParameter(
        default=12345
    )

    def requires(self):
        return {
            'test': ExternalTestSet(),
            'train': ExternalTrainSet(),
        }

    def output(self):
        filename = encode_task_to_filename(self)

        return {
            'model': luigi.LocalTarget(
                f'models/{self.model_name}/{filename}'
            ),
            'score': luigi.LocalTarget(
                f'reports/scores/{self.model_name}/{filename}'
            ),
        }

    def run(self):
        # seed random everywhere
        # there is still problem with GPU
        # https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
        tf.random.set_random_seed(self.random_seed)
        np.random.seed(self.random_seed)
        rn.seed(self.random_seed)

        with mlflow.start_run(experiment_id=self.experiment_id if self.experiment_id else None,
                              nested=self.experiment_id is not None) as run:
            # scores['run_id'] = run.info.run_id
            X_train, X_valid, y_train, y_valid = train_test_split(
                *extract_x_and_y(self.input()['train']),
                test_size=self.valid_size,
                random_state=self.random_seed,
            )
            X_test, y_test = extract_x_and_y(self.input()['test'])
            model = self._train_model(
                reshape_X_to_2d(X_train), y_train,
                reshape_X_to_2d(X_valid), y_valid,
                reshape_X_to_2d(X_test), y_test
            )

            # TODO: store the best model

    def _train_model(self,
                     train_x, train_y,
                     valid_x, valid_y,
                     test_x, test_y):
        # doesn't callback have other ways to catch exception
        # inside of model training loop?
        with MLflowCheckpoint(test_x, test_y,
                              self.metrics) as mlflow_logger:
            # TODO: add batch normalization and dropout
            input_shape = (28, 28, 1)

            model = keras.Sequential([
                Conv2D(32, (3, 3), input_shape=input_shape),
                Activation('relu'),
                MaxPooling2D(pool_size=(2, 2)),

                Conv2D(32, (3, 3)),
                Activation('relu'),
                MaxPooling2D(pool_size=(2, 2)),

                Conv2D(64, (3, 3)),
                Activation('relu'),
                MaxPooling2D(pool_size=(2, 2)),

                Flatten(),
                Dense(64),
                Activation('relu'),
                Dropout(0.5),
                Dense(10),
                Activation('softmax'),
            ])

            # we are getting instance of optimizer here
            optimizer = optimizers.get(self.optimizer)
            # so to create new with out setting we should use `from_config`
            model.compile(optimizer=optimizer.from_config(self.optimizer_props),
                          loss=self.loss,
                          metrics=[self.metrics])

            # log model params to mlflow
            mlflow.log_param('model_name', self.model_name)
            mlflow.log_params(self.to_str_params(only_significant=True, only_public=True))
            mlflow.log_param('num_of_model_params', model.count_params())

            if self.verbose > 0:
                model.summary()

            start = time.time()
            model.fit(
                train_x, train_y,
                epochs=self.epoch,
                batch_size=self.batch_size,
                verbose=self.verbose,
                validation_data=(valid_x, valid_y),
                # TODO: add EarlyStopping, ModelCheckpoint, TensorBoard
                callbacks=[mlflow_logger]
            )
            training_time = time.time() - start
            mlflow.log_metric('total_train_time', training_time)
        return model

    def _predict(self):
        pass


if __name__ == '__main__':
    # with mlflow.start_run():
    luigi.run()
