import luigi
import mlflow
import numpy as np
import os
import shutil
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras import optimizers
import time
import yaml

from src.data.external_label_titles import ExternalLabelTitles
from src.data.external_test_set import ExternalTestSet
from src.data.external_train_set import ExternalTrainSet
from src.features.augmentation import build_augmentation_generator
from src.models.cnn.log_confusion_matrix import LogConfusionMatrix
from src.models.cnn.mlflow_checkpoint import MLflowCheckpoint
from src.utils.extract_x_y import get_train_valid_test_subsets, reshape_X_to_2d
from src.utils.flatten import flatten
from src.utils.mlflow_task import MLFlowTask
from src.utils.params_to_filename import get_params_of_task, encode_task_to_filename
from src.utils.project import get_project_name
from src.utils.seed_randomness import seed_randomness

# FIXME:
# when run simple_cnn under ax got error
# "Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR"
#
# this solution helped from https://github.com/tensorflow/tensorflow/issues/24496#issuecomment-500605481
#
# TODO: can I move it on the top of file?
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))


class TFClassifierBase(MLFlowTask):
    augmentation = luigi.DictParameter(
        default=None,
        description='use augmentation of images'
    )
    batch_size = luigi.IntParameter(
        default=16,
        description='Batch size passed to the learning algorithm.'
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
    train_size = luigi.OptionalParameter(
        default=None
    )
    random_seed = luigi.IntParameter(
        default=12345
    )
    log_confusion_matrix = luigi.BoolParameter(
        default=True,
        significant=False,
    )
    # dir where TensorBoard callback will put logs
    tf_log_dir = luigi.Parameter(
        default=f'/var/models/1/logs/{get_project_name()}',
        significant=False,
    )
    verbose = luigi.IntParameter(
        default=1,
        significant=False
    )

    def requires(self):
        return {
            'test': ExternalTestSet(),
            'train': ExternalTrainSet(),
        }

    def ml_output(self, output_dir):
        return {
            'metrics': luigi.LocalTarget(
                os.path.join(output_dir, 'metrics.yml')
            ),
            'model': luigi.LocalTarget(
                os.path.join(output_dir, 'model.h5'),
                format=luigi.format.Nop
            ),
            'params': luigi.LocalTarget(
                os.path.join(output_dir, 'params.yml')
            ),
        }

    def ml_run(self, run_id=None):
        seed_randomness(self.random_seed)

        X_train, X_valid, X_test, y_train, y_valid, y_test = get_train_valid_test_subsets(
            self.train_size,
            self.valid_size,
            self.random_seed,
            self.input()['train'],
            self.input()['test']
        )

        # TODO: do we need to place preprocessing to separate task?

        X_train = X_train.astype(np.float32) / 255.0
        X_valid = X_valid.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0

        # TODO: can't we get labels inside of self._train_model?
        labels = None
        if self.log_confusion_matrix:
            labels_input = yield ExternalLabelTitles()
            with labels_input.open('r') as f:
                labels = yaml.load(f)

        checkpoint_path, metrics = self._train_model(
            reshape_X_to_2d(X_train), y_train,
            reshape_X_to_2d(X_valid), y_valid,
            reshape_X_to_2d(X_test), y_test,
            labels
        )

        # we move checkpoint model to the place of target model
        # because it's the best model for the moment
        os.rename(checkpoint_path, self.output()['model'].path)

        # store accuracy and loss on train, test, validate sets
        with self.output()['metrics'].open('w') as f:
            yaml.dump(metrics, f, default_flow_style=False)

    def model(self, input_shape):
        raise Exception('model should be overwritten in inheritor')

    def _train_model(self,
                     train_x, train_y,
                     valid_x, valid_y,
                     test_x, test_y,
                     labels):
        if self.augmentation:
            train_generator = build_augmentation_generator(
                train_x, train_y,
                batch_size=self.batch_size,
                random_seed=self.random_seed,
                **self.augmentation,
            )
        else:
            train_generator = None

        # TODO: doesn't callback have other ways to catch exception
        # inside of model training loop?
        with MLflowCheckpoint(test_x, test_y,
                              self.metrics) as mlflow_logger:

            # TODO: add batch normalization and dropout

            model = self.model(input_shape=(28, 28, 1))
            if model is None:
                raise Exception('Model should be defined')

            # TODO: it is solution to save weight only
            #
            # tensorflow:This model was compiled with a Keras optimizer
            # (<tensorflow.python.keras.optimizers.Adam object at 0x7fe86828bf28>)
            # but is being saved in TensorFlow format with `save_weights`.
            # The model's weights will be saved, but unlike with TensorFlow optimizers
            # in the TensorFlow format the optimizer's state will not be saved.
            #
            # Consider using a TensorFlow optimizer from `tf.train`.
            #
            # name_to_optimizer = {
            #     'adam': tf.train.AdamOptimizer,
            # }
            # optimizer = name_to_optimizer[self.optimizer](**self.optimizer_props)
            # print('tf.train[self.optimizer]', tf.train[self.optimizer])

            #
            # but when we need to save not only weights tf.train doesn't work properly
            #

            # WARNING:tensorflow:TensorFlow optimizers do not make it possible to access optimizer attributes
            # or optimizer state after instantiation. As a result, we cannot save the optimizer as part of
            # the model save file.You will have to compile your model again after loading it.
            # Prefer using a Keras optimizer instead (see keras.io/optimizers).

            # we are getting instance of optimizer here
            optimizer = optimizers.get(self.optimizer)
            # so to create new with out setting we should use `from_config`
            optimizer = optimizer.from_config(self.optimizer_props)

            model.compile(optimizer=optimizer,
                          loss=self.loss,
                          metrics=[self.metrics])

            model_name = encode_task_to_filename(self)
            # log model params to mlflow
            mlflow.log_param('model_name', model_name)

            # for the moment mlflow doesn't support nested params
            # so we need to flatten them
            params = get_params_of_task(self)
            mlflow.log_params(flatten(params))
            with self.output()['params'].open('w') as f:
                yaml.dump(params, f, default_flow_style=False)
            mlflow.log_param('model.num_of_params', model.count_params())

            if self.verbose > 0:
                model.summary()

            tf_log_dir = os.path.join(
                self.tf_log_dir,
                model_name
            )

            # remove previous log to prevent duplication
            # once I found way to resume training it could make sense to preserve it

            # TODO: sadly sometime it doesn't because TensorFlow Board could cache
            # logs and if you don't refresh tfb after logs were deleted you will get old logs
            shutil.rmtree(tf_log_dir, ignore_errors=True)

            # so temporal solution - add random tail at the end of path
            # it would force invalidate logs for tfb
            tf_log_dir = os.path.join(tf_log_dir, str(int(time.time())))

            start = time.time()

            # create needed dirs to store model checkpoint
            output_model = self.output()['model']
            output_model.makedirs()
            model_checkpoint_path = f'{output_model.path}_checkpoint'

            callbecks = [
                EarlyStopping(monitor='val_loss', patience=2),
                # isn't clear where to store and how would it work with self.output()['model']
                ModelCheckpoint(
                    filepath=model_checkpoint_path,
                    save_best_only=True,
                    # FIXME:
                    # the goal of that saving that we can continue train from this point
                    # but it doesn't work properly because TF doesn't allow Keras optimizer
                    # save_weights_only=True,
                    save_weights_only=False,
                ),
                TensorBoard(
                    log_dir=tf_log_dir,
                    write_images=True,
                ),
                # TODO: how can I use it?
                # LearningRateScheduler
                # should be optional
                # ReduceLROnPlateau
                mlflow_logger
            ]

            if self.log_confusion_matrix:
                callbecks.append(LogConfusionMatrix(valid_x, valid_y, labels))

            fit_args = dict(
                epochs=self.epoch,
                verbose=self.verbose,
                validation_data=(valid_x, valid_y),
                callbacks=callbecks
            )

            if train_generator:
                model.fit_generator(
                    train_generator,
                    **fit_args
                )
            else:
                model.fit(
                    train_x, train_y,
                    batch_size=self.batch_size,
                    **fit_args
                )
            training_time = time.time() - start
            mlflow.log_metric('train_time.total', training_time)

            metrics = mlflow_logger.get_best_metrics()

        return model_checkpoint_path, metrics
