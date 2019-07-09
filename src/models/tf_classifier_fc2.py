"""
2 layers Fully connected (FC) network
"""
import luigi
import tensorflow as tf
from tensorflow import keras

from src.models.tf_classifier_base import TFClassifierBase


class TFClassifierFC2(TFClassifierBase):
    size_of_hidden_layer = luigi.IntParameter(
        default=128,
        description='The size of hidden layer'
    )

    def model(self, input_shape):
        return keras.Sequential([
            keras.layers.Flatten(input_shape=input_shape),
            keras.layers.Dense(self.size_of_hidden_layer, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])


# export model
Model = TFClassifierFC2

if __name__ == '__main__':
    luigi.run()
