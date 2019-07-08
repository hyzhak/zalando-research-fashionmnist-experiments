"""
2 layers Fully connected (FC) network
"""
import luigi
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

from src.models.tf_classifier_base import TFClassifierBase


class TFClassifierCNNTiny(TFClassifierBase):
    dropout = luigi.IntParameter(
        default=0.5,
        description='Dropout'
    )

    def model(self, input_shape):
        return keras.Sequential([
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
            Dropout(self.dropout),
            Dense(10),
            Activation('softmax'),
        ])


if __name__ == '__main__':
    luigi.run()
