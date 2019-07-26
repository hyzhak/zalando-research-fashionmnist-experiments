import luigi
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
import numpy as np
import os
import pandas as pd
from tensorflow.keras import applications
from time import time

from src.data.image_files import ImageFiles
from src.utils.params_to_filename import encode_task_to_filename
from src.utils.snake import get_class_name_as_snake
from src.utils.luigi_task_callback import LuigiTaskCallback


class LatentSpaceFeature(luigi.Task):
    model = luigi.Parameter(
        default='vgg16'
    )

    batch_size = luigi.IntParameter(
        default=32,
        description='batch size',
        significant=False,
    )

    def requires(self):
        return ImageFiles()

    def output(self):
        class_name = get_class_name_as_snake(self)
        encoded_params = encode_task_to_filename(self)
        return {
            'test': luigi.LocalTarget(
                os.path.join('data', 'interim', class_name, encoded_params,
                             'test.parquet.gzip'),
                format=luigi.format.Nop
            ),
            'train': luigi.LocalTarget(
                os.path.join('data', 'interim', class_name, encoded_params,
                             'train.parquet.gzip'),
                format=luigi.format.Nop
            ),
        }

    def _process_and_store(self, name,
                           input_dir, output_file,
                           image_size,
                           preprocessing, model):
        def preprocessing_fn(i):
            # Upscale the images to input_size*input_size as required by pre-trained models
            i = img_to_array(array_to_img(i, scale=False).resize((image_size, image_size)))
            return preprocessing(i)

        ig = ImageDataGenerator(
            data_format='channels_last',
            preprocessing_function=preprocessing_fn,
        )
        gen = ig.flow_from_directory(
            input_dir.path,
            target_size=(image_size, image_size),
            class_mode=None,
            shuffle=False,
            batch_size=self.batch_size,
        )
        filenames = gen.filenames
        num_of_samples = len(filenames) / self.batch_size

        gen.reset()

        features = model.predict_generator(gen,
                                           callbacks=[
                                               LuigiTaskCallback(
                                                   self, name, num_of_samples
                                               ),
                                           ],
                                           workers=4,
                                           use_multiprocessing=True,
                                           steps=num_of_samples,
                                           verbose=1)

        features = np.squeeze(features)
        print('features.shape (after squeezing):', features.shape)

        output_file.makedirs()

        df = pd.DataFrame(features,
                          columns=[f'f_{i}' for i in range(features.shape[1])],
                          dtype=np.float32)

        start_save = time()
        df.to_parquet(output_file.path,
                      compression='brotli')
        # TODO: I might need that information to store in mlflow (or not?)
        print('saved to parquet in: ', time() - start_save, 'sec')

    def run(self):
        # weights of models will be loaded from
        # https://github.com/fchollet/deep-learning-models/releases
        image_size = 48
        if self.model == 'vgg16':
            model = applications.vgg16.VGG16(include_top=False,
                                             input_shape=(image_size, image_size, 3),
                                             weights='imagenet')
            preprocessing = applications.vgg16.preprocess_input
        elif self.model == 'resnet':
            model = applications.resnet50.ResNet50(include_top=False,
                                                   weights='imagenet',
                                                   # It should have exactly 3 inputs channels,
                                                   # and width and height should be no smaller than 32
                                                   input_shape=(image_size, image_size, 3),
                                                   # 'avg' means that global average pooling will be applied
                                                   # to the output of the last convolutional layer, and thus
                                                   # the output of the model will be a 2D tensor.
                                                   pooling='avg')
            preprocessing = applications.resnet50.preprocess_input
        elif self.model == 'mobilenet':
            image_size = 128
            # If imagenet weights are being loaded, input must have a static square shape
            # (one of (128, 128), (160, 160), (192, 192), or (224, 224))
            model = applications.mobilenet.MobileNet(include_top=False,
                                                     weights='imagenet',
                                                     # It should have exactly 3 inputs channels,
                                                     # and width and height should be no smaller than 32
                                                     input_shape=(image_size, image_size, 3),
                                                     # controls the width of the network
                                                     # alpha=0,

                                                     # depth_multiplier: depth multiplier for depthwise convolution(also called the resolution multiplier)

                                                     # dropout

                                                     # 'avg' means that global average pooling will be applied
                                                     # to the output of the last convolutional layer, and thus
                                                     # the output of the model will be a 2D tensor.
                                                     pooling='avg')
            preprocessing = applications.mobilenet.preprocess_input
        elif self.model == 'xception':
            image_size = 71  # default 299x299
            model = applications.xception.Xception(include_top=False,
                                                   weights='imagenet',

                                                   input_shape=(image_size, image_size, 3),
                                                   pooling='avg')
            preprocessing = applications.xception.preprocess_input
        else:
            # TODO: add other model
            raise NotImplementedError()

        model.summary()
        start_time = time()
        self._process_and_store(
            'train',
            self.input()['train'],
            self.output()['train'],
            image_size=image_size,
            preprocessing=preprocessing,
            model=model,
        )
        print('processed train:', (time() - start_time) / 1000, 's')

        start_time = time()
        self._process_and_store(
            'test',
            self.input()['test'],
            self.output()['test'],
            image_size=image_size,
            preprocessing=preprocessing,
            model=model,
        )
        print('processed test:', (time() - start_time) / 1000, 's')


if __name__ == '__main__':
    luigi.run()
