import luigi
from keras.preprocessing.image import img_to_array, array_to_img
import numpy as np
import os
import pandas as pd
from tensorflow.keras import applications
from time import time

from src.data.external_test_set import ExternalTestSet
from src.data.external_train_set import ExternalTrainSet
from src.utils.extract_x_y import get_images, get_train_valid_test_subsets, \
    reshape_X_to_2d, img_rows, img_cols
from src.utils.params_to_filename import encode_task_to_filename
from src.utils.snake import get_class_name_as_snake


class LatentSpaceFeature(luigi.Task):
    model = luigi.Parameter(
        default='vgg16'
    )

    def requires(self):
        return {
            'test': ExternalTestSet(),
            'train': ExternalTrainSet(),
        }

    def output(self):
        class_name = get_class_name_as_snake(self)
        encoded_params = encode_task_to_filename(self)
        return {
            'test': luigi.LocalTarget(
                os.path.join('data', 'interim', class_name, encoded_params,
                             'test.parquet.gzip')
            ),
            'train': luigi.LocalTarget(
                os.path.join('data', 'interim', class_name, encoded_params,
                             'train.parquet.gzip')
            ),
        }

    def _process_and_store(self, input_file, output_file, size,
                           preprocessing, model):
        # we should have 3 channels
        images = get_images(input_file, 3)
        print('images.shape:', images.shape)

        # Upscale the images to 48*48 as required by VGG16 (minimum 32)
        images = np.asarray([
            img_to_array(array_to_img(i, scale=False).resize((size, size)))
            for i in images]
        )
        print('images.shape (after resize):', images.shape)

        images = preprocessing(images)
        print('images.shape (after pre-processing):', images.shape)

        features = model.predict(images, verbose=1)
        print('features.shape:', features.shape)

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
        print('save to parquet: ', time() - start_save)

    def run(self):
        image_size = 48
        model = None
        preprocessing = None
        if self.model == 'vgg16':
            model = applications.vgg16.VGG16(include_top=False,
                                             input_shape=(image_size, image_size, 3),
                                             weights='imagenet')
            preprocessing = applications.vgg16.preprocess_input
        else:
            # TODO: add other model
            raise NotImplementedError()

        model.summary()
        start_time = time()
        self._process_and_store(
            self.input()['train'],
            self.output()['train'],
            size=image_size,
            preprocessing=preprocessing,
            model=model,
        )
        print('processed train:', (time() - start_time) / 1000, 's')

        start_time = time()
        self._process_and_store(
            self.input()['test'],
            self.output()['test'],
            size=image_size,
            preprocessing=preprocessing,
            model=model,
        )
        print('processed test:', (time() - start_time) / 1000, 's')


if __name__ == '__main__':
    luigi.run()
