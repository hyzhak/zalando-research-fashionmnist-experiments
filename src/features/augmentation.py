from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import luigi
import os

from src.data.external_test_set import ExternalTestSet
from src.data.external_train_set import ExternalTrainSet
from src.utils.extract_x_y import get_train_valid_test_subsets, reshape_X_to_2d
from src.utils.params_to_filename import encode_task_to_filename
from src.utils.snake import get_class_name_as_snake


def safe_param(f, apply=float, default=None):
    return apply(f) if f is not None else default


def safe_list(l, apply):
    return [apply(v) for v in l.split(',')] if l is not None else None


class Augmentation(luigi.Task):
    valid_size = luigi.FloatParameter(
        default=0.1
    )
    train_size = luigi.OptionalParameter(
        default=None
    )
    random_seed = luigi.IntParameter(
        default=1234
    )
    batch_size = luigi.OptionalParameter(
        default=None
    )

    zca = luigi.OptionalParameter(
        default=None,
        description='epsilon for ZCA whitening. (e.g. 1e-6.), disable by default'
    )
    rotation_range = luigi.IntParameter(
        default=0,
        description='(int) Degree range for random rotations (e.g. 40).'
    )
    width_shift_range = luigi.FloatParameter(
        default=0.0,
        description='e.g. 0.2'
    )
    height_shift_range = luigi.FloatParameter(
        default=0.0,
        description='e.g. 0.2'
    )
    brightness_range = luigi.OptionalParameter(
        default=None,
        description='Two floats. Range for picking a brightness shift value from.'
    )
    shear_range = luigi.FloatParameter(
        default=0.0,
        description='Shear angle in counter-clockwise direction in degrees (e.g. 0.2)'
    )
    zoom_range = luigi.FloatParameter(
        default=0.0,
        description='e.g. 0.2'
    )
    channel_shift_range = luigi.OptionalParameter(
        default=None,
        description='Range for random channel shifts.'
    )
    horizontal_flip = luigi.BoolParameter(
        default=False,
        description='Randomly flip inputs horizontally'
    )
    vertical_flip = luigi.BoolParameter(
        default=False,
        description='Randomly flip inputs vertically'
    )
    # TODO: should have cval when "constant
    fill_mode = luigi.Parameter(
        default='constant',
        description='One of {"constant", "nearest", "reflect" or "wrap"}'
    )

    def output(self):
        class_name = get_class_name_as_snake(self)
        encoded_params = encode_task_to_filename(self)
        return luigi.LocalTarget(
            os.path.join('data', 'interim', class_name, encoded_params)
        )

    def requires(self):
        return {
            'test': ExternalTestSet(),
            'train': ExternalTrainSet(),
        }

    def run(self):
        # self.brightness_range
        # self.channel_shift_range
        datagen = ImageDataGenerator(
            # because train_images.shape: (54000, 28, 28, 1)
            data_format='channels_last',
            zca_whitening=self.zca is not None,
            zca_epsilon=safe_param(self.zca, float),
            rotation_range=self.rotation_range,
            width_shift_range=self.width_shift_range,
            height_shift_range=self.height_shift_range,
            brightness_range=safe_list(self.brightness_range, float),
            shear_range=self.shear_range,
            zoom_range=self.zoom_range,
            horizontal_flip=self.horizontal_flip,
            vertical_flip=self.vertical_flip,
            fill_mode=self.fill_mode,
            cval=0.0)

        X_train, _, _, _, _, _ = get_train_valid_test_subsets(
            self.train_size,
            self.valid_size,
            self.random_seed,
            self.input()['train'],
            self.input()['test']
        )
        train_images = reshape_X_to_2d(X_train)

        # img = load_img('data/train/cats/cat.0.jpg')  # this is a PIL image
        # x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
        # x = train_images.reshape((1,) + train_images.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
        # print('x.shape:', x.shape)

        # the .flow() command below generates batches of randomly transformed images
        # and saves the results to the `preview/` directory
        output_target = self.output()
        os.makedirs(output_target.path)

        if self.zca is not None:
            datagen.fit(train_images,
                        seed=self.random_seed)

        i = 0
        limit = safe_param(self.batch_size, int, len(train_images))

        for _ in datagen.flow(train_images,
                              batch_size=1,
                              save_to_dir=output_target.path,
                              save_prefix='img',
                              save_format='jpeg',
                              seed=self.random_seed):
            self.set_status_message(f'Progress: {i} / 100')
            self.set_progress_percentage(i / 20)
            i += 1
            if i > limit:
                break  # otherwise the generator would loop indefinitely


if __name__ == '__main__':
    luigi.run()
