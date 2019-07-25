import luigi

import os
from PIL import Image
from time import time
import yaml

from src.data.external_label_titles import ExternalLabelTitles
from src.data.external_test_set import ExternalTestSet
from src.data.external_train_set import ExternalTrainSet
from src.utils.extract_x_y import extract_x_and_y, reshape_X_to_2d, get_images
from src.utils.params_to_filename import encode_task_to_filename
from src.utils.snake import get_class_name_as_snake


class ImageFiles(luigi.Task):
    def output(self):
        class_name = get_class_name_as_snake(self)
        encoded_params = encode_task_to_filename(self)
        return {
            'train': luigi.LocalTarget(
                os.path.join('data', 'interim', class_name, encoded_params, 'train'),
                format=luigi.format.Nop
            ),
            'test': luigi.LocalTarget(
                os.path.join('data', 'interim', class_name, encoded_params, 'test'),
                format=luigi.format.Nop
            )
        }

    def requires(self):
        return {
            'label_titles': ExternalLabelTitles(),
            'test': ExternalTestSet(),
            'train': ExternalTrainSet(),
        }

    def run(self):
        with self.input()['label_titles'].open('r') as f:
            label_titles = yaml.load(f)
        self._save_as_images(self.input()['train'], 'train', label_titles)
        self._save_as_images(self.input()['test'], 'test', label_titles)

    def _save_as_images(self, input_file, subset, label_titles):
        start = time()
        output = self.output()[subset]
        X, y = extract_x_and_y(input_file)
        images = reshape_X_to_2d(X, -1)
        for idx, (image, y) in enumerate(zip(images, y)):
            im = Image.fromarray(
                (255 - image).astype('uint8'),
                mode='L'
            )
            dir = os.path.join(output.path, label_titles[y])
            os.makedirs(dir, exist_ok=True)
            im.save(os.path.join(dir, f'{idx}.png'))

        end = time()
        print('elapsed:', end - start, 'sec')


if __name__ == '__main__':
    luigi.run()
