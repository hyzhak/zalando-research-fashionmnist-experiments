import luigi
import os
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

from src.features.latent_space_features import LatentSpaceFeature
from src.utils.params_to_filename import get_task_path


# should I use @inherits instead?
# https://luigi.readthedocs.io/en/stable/api/luigi.util.html
class PCALatentSpaceFeature(luigi.Task):
    model = luigi.Parameter(
        default='vgg16'
    )

    random_seed = luigi.IntParameter(
        default=12345
    )

    def requires(self):
        return LatentSpaceFeature(model=self.model)

    def output(self):
        task_dir = os.path.join(
            'data', 'interim', get_task_path(self)
        )

        return {
            'test': {
                'features': luigi.LocalTarget(
                    os.path.join(task_dir, 'test.features.parquet.gzip')
                ),
                'explained_variance_ratio': luigi.LocalTarget(
                    os.path.join(task_dir, 'test.evr.npy'),
                    format=luigi.format.Nop
                ),
            },
            'train': {
                'features': luigi.LocalTarget(
                    os.path.join(task_dir, 'train.features.parquet.gzip')
                ),
                'explained_variance_ratio': luigi.LocalTarget(
                    os.path.join(task_dir, 'train.evr.npy'),
                    format=luigi.format.Nop
                ),
            },
        }

    def run(self):
        # TODO: could be done in parallel
        self._process(self.input()['train'], self.output()['train'])
        self._process(self.input()['test'], self.output()['test'])

    def _process(self, input_file, output_file):
        X = pd.read_parquet(input_file.open('r'))

        pca = PCA(random_state=self.random_seed)
        pca.fit(X)

        np.save(
            output_file['explained_variance_ratio'].open('w'),
            pca.explained_variance_ratio_.cumsum()
        )

        pca = PCA(random_state=self.random_seed, n_components=2)
        embedded = pca.fit_transform(X)

        df = pd.DataFrame(embedded,
                          columns=['x1', 'x2'],
                          dtype=np.float32)

        output_file['features'].makedirs()
        df.to_parquet(output_file['features'].path,
                      compression='brotli')


if __name__ == '__main__':
    luigi.run()
