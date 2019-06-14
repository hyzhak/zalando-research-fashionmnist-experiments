import luigi
import numpy as np
import pandas as pd

from ..data.external_test_set import ExternalTestSet
from ..data.external_train_set import ExternalTrainSet


class ExtractFlatSet(luigi.Task):
    is_train_set = luigi.BoolParameter()

    def output(self):
        postfix = 'train' if self.is_train_set else 'test'
        return f''

    def requires(self):
        if self.is_train_set:
            return ExternalTrainSet()
        else:
            return ExternalTestSet()

    def run(self):
        with self.input().open('r') as f:
            df = pd.read_csv(f, dtype=np.int16)

        pixel_features = df.columns[df.columns.str.contains('pixel')]
        self.output()
        return df[pixel_features], df['label']


if __name__ == '__main__':
    luigi.run()
