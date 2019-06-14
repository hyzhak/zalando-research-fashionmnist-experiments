import luigi
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import time

from ..data.external_train_set import ExternalTrainSet
from ..utils.params_to_filename import params_to_filename


class TrainBaselineLogisticRegression(luigi.Task):
    solver = luigi.Parameter(default='lbfgs')
    # for small datasets 'liblinear'
    # 'sag' and 'saga' for large
    # 'newton-cg'

    multi_class = luigi.Parameter(default='multinomial')
    random_seed = luigi.IntParameter(default=12345)
    n_jobs = luigi.IntParameter(default=-1)
    max_iter = luigi.IntParameter(default=100)

    # model_file = luigi.Parameter(default='model.pkl')

    def output(self):
        filename = params_to_filename({
            'multi_class': self.multi_class,
            'random_seed': self.random_seed,
            'max_iter': self.max_iter
        })
        return luigi.LocalTarget(
            f'models/baseline/logistic-regression/{filename}.pkl',
            format=luigi.format.Nop
        )

    def requires(self):
        return ExternalTrainSet()

    def run(self):
        X_train, y_train = self._extract_x_and_y(self.input())

        start = time.time()
        clf = LogisticRegression(solver=self.solver,
                                 multi_class=self.multi_class,
                                 random_state=self.random_seed,
                                 n_jobs=self.n_jobs,
                                 max_iter=self.max_iter)
        clf.fit(X_train, y_train)
        fitting_time = time.time() - start
        print('fitting time', fitting_time)
        with self.output().open('w') as f:
            pickle.dump(clf, f)

    def _extract_x_and_y(self, input_file):
        with input_file.open('r') as f:
            df = pd.read_csv(f, dtype=np.int16)

        pixel_features = df.columns[df.columns.str.contains('pixel')]
        return df[pixel_features], df['label']


if __name__ == '__main__':
    luigi.run()
