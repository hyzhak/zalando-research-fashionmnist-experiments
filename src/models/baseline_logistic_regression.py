import luigi
import mlflow
import mlflow.sklearn
import pickle
from sklearn.linear_model import LogisticRegression
import time

from src.data.external_train_set import ExternalTrainSet
from src.utils.params_to_filename import encode_task_to_filename
from src.utils.extract_x_y import extract_x_and_y


class TrainBaselineLogisticRegression(luigi.Task):
    model_name = 'logistic_regression'

    solver = luigi.Parameter(
        default='lbfgs',
        description='Algorithm to use in the optimization problem'
    )
    # for small datasets 'liblinear'
    # 'sag' and 'saga' for large
    # 'newton-cg'

    multi_class = luigi.Parameter(default='multinomial')
    C = luigi.FloatParameter(
        default=1.0,
        description='Inverse of regularization strength; '
                    'must be a positive float. '
                    'Like in support vector machines, smaller values specify stronger regularization.'
    )
    random_seed = luigi.IntParameter(default=12345)
    n_jobs = luigi.IntParameter(
        default=-1,
        significant=False
    )
    max_iter = luigi.IntParameter(default=100)

    # model_file = luigi.Parameter(default='model.pkl')

    def output(self):
        filename = encode_task_to_filename(self)
        return luigi.LocalTarget(
            f'models/baseline/{self.model_name}/{filename}.pkl',
            format=luigi.format.Nop
        )

    def requires(self):
        return ExternalTrainSet()

    def run(self):
        X_train, y_train = extract_x_and_y(self.input())

        start = time.time()
        clf = LogisticRegression(solver=self.solver,
                                 multi_class=self.multi_class,
                                 random_state=self.random_seed,
                                 n_jobs=self.n_jobs,
                                 max_iter=self.max_iter)
        clf.fit(X_train, y_train)
        training_time = time.time() - start

        mlflow.sklearn.log_model(clf, 'model')
        with self.output().open('w') as f:
            pickle.dump(clf, f)
        mlflow.log_param('model_name', self.model_name)
        mlflow.log_param('solver', self.solver)
        mlflow.log_param('multi_class', self.multi_class)
        mlflow.log_param('random_seed', self.random_seed)
        mlflow.log_param('max_iter', self.max_iter)
        mlflow.log_metric('training_time', training_time)


if __name__ == '__main__':
    with mlflow.start_run():
        luigi.run()
