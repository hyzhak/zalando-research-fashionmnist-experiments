import luigi
import pickle
import mlflow
from sklearn import metrics
import yaml

from src.data.external_test_set import ExternalTestSet
from src.data.external_train_set import ExternalTrainSet
from src.models.baseline_logistic_regression import TrainBaselineLogisticRegression
from src.utils.params_to_filename import params_to_filename
from src.utils.extract_x_y import extract_x_and_y


class ScikitScore(luigi.Task):
    model_name = luigi.Parameter(
        default='logistic-regression'
    )
    model_params = luigi.DictParameter(
        default={},
        description='model params'
    )

    def output(self):
        filename = params_to_filename({
            'model': self.model_name,
            **self.model_params
        })
        return luigi.LocalTarget(f'reports/scores/{filename}.yaml')

    def requires(self):
        return (
            TrainBaselineLogisticRegression(**self.model_params),
            ExternalTestSet(),
            ExternalTrainSet(),
        )

    def _score(self, y_true, y_pred):
        return {
            'accuracy': float(metrics.accuracy_score(y_true, y_pred)),
            'cohen_kappa': float(metrics.cohen_kappa_score(y_true, y_pred)),
            'f1': float(metrics.f1_score(y_true, y_pred, average='micro')),
            'precision': float(metrics.precision_score(y_true, y_pred, average='micro')),
            'recall': float(metrics.recall_score(y_true, y_pred, average='micro')),
        }

    def run(self):
        model = pickle.load(self.input()[0].open('r'))
        X_test, y_test = extract_x_and_y(self.input()[1])
        X_train, y_train = extract_x_and_y(self.input()[2])

        scores = {
            'test': self._score(y_test, model.predict(X_test)),
            'train': self._score(y_train, model.predict(X_train)),
        }

        with self.output().open('w') as f:
            yaml.dump(scores, f, default_flow_style=False)

        mlflow.log_param('model_name', self.model_name)
        for (score_name_train, score_value_train), (score_name_test, score_value_test) \
                in zip(scores['train'].items(), scores['test'].items()):
            # Names may only contain alphanumerics, underscores (_),
            #  dashes (-), periods (.), spaces ( ), and slashes (/).
            mlflow.log_metric(f'{score_name_train} train', score_value_train)
            mlflow.log_metric(f'{score_name_test} test', score_value_test)

        print(f'Model saved in run {mlflow.active_run().info.run_uuid}')


if __name__ == '__main__':
    with mlflow.start_run():
        luigi.run()
