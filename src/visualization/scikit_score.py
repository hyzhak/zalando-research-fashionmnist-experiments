import luigi
import pickle
import yaml
from ..data.external_test_set import ExternalTestSet
from ..data.external_train_set import ExternalTrainSet
from ..models.baseline_logistic_regression import TrainBaselineLogisticRegression
from ..utils.params_to_filename import params_to_filename
from ..utils.extract_x_y import extract_x_and_y


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

    def run(self):
        model = pickle.load(self.input()[0].open('r'))
        X_test, y_test = extract_x_and_y(self.input()[1])
        X_train, y_train = extract_x_and_y(self.input()[2])

        scores = {
            'accuracy': {
                'test': float(model.score(X_test, y_test)),
                'train': float(model.score(X_train, y_train)),
            }
        }

        with self.output().open('w') as f:
            yaml.dump(scores, f, default_flow_style=False)


if __name__ == '__main__':
    luigi.run()
