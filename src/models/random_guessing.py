import luigi
import mlflow
import numpy as np
import os
import yaml

from src.data.external_test_set import ExternalTestSet
from src.data.external_train_set import ExternalTrainSet
from src.utils.extract_x_y import extract_x_and_y
from src.utils.flatten import flatten
from src.utils.mlflow_task import MLFlowTask


class RandomGuessing(MLFlowTask):
    """
    Baseline model which could help to monitor that our model is better
    than just random guessing

    https://machinelearningmastery.com/dont-use-random-guessing-as-your-baseline-classifier/
    random guessing baseline isn't good enough, when classes in-balanced.

    So we may use something more opinionated like ZeroRule,
    which simple predict mode of dataset

    """
    random_seed = luigi.IntParameter(
        default=12345
    )

    def requires(self):
        return {
            'test': ExternalTestSet(),
            'train': ExternalTrainSet(),
        }

    def ml_output(self, output_dir):
        return {
            'metrics': luigi.LocalTarget(
                os.path.join(output_dir, 'metrics.yml')
            ),
        }

    def _get_true_class_p(self, y):
        classes_values, classes_count = np.unique(y, return_counts=True)
        classes_feq = classes_count / y.shape[0]

        # we have even probability for all classes
        one_class_p = 1 / len(classes_values)

        # so here is probability to choose the same class is it was picked from a dataset
        return (classes_feq * one_class_p).sum()

        # we actually could use simple form:
        # just sample random values and calc accuracy
        # and btw we could use any metrics this way
        #
        # from sklearn import metrics
        #
        # estimate_p = metrics.accuracy_score(
        #     np.random.randint(0, 10, y.shape[0]),
        #     y
        # )

    def ml_run(self, run_id=None):
        _, y_train = extract_x_and_y(self.input()['train'])
        _, y_test = extract_x_and_y(self.input()['test'])

        # so here is probability to choose the same class is it was picked from a dataset
        true_class_p_for_train = self._get_true_class_p(y_train)
        true_class_p_for_test = self._get_true_class_p(y_test)

        # TODO: we may want to add other metrics (like confusion metrics)

        metrics = {
            'accuracy': {
                'train': true_class_p_for_train,
                'test': true_class_p_for_test,
            },
            # I think we can get loss only for specific model
            # 'loss'
        }
        with self.output()['metrics'].open('w') as f:
            yaml.dump(metrics, f, default_flow_style=False)

        mlflow.log_metrics(flatten(metrics))


if __name__ == '__main__':
    luigi.run()
