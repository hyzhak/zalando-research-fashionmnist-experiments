import luigi
import mlflow
import numpy as np
from os import path
import pickle
import yaml

from src.models.get_model_task_by_name import get_model_task_by_name
from src.utils.flatten import flatten, unflatten
from src.utils.metrics import is_better_score, should_minimize
from src.utils.mlflow_task import MLFlowTask
from src.utils.params_to_filename import get_params_of_task


class SearchGrid(MLFlowTask):
    model_name = luigi.Parameter(
        default='simple_cnn',
        description='model name (e.g. logistic_regression, simple_cnn)'
    )
    metric = luigi.Parameter(
        default='accuracy',
        description='metric to optimize on accuracy or loss'
    )
    random_seed = luigi.IntParameter(
        default=12345,
        description='seed for the random generator'
    )

    def ml_output(self, output_dir):
        return {
            'metrics': luigi.LocalTarget(
                path.join(output_dir, 'metrics.yml')
            ),
            'params': luigi.LocalTarget(
                path.join(output_dir, 'params.yml')
            ),
            'grid_experiment': luigi.LocalTarget(
                path.join(output_dir, 'grid_experiment.pickle'),
                format=luigi.format.Nop
            )
        }

    def ml_run(self, run_id=None):
        mlflow.log_params(flatten(get_params_of_task(self)))

        params_space = flatten({
            'train_size': np.linspace(1.0 / 10, 1.0, 10),
            # 'batch_size': np.linspace(1, 16, 3, dtype=int),
        })
        params_grid = np.meshgrid(*params_space.values(), copy=False)
        # we should use dtype=object because we mix different param types(int, float, string)
        params_sequence = np.array([p.ravel() for p in params_grid], dtype=object).T

        model_task = get_model_task_by_name(self.model_name)

        total_training_time = 0

        search_state = GridSearchState(self.metric)

        # TODO: could store result and current trial
        for param_values in params_sequence:
            params = unflatten(dict(zip(params_space.keys(), param_values)))
            model_result = yield model_task(
                parent_run_id=run_id,
                random_seed=self.random_seed,
                # TODO: actually we should be able to pass even nested params
                **params
                # **parameters,
                # optimizer_props=param_values
            )

            # TODO: store run_id in Trial
            model_run_id = self.get_run_id_from_result(model_result)

            with model_result['metrics'].open('r') as f:
                model_metrics = yaml.load(f)
                model_score_mean = model_metrics[self.metric]['val']
                total_training_time += model_metrics['train_time']['total']

            with model_result['params'].open('r') as f:
                model_params = yaml.load(f)

            mlflow.log_metric('train_time.epoch', search_state.get_last_epoch_time())

            search_state.complete_trial(
                score=model_score_mean,
                metrics=model_metrics,
                params=model_params,
                run_id=model_run_id
            )

        mlflow.log_metric('train_time.total', total_training_time)

        best_trial = search_state.get_best_trial()
        mlflow.log_params(flatten(best_trial.params))
        mlflow.log_metrics(flatten(best_trial.metrics))

        with self.output()['metrics'].open('w') as f:
            yaml.dump(best_trial.metrics, f)

        with self.output()['params'].open('w') as f:
            yaml.dump(best_trial.params, f)


# TODO: make GridSearch iterator
# which will store current and best trial
class Trial:
    def __init__(self, metrics, params, score, run_id):
        self.metrics = metrics
        self.params = params
        self.run_id = run_id
        self.score = score


class GridSearchState:
    def __init__(self, metric):
        self._metric = metric
        self._best_trial = None

    def complete_trial(self, score, metrics, params, run_id):
        if self._best_trial is None or \
                is_better_score(self._metric, self._best_trial.score, score):
            self._best_trial = Trial(
                metrics=metrics,
                params=params,
                score=score,
                run_id=run_id,
            )

    def get_last_epoch_time(self):
        # TODO:
        return 0.0

    def get_best_trial(self):
        return self._best_trial


if __name__ == '__main__':
    luigi.run()
