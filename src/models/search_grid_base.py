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


class SearchGridBase(MLFlowTask):
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
            'experiment': luigi.LocalTarget(
                path.join(output_dir, 'experiment.pickle'),
                format=luigi.format.Nop
            )
        }

    def _get_experiment(self):
        ax_experiment_file = self.output()['experiment']
        if not ax_experiment_file.exists():
            return None

        with ax_experiment_file.open('r') as f:
            return pickle.load(f)

    def get_params_space(self):
        """
        could be overwritten
        :return:
        """
        return {
            'train_size': np.linspace(
                1.0 / self.max_runs, 1.0, self.max_runs, endpoint=False
            ),
        }

    def ml_run(self, run_id=None):
        mlflow.log_params(flatten(get_params_of_task(self)))

        search_state = self._get_experiment()
        if search_state is None:
            search_state = GridSearchState(self.metric,
                                           params_space=flatten(self.get_params_space()))

        model_task = get_model_task_by_name(self.model_name)

        total_training_time = 0

        for idx, params in enumerate(search_state):
            # preserve search state
            with self.output()['experiment'].open('w') as f:
                pickle.dump(search_state, f)

            model_result = yield model_task(
                parent_run_id=run_id,
                random_seed=self.random_seed,
                **params
            )

            model_run_id = self.get_run_id_from_result(model_result)

            with model_result['metrics'].open('r') as f:
                model_metrics = yaml.load(f)
                model_score_mean = model_metrics[self.metric]['val']
                total_training_time += model_metrics['train_time']['total']

            with model_result['params'].open('r') as f:
                model_params = yaml.load(f)

            mlflow.log_metric(
                'train_time.epoch',
                model_metrics['train_time']['epoch'],
                step=idx
            )

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
            yaml.dump(best_trial.metrics, f, default_flow_style=False)

        with self.output()['params'].open('w') as f:
            yaml.dump(best_trial.params, f, default_flow_style=False)


class Trial:
    def __init__(self, metrics, params, score, run_id):
        self.metrics = metrics
        self.params = params
        self.run_id = run_id
        self.score = score


class GridSearchState:
    def __init__(self, metric, params_space):
        self._metric = metric
        self._best_trial = None
        self._params_space = params_space
        params_grid = np.meshgrid(*params_space.values(), copy=False)
        # we should use dtype=object because we mix different param types(int, float, string)
        self._idx = -1
        self._params_sequence = np.array([p.ravel() for p in params_grid], dtype=object).T
        self._last_idx = len(self._params_sequence) - 1

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= self._last_idx:
            raise StopIteration

        self._idx += 1
        return unflatten(dict(zip(
            self._params_space.keys(),
            self._params_sequence[self._idx]
        )))

    def complete_trial(self, score, metrics, params, run_id):
        if self._best_trial is None or \
                is_better_score(self._metric, self._best_trial.score, score):
            self._best_trial = Trial(
                metrics=metrics,
                params=params,
                score=score,
                run_id=run_id,
            )

    def get_best_trial(self):
        return self._best_trial


if __name__ == '__main__':
    luigi.run()
