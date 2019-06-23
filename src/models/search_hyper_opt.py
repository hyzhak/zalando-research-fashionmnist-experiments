import luigi
from hyperopt import fmin, hp, tpe, rand
import json
import numpy as np
import mlflow
import pickle
import yaml

from src.models import get_model_task_by_name
from src.utils.mlflow_task import MLFlowTask
from src.utils.params_to_filename import encode_task_to_filename
from src.utils.seed_randomness import seed_randomness


class SearchHyperOpt(MLFlowTask):
    model_name = luigi.Parameter(
        default='simple_cnn',
        description='model name (e.g. logistic_regression, simple_cnn)'
    )
    algo = luigi.Parameter(
        default='tpe.suggest',
        description='Optimizer algorithm.'
    )
    metric = luigi.Parameter(
        default='accuracy',
        description='metric to optimize on accuracy or loss'
    )
    max_runs = luigi.IntParameter(
        default=10,
        description='maximum number of runs to evaluate'
    )
    random_seed = luigi.IntParameter(
        default=12345,
        description='seed for the random generator'
    )

    def ml_output(self):
        optimizer_name = 'hyper_opt'
        filename = encode_task_to_filename(self, ['model_name'])
        hyper_opt_runs_filename = encode_task_to_filename(self)
        return {
            'hyper_opt_runs': luigi.LocalTarget(
                f'reports/optimizers/{optimizer_name}/{hyper_opt_runs_filename}__runs.pickle',
                format=luigi.format.Nop
            ),
            'metrics': luigi.LocalTarget(
                f'reports/metrics/{self.model_name}/best_search_hyperopt__{filename}.yaml'
            )
        }

    def _get_hyper_opt_runs(self):
        output_hyper_opt_runs = self.output()['hyper_opt_runs']
        if output_hyper_opt_runs.exists():
            with output_hyper_opt_runs.open('r') as f:
                return pickle.load(f)
        return None

    def _fn(self, x):
        print('# _fn', x)
        # we can't use params as they are because they can be not hashable
        params_key = get_key_by_params(x)
        hyper_opt_runs = self._get_hyper_opt_runs()

        if hyper_opt_runs and params_key in hyper_opt_runs:
            metrics = hyper_opt_runs[params_key]['metrics']
            loss = metrics[self.metric]['val']
            # in case of accuracy we would like maximize function :)
            if self.metric == 'accuracy':
                loss = -loss
            print('we have right value! ', loss)
            return loss

        print('it seems we got a new value, lets go for a search', x)
        raise NewValueForOptimizer(x)

    def ml_run(self, run_id=None):
        seed_randomness(self.random_seed)

        params_space = {
            'optimizer_props': {
                'lr': hp.uniform('lr', 0.001, 0.00001),
                'beta_1': hp.uniform('beta_1', .0, 0.9999),
            }
        }

        # Solution

        # 1) pass custom method
        # and if we haven't calculated loss, break hyperopt

        # 2) once we got outside we try to yield last unsolved task

        # 3) because luigi would run again current task.

        # TODO: to solve this problem I would need to pickle inner state of Trail of hyperopt
        # https://github.com/hyperopt/hyperopt/wiki/FMin#13-the-trials-object

        # TODO: how to leverage parallelism?
        # https://github.com/hyperopt/hyperopt/wiki/Parallelizing-Evaluations-During-Search-via-MongoDB

        best = None
        while not best:
            try:
                best = fmin(fn=self._fn,
                            space=params_space,
                            algo=tpe.suggest if self.algo == "tpe.suggest" else rand.suggest,
                            max_evals=self.max_runs,
                            rstate=np.random.RandomState(self.random_seed),
                            show_progressbar=False,
                            )
            except NewValueForOptimizer as e:
                # TODO: maybe we can run multiple runs in parallel?
                params = e.new_value
                model_task = get_model_task_by_name(self.model_name)
                model_result = yield model_task(
                    parent_run_id=run_id,
                    random_seed=self.random_seed,
                    **params,
                )

                model_run_id = None
                if 'ml_flow' in model_result:
                    with model_result['ml_flow'].open('r') as f:
                        model_run_id = yaml.load(f).get('run_id')

                with model_result['metrics'].open('r') as f:
                    model_metrics = yaml.load(f)
                    hyper_opt_runs = self._get_hyper_opt_runs() or {}
                    hyper_opt_runs[get_key_by_params(params)] = {
                        'metrics': model_metrics,
                        'run_id': model_run_id,
                    }
                    with self.output()['hyper_opt_runs'].open('w') as f:
                        pickle.dump(hyper_opt_runs, f)

        print('we got the best params:', best)

        hyper_opt_runs = self._get_hyper_opt_runs() or {}

        # TODO: hyperopts drops 'optimizer_props' for some reasons
        # need to protect it
        best_model_state = hyper_opt_runs.get(get_key_by_params({
            'optimizer_props': best
        }))

        if best_model_state is None:
            raise Exception('it seems we do not have any runs here')

        # TODO: format of the best model metrics should look the same
        # as metrics of experiments
        metrics = {
            f'train_{self.metric}': float(best_model_state['metrics'][self.metric]['train']),
            f'val_{self.metric}': float(best_model_state['metrics'][self.metric]['val']),
            f'test_{self.metric}': float(best_model_state['metrics'][self.metric]['test']),
        }

        mlflow.log_metrics(metrics)

        # child task could be mlflow_task so we can get run_id from its 'mlflow' state
        best_run = best_model_state.get('run_id', None)
        if best_run:
            mlflow.set_tag('best_run', best_run)

        with self.output()['metrics'].open('w') as f:
            yaml.dump(best_model_state, f, default_flow_style=False)


class NewValueForOptimizer(Exception):
    def __init__(self, new_value):
        self.new_value = new_value


def get_key_by_params(p):
    return json.dumps(p)


if __name__ == '__main__':
    luigi.run()
