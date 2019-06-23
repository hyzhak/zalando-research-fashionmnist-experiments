import luigi
from hyperopt import fmin, hp, tpe, rand
import json
import numpy as np
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
        print(f'FILENAME: reports/optimizers/{optimizer_name}/{hyper_opt_runs_filename}.pickle')
        return {
            'hyper_opt_runs': luigi.LocalTarget(
                f'reports/optimizers/{optimizer_name}/{hyper_opt_runs_filename}.pickle',
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
        params_key = get_key_by_params(x)
        hyper_opt_runs = self._get_hyper_opt_runs()

        if hyper_opt_runs and params_key in hyper_opt_runs:
            print('we have right value! ', hyper_opt_runs[params_key])
            # we can't use params as they are because they can be not hashable
            return hyper_opt_runs[params_key]

        print('it seems we got a new value, lets go for a search', x)
        raise NewValueForOptimizer(x)

    def ml_run(self, run_id=None):
        print('# ml_run', run_id)
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

                with model_result['metrics'].open('r') as f:
                    res = yaml.load(f)
                    loss = res[self.metric]['val']
                    if self.metric == 'accuracy':
                        loss = -loss
                    print('now time to store new loss:', loss)
                    hyper_opt_runs = self._get_hyper_opt_runs() or {}
                    hyper_opt_runs[get_key_by_params(params)] = loss
                    with self.output()['hyper_opt_runs'].open('w') as f:
                        pickle.dump(hyper_opt_runs, f)

        print('TODO: store best result :', best)

        # with self.output()['metrics'].open('w') as f:
        #     yaml.dump()

        #
        # # TODO: simple duplicate search_random (could I reuse code?)
        # best_run = None
        # best_val_train = -math.inf
        # best_val_valid = -math.inf
        # best_val_test = -math.inf
        #
        # for model_output in tasks:
        #     # TODO: get the score and compare with the best
        #     with model_output['metrics'].open('r') as f:
        #         res = yaml.load(f)
        #         # TODO: we don't have yet validation set (should add)
        #         new_val_valid = res[self.metric]['val']
        #
        #         # TODO: in case of accuracy it is "<"
        #         # in case of loss it should be ">"
        #         if best_val_valid < new_val_valid:
        #             best_run = res.get('run_id', None)
        #             best_val_train = res[self.metric]['train']
        #             best_val_valid = new_val_valid
        #             best_val_test = res[self.metric]['test']
        #
        # metrics = {
        #     f'train_{self.metric}': float(best_val_train),
        #     f'val_{self.metric}': float(best_val_valid),
        #     f'test_{self.metric}': float(best_val_test),
        # }
        # mlflow.set_tag('best_run', best_run)
        # mlflow.log_metrics(metrics)
        # # TODO: maybe it also make sense to duplicate the best params in experiment?
        #
        # with self.output()['metrics'].open('w') as f:
        #     yaml.dump({
        #         'metrics': metrics,
        #         'best_run_id': best_run,
        #     }, f, default_flow_style=False)


class NewValueForOptimizer(Exception):
    def __init__(self, new_value):
        self.new_value = new_value


def get_key_by_params(p):
    return json.dumps(p)


if __name__ == '__main__':
    luigi.run()
