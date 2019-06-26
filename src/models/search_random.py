import luigi
import math
import mlflow
import numpy as np
import random as rn
import yaml

from src.models.get_model_task_by_name import get_model_task_by_name
from src.utils.params_to_filename import encode_task_to_filename
from src.utils.seed_randomness import seed_randomness


class SearchRandom(luigi.Task):
    model_name = luigi.Parameter(
        default='simple_cnn',
        description='model name (e.g. logistic_regression, simple_cnn)'
    )
    metric = luigi.Parameter(
        default='accuracy',
        description='metric to optimize on accuracy or loss'
    )
    experiment = luigi.Parameter(
        default=None,
        description='ml experiment name',
        significant=False,
    )
    max_runs = luigi.IntParameter(
        default=10,
        description='maximum number of runs to evaluate'
    )
    random_seed = luigi.IntParameter(
        default=12345,
        description='seed for the random generator'
    )

    def output(self):
        # TODO: Do I really need to store best solution?
        filename = encode_task_to_filename(self, ['model_name'])
        return {
            'mlflow': luigi.LocalTarget(
                f'reports/metrics/{self.model_name}/best_search_random__{filename}_mlflow.yaml'
            ),
            'metrics': luigi.LocalTarget(
                f'reports/metrics/{self.model_name}/best_search_random__{filename}.yaml'
            )
        }

    def run(self):
        seed_randomness(self.random_seed)

        # TODO: should put it to the params of task
        params_space = [{
            'optimizer': rn.choice(['adam', 'sgd']),
            'optimizer_props': {'lr': pow(10, np.random.uniform(-4, -1))},
            # TODO: because optimizer have different signatures maybe I should use
            # https://github.com/hyperopt/hyperopt/wiki/FMin#2-defining-a-search-space
            # to have different space to sample for different optimizers
            # keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            # keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
        } for _ in range(self.max_runs)]

        model_task = get_model_task_by_name(self.model_name)

        mlflow_output = self.output()['mlflow']
        parent_run_id = None
        if mlflow_output.exists():
            with mlflow_output.open('r') as f:
                parent_run_id = yaml.load(f).get('run_id')
                print('continue mlflow run:', parent_run_id)
        elif self.experiment:
            mlflow.set_experiment(self.experiment)

        with mlflow.start_run(run_id=parent_run_id) as run:
            print('run.info', run.info)
            if parent_run_id is None:
                with mlflow_output.open('w') as f:
                    yaml.dump({
                        'run_id': run.info.run_id
                    }, f, default_flow_style=False)
                parent_run_id = run.info.run_id

            # run all random tasks in parallel
            tasks = yield [model_task(
                parent_run_id=parent_run_id,
                random_seed=self.random_seed,
                **params,
            ) for params in params_space]

            best_run = None
            best_val_train = -math.inf
            best_val_valid = -math.inf
            best_val_test = -math.inf

            for model_output in tasks:
                # TODO: get the score and compare with the best
                with model_output['metrics'].open('r') as f:
                    res = yaml.load(f)
                    # TODO: we don't have yet validation set (should add)
                    new_val_valid = res[self.metric]['val']

                    # TODO: in case of accuracy it is "<"
                    # in case of loss it should be ">"
                    if best_val_valid < new_val_valid:
                        best_run = res.get('run_id', None)
                        best_val_train = res[self.metric]['train']
                        best_val_valid = new_val_valid
                        best_val_test = res[self.metric]['test']

            metrics = {
                f'train_{self.metric}': float(best_val_train),
                f'val_{self.metric}': float(best_val_valid),
                f'test_{self.metric}': float(best_val_test),
            }
            mlflow.set_tag('best_run', best_run)
            mlflow.log_metrics(metrics)
            # TODO: maybe it also make sense to duplicate the best params in experiment?

            with self.output()['metrics'].open('w') as f:
                yaml.dump({
                    'metrics': metrics,
                    'best_run_id': best_run,
                }, f, default_flow_style=False)


if __name__ == '__main__':
    luigi.run()
