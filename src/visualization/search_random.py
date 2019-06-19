import luigi
import mlflow
import numpy as np
import random
import yaml

# from src.models import get_model_task_by_name
from src.utils.params_to_filename import encode_task_to_filename
from src.visualization.log_metrics import LogMetrics

_inf = np.finfo(np.float64).max


class SearchRandom(luigi.Task):
    model_name = luigi.Parameter(
        default='logistic-regression',
        description='model name (e.g. logistic-regression)'
    )
    metric = luigi.Parameter(
        default='accuracy',
        description='metric to optimize on'
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
        filename = encode_task_to_filename(self)
        return luigi.LocalTarget(
            f'reports/scores/best_random_search__{filename}.yaml'
        )

    def run(self):
        print('Search Random # 1')
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        params_space = [{
            # remove saga (because it is way too slow, x10 of so)
            # aslo newton-cg, because it is x100 slower
            # 'solver': random.choice(['newton-cg', 'sag', 'saga', 'lbfgs']),
            'solver': random.choice(['sag', 'lbfgs']),
            'C': np.random.uniform(0, 1.0)
        } for _ in range(self.max_runs)]

        print('Search Random # 2')
        with mlflow.start_run() as run:
            experiment_id = run.info.experiment_id

            print('Search Random # 3')
            # run all random tasks in parallel
            # TODO: pass train, test, and validation sets?
            tasks = yield [LogMetrics(
                model_name=self.model_name,
                model_params={
                    **params,
                    'random_seed': self.random_seed,
                },
                experiment_id=experiment_id,
            ) for params in params_space]

            print('Search Random # 4')
            # find the best params (based on validation metric)

            best_run = None
            best_val_train = -_inf
            best_val_valid = -_inf
            best_val_test = -_inf

            for model_output in tasks:
                # TODO: get the score and compare with the best
                with model_output['score'].open('r') as f:
                    res = yaml.load(f)
                    # TODO: we don't have yet validation set (should add)
                    # new_val_valid = res['valid'][self.metric]
                    new_val_valid = res['test'][self.metric]

                    # TODO: in case of accuracy it is "<"
                    # in case of loss it should be ">"
                    print('best_val_valid < new_val_valid', best_val_valid, new_val_valid)
                    if best_val_valid < new_val_valid:
                        print('find better', new_val_valid)
                        best_run = res['run_id']
                        best_val_train = res['train'][self.metric]
                        best_val_valid = new_val_valid
                        best_val_test = res['test'][self.metric]

            metrics = {
                f'train_{self.metric}': float(best_val_train),
                # f'val_{self.metric}': best_val_valid,
                f'test_{self.metric}': float(best_val_test),
            }
            mlflow.set_tag('best_run', best_run)
            mlflow.log_metrics(metrics)

            with self.output().open('w') as f:
                yaml.dump({
                    'metrics': metrics,
                    'best_run_id': best_run,
                }, f, default_flow_style=False)


if __name__ == '__main__':
    luigi.run()
