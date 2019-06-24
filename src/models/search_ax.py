from ax.service.ax_client import AxClient
import luigi
import yaml

from src.models import get_model_task_by_name
from src.utils.metrics import should_minimize
from src.utils.mlflow_task import MLFlowTask
from src.utils.params_to_filename import encode_task_to_filename
from src.utils.seed_randomness import seed_randomness
from src.utils.snake import get_class_name_as_snake


class SearchAx(MLFlowTask):
    model_name = luigi.Parameter(
        default='simple_cnn',
        description='model name (e.g. logistic_regression, simple_cnn)'
    )
    algo = luigi.Parameter(
        default='',
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
        optimizer_name = 'hyper_ax'
        filename = encode_task_to_filename(self, ['model_name'])
        class_name = get_class_name_as_snake(self)
        return {
            'metrics': luigi.LocalTarget(
                f'reports/metrics/{self.model_name}/{class_name}/{filename}.yaml'
            )
        }

    def ml_run(self, run_id=None):
        seed_randomness(self.random_seed)

        # should land to 'optimizer_props'
        params_space = [
            {
                'name': 'lr',
                'type': 'range',
                'bounds': [0.00001, 0.001],
                'value_type': 'float',
                # 'log_scale': True,
            },
            {
                'name': 'beta_1',
                'type': 'range',
                'bounds': [.0, 0.9999],
                'value_type': 'float',
                # 'log_scale': True,
            },
            # {
            #     'name': 'beta_2',
            #     'type': 'range',
            #     'bounds': [.0, 0.9999],
            #     'value_type': 'float',
            #     'log_scale': True,
            # }
        ]

        ax = AxClient()

        ax.create_experiment(
            name='hartmann_test_experiment',
            parameters=params_space,
            objective_name='hartmann6',
            minimize=should_minimize(self.metric),
            # parameter_constraints=['x1 + x2 <= 2.0'],  # Optional.
            # outcome_constraints=['l2norm <= 1.25'],  # Optional.
        )

        model_task = get_model_task_by_name(self.model_name)

        for i in range(self.max_runs):
            print(f'Running trial {i+1}/{self.max_runs}...')
            parameters, trial_index = ax.get_next_trial()

            # TODO: we may need to pickle experiment so it could lift off from the same point
            # in case when Luigi break evaluation because model task output doesn't ready yet

            print('parameters', parameters)

            # now is time to evaluate model
            model_result = yield model_task(
                parent_run_id=run_id,
                random_seed=self.random_seed,
                # TODO: actually we should be able to pass even nested params
                # **parameters,
                optimizer_props=parameters
            )

            model_run_id = self.get_run_id_from_result(model_result)

            with model_result['metrics'].open('r') as f:
                model_metrics = yaml.load(f)
                model_score = model_metrics[self.metric]['val']

            ax.complete_trial(
                trial_index=trial_index,
                raw_data={
                    'score': model_score,
                    'run_id': model_run_id,
                }
            )

        best_parameters, values = ax.get_best_parameters()
        means, covariances = values

        # TODO: store results
        print('TODO: store score')
        print('best params', best_parameters)
        print('best means', means)
        print('best covariances', covariances)

        # TODO: store plots as mlflow artifcats
        # https://ax.dev/tutorials/gpei_hartmann_service.html#6.-Plot-the-response-surface-and-optimization-trace


if __name__ == '__main__':
    luigi.run()
