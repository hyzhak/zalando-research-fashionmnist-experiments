from ax.service.ax_client import AxClient
from ax.storage.sqa_store.structs import DBSettings
import luigi
import pickle
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
        filename = encode_task_to_filename(self, ['model_name'])
        class_name = get_class_name_as_snake(self)
        return {
            'metrics': luigi.LocalTarget(
                f'reports/metrics/{self.model_name}/{class_name}/{filename}.yaml'
            ),
            'ax_experiment': luigi.LocalTarget(
                f'reports/metrics/{self.model_name}/{class_name}/{filename}.pickle',
                format=luigi.format.Nop
            )
        }

    def _get_ax_experiment(self):
        ax_experiment_file = self.output()['ax_experiment']
        if not ax_experiment_file.exists():
            return None

        with ax_experiment_file.open('r') as f:
            return pickle.load(f)

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

        # TODO: make reproducibility of search
        # without it we will get each time new params

        # for example we can use:
        # ax.storage.sqa_store.structs.DBSettings
        # DBSettings(url="sqlite://<path-to-file>")
        # to store experiments
        ax = AxClient(
            # can't use that feature yet.
            # got error
            # NotImplementedError:
            # Saving and loading experiment in `AxClient` functionality currently under development.
            # db_settings=DBSettings(url=self.output()['ax_settings'].path)
        )

        # FIXME: temporal solution while ax doesn't have api to (re-)store state

        class_name = get_class_name_as_snake(self)
        ax.create_experiment(
            name=f'{class_name}_experiment',
            parameters=params_space,
            objective_name='score',
            minimize=should_minimize(self.metric),
            # parameter_constraints=['x1 + x2 <= 2.0'],  # Optional.
            # outcome_constraints=['l2norm <= 1.25'],  # Optional.
        )

        trial_index = 0
        experiment = self._get_ax_experiment()
        if experiment:
            print('AX: restore experiment')
            print('AX: num_trials:', experiment.num_trials)
            ax._experiment = experiment
            trial_index = experiment.num_trials - 1

        model_task = get_model_task_by_name(self.model_name)

        while trial_index < self.max_runs:
            print(f'AX: Running trial {trial_index + 1}/{self.max_runs}...')

            # get last unfinished trial
            parameters = get_last_unfinished_params(ax)

            if parameters is None:
                print('AX: generate new Trial')
                parameters, trial_index = ax.get_next_trial()

                # good time to store experiment (with new Trial)
                with self.output()['ax_experiment'].open('w') as f:
                    print('AX: store experiment: ', ax.experiment)
                    pickle.dump(ax.experiment, f)

            print('AX: parameters', parameters)

            # now is time to evaluate model
            model_result = yield model_task(
                parent_run_id=run_id,
                random_seed=self.random_seed,
                # TODO: actually we should be able to pass even nested params
                # **parameters,
                optimizer_props=parameters
            )

            # TODO: store run_id in Trial
            model_run_id = self.get_run_id_from_result(model_result)

            with model_result['metrics'].open('r') as f:
                model_metrics = yaml.load(f)
                model_score = model_metrics[self.metric]['val']

            print('AX: complete trial:', trial_index)

            ax.complete_trial(
                trial_index=trial_index,
                raw_data=model_score  # {
                #     'score': model_score,
                #     'run_id': model_run_id,
                # }
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


def get_last_unfinished_params(ax):
    """
    get params of the last unfinished Trial

    :param ax:
    :return:
    """
    if ax.experiment.num_trials == 0:
        return None

    trial_index = ax.experiment.num_trials - 1
    last_trial = ax.experiment.trials[trial_index]
    print('AX: last trial is', trial_index)
    print('AX: status of last trial:', last_trial.status)
    # based on ax_client.get_next_trial
    if not last_trial.status.is_deployed:
        return None

    # we have only one arm here (until Fb will implement more)
    parameters = last_trial.arm.parameters
    print('AX: restore last params', parameters)
    return parameters


if __name__ == '__main__':
    luigi.run()
