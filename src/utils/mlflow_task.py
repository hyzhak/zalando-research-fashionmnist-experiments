import luigi
from os import path
import mlflow
import yaml

from src.utils.params_to_filename import encode_task_to_filename
from src.utils.project import get_project_name
from src.utils.snake import get_class_name_as_snake


class MLFlowTask(luigi.Task):
    experiment = luigi.Parameter(
        default=get_project_name(),
        description='ml experiment name',
        significant=False,
    )

    # TODO: can use OptionalParameter
    parent_run_id = luigi.Parameter(
        default='',
        significant=False,
    )

    run_name = luigi.OptionalParameter(
        default=None,
        significant=False,
    )

    def output(self):
        encoded_params = encode_task_to_filename(self)
        class_name = get_class_name_as_snake(self)
        # for the moment mlflow task does output only for models
        output_dir = path.join('models', class_name, encoded_params)
        return {
            'mlflow': luigi.LocalTarget(
                path.join(output_dir, 'mlflow.yml')
            ),
            **self.ml_output(output_dir),
        }

    @staticmethod
    def get_run_id_from_result(model_result):
        """
        get mlflow run_id from MLFlowTask result

        :param model_result:
        :return:
        """
        if 'ml_flow' not in model_result:
            return None

        with model_result['ml_flow'].open('r') as f:
            return yaml.load(f).get('run_id')

    def ml_output(self, output_dir):
        """
        :param output_dir:
        should be overwritten by successor
        :return:
        """
        return {}

    def run(self):
        # because each luigi task inside it own worker
        # we need to simulate nesting parent -> child in case of child run
        if self.parent_run_id != '':
            with mlflow.start_run(
                    run_id=self.parent_run_id
            ):
                print('MLFLOW: before come to parent run', self.parent_run_id)
                yield from safe_iterator(self._run())
                print('MLFLOW: after parent run', self.parent_run_id)
        else:
            yield from safe_iterator(self._run())

    def _run(self):
        mlflow_output = self.output()['mlflow']
        run_id = None
        if mlflow_output.exists():
            with mlflow_output.open('r') as f:
                run_id = yaml.load(f).get('run_id')
                print('MLFLOW: continue mlflow run:', run_id)

        if self.experiment:
            # TODO: maybe we should get experiment from parent run?
            mlflow.set_experiment(self.experiment)

        print('MLFLOW: active_run() mlflow_task', mlflow.active_run())

        with mlflow.start_run(
                run_id=run_id,
                run_name=self.run_name,
                nested=self.parent_run_id != ''
        ) as run:
            with mlflow_output.open('w') as f:
                yaml.dump({
                    'run_id': run.info.run_id
                }, f, default_flow_style=False)
            yield from safe_iterator(self.ml_run(run.info.run_id))

    def ml_run(self, run_id):
        """
        should be overwritten by successor
        :return:
        """
        raise NotImplementedError()


def safe_iterator(i):
    """
    some methods doesn't return any
    :param i:
    :return:
    """
    return i or []
