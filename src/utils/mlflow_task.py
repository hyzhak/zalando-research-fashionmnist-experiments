import luigi
import mlflow
import re
import yaml

from src.utils.params_to_filename import encode_task_to_filename
from src.utils.seed_randomness import seed_randomness

_reg = re.compile(r'(?!^)(?<!_)([A-Z])')


def camel_to_snake(s):
    """
    Is it ironic that this function is written in camel case, yet it
    converts to snake case? hmm..
    """
    return _reg.sub(r'_\1', s).lower()


class MLFlowTask(luigi.Task):
    experiment = luigi.Parameter(
        default='',
        description='ml experiment name',
        significant=False,
    )

    parent_run_id = luigi.Parameter(
        default='',
        significant=False,
    )

    def output(self):
        filename = encode_task_to_filename(self)
        class_name = camel_to_snake(type(self).__name__)
        return {
            'mlflow': luigi.LocalTarget(
                f'reports/mlflow/{class_name}/{filename}'
            ),
            **self.ml_output(),
        }

    def ml_output(self):
        """
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
                yield from self._run()
        else:
            yield from self._run()

    def _run(self):
        mlflow_output = self.output()['mlflow']
        run_id = None
        if mlflow_output.exists():
            with mlflow_output.open('r') as f:
                run_id = yaml.load(f).get('run_id')
                print('continue mlflow run:', run_id)
        elif self.experiment:
            mlflow.set_experiment(self.experiment)

        print('run_id', run_id)
        with mlflow.start_run(
                run_id=run_id,
                nested=self.parent_run_id != ''
        ) as run:
            with mlflow_output.open('w') as f:
                yaml.dump({
                    'run_id': run.info.run_id
                }, f, default_flow_style=False)
            yield from self.ml_run(run.info.run_id)

    def ml_run(self, run_id):
        """
        should be overwritten by successor
        :return:
        """
        raise NotImplementedError()
