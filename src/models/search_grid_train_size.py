import luigi
import numpy as np

from src.models.search_grid_base import SearchGridBase


class SearchGridTrainSize(SearchGridBase):
    max_runs = luigi.IntParameter(
        default=16,
        description=''
    )

    def get_params_space(self):
        return {
            'train_size': np.linspace(
                1.0 / self.max_runs, 1.0, self.max_runs, endpoint=False
            ),
        }

    def get_static_params(self):
        return {
            'batch_size': 32,
        }


if __name__ == '__main__':
    luigi.run()
