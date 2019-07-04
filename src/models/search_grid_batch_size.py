import luigi
import numpy as np

from src.models.search_grid_base import SearchGridBase


class SearchGridBatchSize(SearchGridBase):
    max_runs = luigi.IntParameter(
        default=16,
        description=''
    )

    min_batch_size = luigi.IntParameter(
        default=1,
        description='min batch size'
    )

    max_batch_size = luigi.IntParameter(
        default=64,
        description='min batch size'
    )

    def get_params_space(self):
        return {
            'batch_size': np.linspace(self.min_batch_size,
                                      self.max_batch_size,
                                      self.max_runs, dtype=int),
        }


if __name__ == '__main__':
    luigi.run()
