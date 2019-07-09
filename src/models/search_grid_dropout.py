import luigi
import numpy as np

from src.models.search_grid_base import SearchGridBase


class SearchGridDropout(SearchGridBase):
    max_runs = luigi.IntParameter(
        default=16,
        description=''
    )

    dropout_min = luigi.IntParameter(
        default=0.0,
        description='min dropout'
    )

    dropout_max = luigi.IntParameter(
        default=0.9,
        description='max dropout'
    )

    def get_params_space(self):
        return {
            'dropout': np.linspace(
                self.dropout_min, self.dropout_max, self.max_runs, endpoint=False
            ),
        }


if __name__ == '__main__':
    luigi.run()
