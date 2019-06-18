import luigi

from src.visualization.scikit_score import ScikitScore


class AllReports(luigi.WrapperTask):
    def requires(self):
        return ScikitScore()


if __name__ == '__main__':
    luigi.run(main_task_cls=AllReports)
