import luigi


class LabelTitles(luigi.ExternalTask):
    def output(self):
        return luigi.LocalTarget('data/raw/label-titles.yaml')


if __name__ == '__main__':
    luigi.run()
