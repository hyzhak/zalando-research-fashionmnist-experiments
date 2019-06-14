import luigi


class ExternalTestSet(luigi.ExternalTask):
    def output(self):
        return luigi.LocalTarget('data/raw/fashion-mnist_test.csv')


if __name__ == '__main__':
    luigi.run()
