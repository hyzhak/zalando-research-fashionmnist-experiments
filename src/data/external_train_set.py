import luigi


class ExternalTrainSet(luigi.ExternalTask):
    def output(self):
        return luigi.LocalTarget('data/raw/fashion-mnist_train.csv')


if __name__ == '__main__':
    luigi.run()
