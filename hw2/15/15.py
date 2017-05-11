import argparse
import numpy as np
import copy
import pdb


class LSSVM:

    def __init__(self, gamma=0.125, l=1e-5):
        self.gamma = gamma
        self.l = l

    def fit(self, X, y):
        inv = np.linalg.inv(X.T @ X + self.l * np.identity(X.shape[1]))
        self.w = inv @ X.T @ y.reshape(-1, 1)

    def predict(self, X):
        predict = X @ self.w
        y = np.where(predict > 0, 1, -1)
        # pdb.set_trace()
        return y.reshape(-1,)


class Bagging:

    def __init__(self, n_estimators=10, base_estimator=None):
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator

    def fit(self, X, y):
        self.estimators = []

        # initialize estimators
        for i in range(self.n_estimators):
            self.estimators.append(copy.deepcopy(self.base_estimator))

        for i in range(self.n_estimators):
            rand_inds = np.random.randint(0, X.shape[0], X.shape[0])
            sample_xs = X[rand_inds]
            sample_ys = y[rand_inds]
            self.estimators[i].fit(sample_xs, sample_ys)

    def predict(self, X):
        vote = np.zeros(X.shape[0])
        for i in range(self.n_estimators):
            vote += self.estimators[i].predict(X)

        ys = np.where(vote > 0, 1, -1)
        return ys


def read_data(filename):
    x = []
    y = []
    with open(filename) as f:
        for l in f:
            cols = list(map(float, l.split()))
            x.append(cols[:-1])
            y.append(cols[-1])
    return {'x': np.array(x), 'y': np.array(y)}


def accuracy(y, y_):
    return np.sum(y == y_) / y.shape[0]


def main():
    parser = argparse.ArgumentParser(description='ML HW2 Problem 11')
    parser.add_argument('data', type=str, help='hw2_lssvm_all.dat')
    args = parser.parse_args()

    raw_data = read_data(args.data)
    zeros = np.ones([raw_data['x'].shape[0], 1])
    raw_data['x'] = np.concatenate([zeros, raw_data['x']], axis=1)
    train = {'x': raw_data['x'][:400], 'y': raw_data['y'][:400]}
    test = {'x': raw_data['x'][400:], 'y': raw_data['y'][400:]}

    lambdas = [0.01, 0.1, 1, 10, 100]

    for l in lambdas:
        lssvm = LSSVM(l=l)
        classifier = Bagging(base_estimator=lssvm, n_estimators=201)
        classifier.fit(train['x'], train['y'])
        train['y_'] = classifier.predict(train['x'])

        print('lambda :', l)
        print('E in: %f' % (1 - accuracy(train['y'], train['y_'])))

        test['y_'] = classifier.predict(test['x'])
        print('E out: %f' % (1 - accuracy(test['y'], test['y_'])))


if __name__ == '__main__':
    main()
