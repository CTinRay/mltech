import argparse
import numpy as np


class LSSVM:

    def __init__(self, gamma=0.125, l=1e-5):
        self.gamma = gamma
        self.l = l
        self._kernel = self._rbf
        pass

    def _rbf(self, x1, x2, gamma):
        return np.exp(-gamma * np.dot(x1 - x2, (x1 - x2).T))

    def fit(self, X, y):
        K = self._kernel(X, X, self.gamma)
        self.X = X
        inv = np.linalg.inv(self.l * np.identity(X.shape[0]) + K)
        self.beta = np.dot(inv, y)

    def predict(self, X):
        y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            h = 0
            for j in range(self.X.shape[0]):
                x1 = X[i].reshape(1, -1)
                x2 = self.X[j].reshape(1, -1)
                h += self.beta[j] * self._kernel(x1, x2, self.gamma)

            y[i] = 1 if h > 0 else -1

        return y


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
    train = {'x': raw_data['x'][:400], 'y': raw_data['y'][:400]}
    test = {'x': raw_data['x'][400:], 'y': raw_data['y'][400:]}

    gammas = [32, 2, 0.125]
    lambdas = [0.001, 1, 1000]

    for gamma in gammas:
        for l in lambdas:
            classifier = LSSVM(gamma=gamma, l=l)
            classifier.fit(train['x'], train['y'])
            train['y_'] = classifier.predict(train['x'])

            print('Gamma :', gamma, 'lambda :', l)
            print('E in:', 1 - accuracy(train['y'], train['y_']))

            test['y_'] = classifier.predict(test['x'])
            print('E out:', 1 - accuracy(test['y'], test['y_']))


if __name__ == '__main__':
    main()
