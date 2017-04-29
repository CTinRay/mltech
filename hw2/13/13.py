import argparse
import numpy as np
from sklearn.svm import SVC


class SVR:

    def __init__(self, gamma=0.125, l=1, epsilon=0.5, kernel='rbf'):
        self.gamma = gamma
        self.l = l
        self.epsilon = epsilon
        self.kernel = kernel

    def fit(self, X, y):
        self.svc = SVC(C=1 / 2 / self.l, kernel=self.kernel, gamma=self.gamma)
        self.svc.fit(X, y)

    def predict(self, X):
        return self.svc.predict(X)


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
    parser = argparse.ArgumentParser(description='ML HW2 Problem 13')
    parser.add_argument('data', type=str, help='hw2_lssvm_all.dat')
    args = parser.parse_args()

    raw_data = read_data(args.data)
    train = {'x': raw_data['x'][:400], 'y': raw_data['y'][:400]}
    test = {'x': raw_data['x'][400:], 'y': raw_data['y'][400:]}

    gammas = [32, 2, 0.125]
    lambdas = [0.001, 1, 1000]

    for gamma in gammas:
        for l in lambdas:
            classifier = SVR(gamma=gamma, l=l)
            classifier.fit(train['x'], train['y'])
            train['y_'] = classifier.predict(train['x'])

            print('Gamma : ', gamma, 'lambda :', l)
            print('E in: %f' % (1 - accuracy(train['y'], train['y_'])))

            test['y_'] = classifier.predict(test['x'])
            print('E out: %f' % (1 - accuracy(test['y'], test['y_'])))


if __name__ == '__main__':
    main()
