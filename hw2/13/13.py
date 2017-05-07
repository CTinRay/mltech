import argparse
import numpy as np
# from sklearn.svm import SVR
import sklearn.svm


class SVR:

    def __init__(self, gamma=0.125, c=1, epsilon=0.5, kernel='rbf'):
        self.gamma = gamma
        self.c = c
        self.epsilon = epsilon
        self.kernel = kernel

    def fit(self, X, y):
        self.svc = sklearn.svm.SVR(C=self.c, kernel=self.kernel,
                                   gamma=self.gamma, epsilon=self.epsilon)
        self.svc.fit(X, y)

    def predict(self, X):
        predict = self.svc.predict(X)
        return np.where(predict > 0, 1, -1)


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
    cs = [0.001, 1, 1000]

    for gamma in gammas:
        for c in cs:
            classifier = SVR(gamma=gamma, c=c)
            classifier.fit(train['x'], train['y'])
            train['y_'] = classifier.predict(train['x'])

            print('Gamma:', gamma, 'C:', c)
            print('E in: %f' % (1 - accuracy(train['y'], train['y_'])))

            test['y_'] = classifier.predict(test['x'])
            print('E out: %f' % (1 - accuracy(test['y'], test['y_'])))


if __name__ == '__main__':
    main()
