import numpy as np
import pdb
import sys
import traceback
import math
from math import inf
from copy import deepcopy
import argparse


class DecisionStump:

    def __init__(self):
        pass

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(y.shape)

        sorted_indices = []
        n_features = X.shape[1]
        for f in range(n_features):
            index = np.argsort(X[:, f])
            sorted_indices.append(index)

        n_data = X.shape[0]
        n_positive = np.sum(sample_weight[np.where(y == 1)])
        n_negative = np.sum(sample_weight[np.where(y != 1)])

        self.feature = 0
        self.threshold = - inf
        self.sign = 1 if n_positive > n_negative else -1
        self.min_err = min(n_positive, n_negative)

        for f in range(n_features):
            n_err_positive = n_positive
            n_err_negative = n_negative
            for i in range(n_data - 1):
                ind1 = sorted_indices[f][i]
                ind2 = sorted_indices[f][i + 1]
                n_err_positive += sample_weight[ind1] * y[ind1]
                n_err_negative -= sample_weight[ind1] * y[ind1]

                if n_err_positive < self.min_err or \
                   n_err_negative < self.min_err:
                    self.feature = f
                    self.threshold = (X[ind1, f] + X[ind2, f]) / 2

                    if n_err_positive < self.min_err:
                        self.sign = 1
                        self.min_err = n_err_positive

                    elif n_err_negative < self.min_err:
                        self.sign = -1
                        self.min_err = n_err_negative
        # pdb.set_trace()

    def predict(self, X):
        if self.sign == 1:
            y = np.where(X[:, self.feature] >= self.threshold,
                         1, -1)
        else:
            y = np.where(X[:, self.feature] < self.threshold,
                         1, -1)

        return y


class AdaBoost:

    def __init__(self, n_estimators,
                 base_estimator=DecisionStump()):
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator

    def fit(self, X, y):
        self.estimators = []
        self.estimator_weights = []
        weights = np.ones(y.shape)
        for i in range(self.n_estimators):
            estimator = deepcopy(self.base_estimator)
            estimator.fit(X, y, weights)
            y_ = estimator.predict(X)
            err = np.sum(weights[np.where(y_ != y)]) / y.shape[0]
            rescale = math.sqrt((1 - err) / (err))
            weights[np.where(y == y_)] /= rescale
            weights[np.where(y != y_)] *= rescale
            self.estimator_weights.append(math.log(rescale))
            self.estimators.append(estimator)

    def predict(self, X):
        y = np.zeros(X.shape[0])
        for i in range(self.n_estimators):
            y += self.estimator_weights[i] * self.estimators[i].predict(X)

        y = np.where(y >= 0, 1, -1)
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
    parser = argparse.ArgumentParser(description='MLT HW3 7')
    parser.add_argument('train', type=str, help='train.csv')
    parser.add_argument('test', type=str, help='train.csv')
    args = parser.parse_args()

    train = read_data(args.train)
    test = read_data(args.test)

    classifier = AdaBoost(200)
    classifier.fit(train['x'], train['y'])
    train['y_'] = classifier.predict(train['x'])
    test['y_'] = classifier.predict(test['x'])

    print('Train Accuracy =', accuracy(train['y'], train['y_']))
    print('Test Accuracy =', accuracy(test['y'], test['y_']))


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
