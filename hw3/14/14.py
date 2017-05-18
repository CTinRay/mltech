import numpy as np
import pdb
import sys
import traceback
import math
from math import inf
from copy import deepcopy
import argparse
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree


class DecisionStump:

    def _gini(self, np_left, nn_left, np_right, nn_right):
        def gini_sub(n_positive, n_negative):
            total = n_positive + n_negative + sys.float_info.epsilon
            mu_p = n_positive / total
            mu_n = n_negative / total
            return 1 - mu_p ** 2 - mu_n ** 2

        return gini_sub(np_left, nn_left) + gini_sub(np_right, nn_right)

    def __init__(self):
        self.loss = self._gini
        pass

    def fit(self, X, y):
        sorted_indices = []
        n_features = X.shape[1]
        for f in range(n_features):
            index = np.argsort(X[:, f])
            sorted_indices.append(index)

        n_data = X.shape[0]
        n_positive = np.sum(y == 1)
        n_negative = np.sum(y != 1)

        self.feature = 0
        self.threshold = - inf
        self.min_loss = self._gini(0, 0, n_positive, n_negative)

        for f in range(n_features):
            np_left, nn_left = 0, 0
            np_right, nn_right = n_positive, n_negative
            for i in range(n_data - 1):
                ind1 = sorted_indices[f][i]
                ind2 = sorted_indices[f][i + 1]
                if y[ind1] == 1:
                    np_left += 1
                    np_right -= 1
                else:
                    nn_left += 1
                    nn_right -= 1

                loss = self.loss(np_left, nn_left, np_right, nn_right)
                if loss < self.min_loss:
                    # pdb.set_trace()
                    self.min_loss = loss
                    self.threshold = (X[ind1, f] + X[ind2, f]) / 2
                    self.feature = f

        right_inds = np.where(X[:, self.feature] > self.threshold)
        left_inds = np.where(X[:, self.feature] < self.threshold)
        nn_right = np.sum(y[right_inds] != 1)
        np_left = np.sum(y[left_inds] == 1)
        err_positive = (nn_right + np_left) / y.shape[0]
        self.sign = +1 if err_positive < 0.5 else -1
        # pdb.set_trace()

    def predict(self, X):
        if self.sign == 1:
            y = np.where(X[:, self.feature] >= self.threshold,
                         1, -1)
        else:
            y = np.where(X[:, self.feature] <= self.threshold,
                         1, -1)

        return y


class DecisionTree:

    def __init__(self):
        self.base_estimator = DecisionStump()
        return

    def _make_tree(self, X, y):
        node = {}
        node['classifier'] = deepcopy(self.base_estimator)
        node['classifier'].fit(X, y)
        y_ = node['classifier'].predict(X)
        left_inds = np.where(y_ == -1)
        right_inds = np.where(y_ == 1)

        # pdb.set_trace()
        if np.any(y[left_inds] != -1):
            node['l'] = self._make_tree(X[left_inds], y[left_inds])
        else:
            node['l'] = None

        if np.any(y[right_inds] != 1):
            node['r'] = self._make_tree(X[right_inds], y[right_inds])
        else:
            node['r'] = None

        return node

    def fit(self, X, y, valid=None):
        self.root = self._make_tree(X, y)

    def predict(self, X):
        y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            node = self.root
            while node is not None:
                y[i] = node['classifier'].predict(X[i:i + 1])
                if y[i] == 1:
                    node = node['r']

                else:
                    node = node['l']

        return y

    def plot(self, filename):
        def plot_node(f, node, start):
            clf = node['classifier']
            if clf.sign == +1:
                f.write('%d [label="X[%d] >= %f"] ;\n'
                        % (start, clf.feature, clf.threshold))
            else:
                f.write('%d [label="X[%d] >= %f"] ;\n'
                        % (start, clf.feature, clf.threshold))

            parent = start
            child = start + 1
            if node['l'] is not None:
                start = plot_node(f, node['l'], child)
            else:
                f.write('%d [label="+1"] ;\n' % child)
                start += 1
                
            f.write('%d -> %d ;\n' % (parent, child))

            child = start + 1
            if node['r'] is not None:
                start = plot_node(f, node['r'], child)
            else:
                f.write('%d [label="-1"] ;\n' % child)
                start += 1
                
            f.write('%d -> %d ;\n' % (parent, child))

            return start + 1

        with open(filename, 'w') as f:
            f.write('digraph Tree {\n')
            plot_node(f, self.root, 1)
            f.write('}')


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
    parser.add_argument('--plot', type=str, help='error.png',
                        default='tree.dot')
    args = parser.parse_args()

    train = read_data(args.train)
    test = read_data(args.test)

    # classifier = DecisionTreeClassifier()
    # classifier.fit(train['x'], train['y'])
    classifier = DecisionTree()
    classifier.fit(train['x'], train['y'], test)
    train['y_'] = classifier.predict(train['x'])
    test['y_'] = classifier.predict(test['x'])

    print('Train Accuracy =', accuracy(train['y'], train['y_']))
    print('Test Accuracy =', accuracy(test['y'], test['y_']))

    classifier.plot(args.plot)

if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
