import math
from math import inf
from copy import deepcopy
import numpy as np
import sys
from bagging import Bagging
from numba import jit


class DecisionStump:

    def _gini(self, np_left, nn_left, np_right, nn_right):
        def gini_sub(n_positive, n_negative):
            total = n_positive + n_negative + sys.float_info.epsilon
            mu_p = n_positive / total
            mu_n = n_negative / total
            return 1 - mu_p ** 2 - mu_n ** 2

        loss_left = (np_left + nn_left) * gini_sub(np_left, nn_left)
        loss_right = (np_right + nn_right) * gini_sub(np_right, nn_right)
        return loss_left + loss_right

    def __init__(self):
        self.loss = self._gini
        pass

    @jit
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

    @jit
    def _make_tree(self, X, y):
        node = {}
        node['classifier'] = deepcopy(self.base_estimator)
        node['classifier'].fit(X, y)
        node['n_sample'] = X.shape[0]
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

    @jit
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
                f.write('%d [label="X[%d] >= %f"] ;\n' %
                        (start, clf.feature, clf.threshold))
            else:
                f.write('%d [label="X[%d] <= %f"] ;\n' %
                        (start, clf.feature, clf.threshold))

            parent = start
            child = start + 1
            if node['l'] is not None:
                start = plot_node(f, node['l'], child)
            else:
                f.write('%d [label="-1"] ;\n' % child)
                start += 1

            f.write('%d -> %d ;\n' % (parent, child))

            child = start + 1
            if node['r'] is not None:
                start = plot_node(f, node['r'], child)
            else:
                f.write('%d [label="+1"] ;\n' % child)
                start += 1

            f.write('%d -> %d ;\n' % (parent, child))

            return start + 1

        with open(filename, 'w') as f:
            f.write('digraph Tree {\n')
            f.write('node [shape=box] ;')
            plot_node(f, self.root, 1)
            f.write('}')


class RandomForest:

    def __init__(self, n_estimators=20):
        self.n_estimators = n_estimators
        self.forest = Bagging(base_estimator=DecisionTree(),
                              n_estimators=self.n_estimators)

    def fit(self, X, y):
        self.forest.fit(X, y)

    def predict(self, X, n_trees=None):
        return self.forest.predict(X, n_trees)
