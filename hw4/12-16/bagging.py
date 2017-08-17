import copy
import numpy as np


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

    def predict(self, X, n_estimators=None):
        if n_estimators is None:
            n_estimators = self.n_estimators

        vote = np.zeros(X.shape[0])
        for i in range(n_estimators):
            vote += self.estimators[i].predict(X)

        ys = np.where(vote > 0, 1, -1)
        return ys
