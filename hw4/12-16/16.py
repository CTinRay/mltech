import numpy as np
import pdb
import sys
import traceback
import argparse
from classifiers import DecisionStump
from bagging import Bagging
from utils import accuracy, read_data
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='MLT HW4 13')
    parser.add_argument('train', type=str, help='train.csv')
    parser.add_argument('test', type=str, help='train.csv')
    parser.add_argument('--plot', type=str, help='ein.png',
                        default='stump-eout.png')
    parser.add_argument('--n_stumps', type=int, help='number of trees',
                        default=30000)
    args = parser.parse_args()

    train = read_data(args.train)
    test = read_data(args.test)

    classifier = Bagging(base_estimator=DecisionStump(),
                         n_estimators=args.n_stumps)
    classifier.fit(train['x'], train['y'])
    train['y_'] = classifier.predict(train['x'])
    test['y_'] = classifier.predict(test['x'])

    e_outs = []
    vote = np.zeros(test['x'].shape[0])
    for i in range(args.n_stumps):
        vote += classifier.estimators[i].predict(test['x'])

        y_ = np.where(vote > 0, 1, -1)
        err = 1 - accuracy(test['y'], y_)
        e_outs.append(err)

    plt.figure(1)
    plt.plot(np.arange(args.n_stumps), e_outs)
    plt.xlabel('First n trees')
    plt.ylabel('$E_{out}$')
    plt.savefig(args.plot, dpi=400)

    print('Train Error =', 1 - accuracy(train['y'], train['y_']))
    print('Test Error =', 1 - accuracy(test['y'], test['y_']))


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
