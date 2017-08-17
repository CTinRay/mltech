import numpy as np
import pdb
import sys
import traceback
import argparse
import matplotlib.pyplot as plt
from classifiers import AdaBoost
from utils import read_data, accuracy


def main():
    parser = argparse.ArgumentParser(description='MLT HW3 7')
    parser.add_argument('train', type=str, help='train.csv')
    parser.add_argument('test', type=str, help='train.csv')
    parser.add_argument('--plot', type=str,
                        help='ada-eout-g.png', default='ada-eout-g.png')
    args = parser.parse_args()

    train = read_data(args.train)
    test = read_data(args.test)

    classifier = AdaBoost(300)
    classifier.fit(train['x'], train['y'], test)
    train['y_'] = classifier.predict(train['x'])
    test['y_'] = classifier.predict(test['x'])

    print('Train Accuracy =', accuracy(train['y'], train['y_']))
    print('Test Accuracy =', accuracy(test['y'], test['y_']))

    eout_g = []
    for i in range(classifier.n_estimators):
        y_ = classifier.estimators[i].predict(test['x'])
        eout_g.append(1 - accuracy(y_, test['y']))

    print('eout g1 =', eout_g[0],
          'alpha 1 =', classifier.estimator_weights[0])

    plt.figure(1)
    plt.xlabel('$t$', fontsize=18)
    plt.ylabel('$E_{out}(G_t)$', fontsize=16)
    plt.plot(np.arange(len(eout_g)), eout_g)
    plt.savefig(args.plot, dpi=300)


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
