import numpy as np
import pdb
import sys
import traceback
import argparse
from classifiers import DecisionTree
from utils import accuracy, read_data


def main():
    parser = argparse.ArgumentParser(description='MLT HW3 7')
    parser.add_argument('train', type=str, help='train.csv')
    parser.add_argument('test', type=str, help='train.csv')
    parser.add_argument('--plot', type=str, help='error.png',
                        default='tree.dot')
    args = parser.parse_args()

    train = read_data(args.train)
    test = read_data(args.test)

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
