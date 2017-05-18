import numpy as np
import pdb
import sys
import traceback
import argparse
from classifiers import DecisionTree
from utils import accuracy, read_data


def prune_leaves(tree, train, test):
    def check_is_leave(node):
        if node['l'] is None and node['r'] is None:
            return True
        else:
            return False

    def prune_and_predict(node, lr):
        child = node[lr]
        clfr = child['classifier']
        if clfr.sign == +1:
            print('prune leave: X[%d] >= %f'
                  % (clfr.feature, clfr.threshold))
        else:
            print('prune leave: X[%d] <= %f'
                  % (clfr.feature, clfr.threshold))

        node[lr] = None
        y_train = tree.predict(train['x'])
        err_train = 1 - accuracy(y_train, train['y'])
        y_test = tree.predict(test['x'])
        err_test = 1 - accuracy(y_test, test['y'])
        print('Train Error = %f, Test Error = %f' % (err_train, err_test))
        node[lr] = child

    def find_and_prune(node):
        for lr in ['l', 'r']:
            if node[lr] is None:
                continue
            if check_is_leave(node[lr]):
                prune_and_predict(node, lr)
            else:
                find_and_prune(node[lr])

    find_and_prune(tree.root)


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

    prune_leaves(classifier, train, test)


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
