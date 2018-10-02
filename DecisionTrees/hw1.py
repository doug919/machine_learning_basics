##
# CS 578 - HW1
#
# Fall 2015 @ Purdue University
#
# @author I-Ta Lee
#
# @date 09/10/2015
#

import sys 
import os
import argparse
import numpy as np
import logging

from mylib import Utils
from mylib import Learners


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='CS180 HW1 Decision Tree by I-Ta Lee')
    parser.add_argument('training_data', metavar='TRAINING_DATA', 
                        help='training data in csv format')
    parser.add_argument('validation_data', metavar='VALIDATION_DATA', 
                        help='validation data in csv format')
    parser.add_argument('testing_data', metavar='TESTINIG_DATA', 
                        help='testing data in csv format')

    parser.add_argument('-p', '--do_prune', action='store_true', default=False,
                        help='reduced error pruning (DEFAULT: False)')
    parser.add_argument('-t', '--use_threshold', action='store_true', default=False,
                        help='use threshold to partition tree (DEFAULT: False)')
    parser.add_argument('-m', '--max_depth', metavar='DEPTH', type=int, default=-1, 
                        help='maximum depth of decision trees (DEFAULT: INFINITE)')
    parser.add_argument('-f', '--default_label', metavar='DEFAULT_LABEL', type=int, default=1, 
                        help='default label used when the majority vote has the same number of votes (DEFAULT: 1)')
    parser.add_argument('-r', '--int_range', metavar='RANGE', type=Utils.parse_range, default=range(1, 11), 
                        help='integer ranger of data. This follows the format "x-y", and it means range(x, y+1) (DEFAULT: 1-10, which is equal to range(1, 11)')

    parser.add_argument('-e', '--print_tree', action='store_true', default=False,
                        help='print tree (DEFAULT: False)')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, 
                        help='show info messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False, 
                        help='show debug messages')
    args = parser.parse_args(argv)
    return args

# the last column in the data is label
# return X, y
def split_label(data):
    label_idx = data.shape[1] - 1
    X = data[:, 0:label_idx]
    y = data[:, label_idx]
    return X, y

if __name__ == '__main__':
    
    # get arguments
    args = get_arguments(sys.argv[1:])

    # set debug level
    if args.debug:
        loglevel = logging.DEBUG
    elif args.verbose:
        loglevel = logging.INFO
    else:
        loglevel = logging.ERROR
    logging.basicConfig(format='[%(levelname)s][%(name)s] %(message)s', level=loglevel) 
    logger = logging.getLogger(__name__)

    # check files exist
    if not os.path.exists(args.training_data):
        raise Exception("file %s does not exist." % (args.training_data))

    if not os.path.exists(args.validation_data):
        raise Exception("file %s does not exist." % (args.validation_data))

    if not os.path.exists(args.testing_data):
        raise Exception("file %s does not exist." % (args.testing_data))

    # load data
    training_set = np.loadtxt(args.training_data)
    validation_set = np.loadtxt(args.validation_data)
    testing_set = np.loadtxt(args.testing_data)

    # process data
    X, y = split_label(training_set)
    X_validation, y_validation = split_label(validation_set)
    X_test, y_test = split_label(testing_set)

    # learning
    learner = Learners.ID3(loglevel=loglevel, 
                           int_range=args.int_range, 
                           use_threshold=args.use_threshold, 
                           max_depth=args.max_depth,
                           default_label=args.default_label)
    learner.fit(X=X, y=y)

    # results
    if args.print_tree:
        learner.print_tree()
    print "training accuracy = %f" % (learner.score(X, y))
    print "validating accuracy = %f" % (learner.score(X_validation, y_validation))
    print "testing accuracy = %f" % (learner.score(X_test, y_test))

    if args.do_prune:
        learner.prune_tree(X, y, X_validation, y_validation)
        if args.print_tree:
            learner.print_tree()
        print "post-pruning training accuracy = %f" % (learner.score(X, y))
        print "post-pruning validating accuracy = %f" % (learner.score(X_validation, y_validation))
        print "post-pruning testing accuracy = %f" % (learner.score(X_test, y_test))
        











