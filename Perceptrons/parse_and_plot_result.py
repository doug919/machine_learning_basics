"""
CS578 - HW2, Fall 2015, Purdue University
main function
@author I-Ta Lee
@date 09/28/2015
"""
import sys 
import os
import argparse
import logging

import numpy as np
#import matplotlib.pyplot as plt

def get_arguments(argv):
    parser = argparse.ArgumentParser(description='This program processes the stdout of hw2.py, collecting the results and ploting figures.')
    parser.add_argument('result_file', metavar='FILE_PATH', 
                        help='a file containing the results')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, 
                        help='show info messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False, 
                        help='show debug messages')
    args = parser.parse_args(argv)
    return args



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

    # Open a file
    fp = open(args.result_file, "r")

    STATE_EPOCH = 0
    STATE_TRAIN_ACC = 1
    STATE_TRAIN_PREC = 2
    STATE_TRAIN_RECALL = 3
    STATE_TRAIN_F1 = 4
    STATE_VALID_ACC = 5
    STATE_VALID_PREC = 6
    STATE_VALID_RECALL = 7
    STATE_VALID_F1 = 8
    STATE_TEST_ACC = 9
    STATE_TEST_PREC = 10
    STATE_TEST_RECALL = 11
    STATE_TEST_F1 = 12
    STATE_SEP_LINE = 13
    
    state = STATE_EPOCH
    epoch = []
    train_acc = []
    train_prec = []
    train_recall = []
    train_f1 = []
    valid_acc = []
    valid_prec = []
    valid_recall = []
    valid_f1 = []
    test_acc = []
    test_prec = []
    test_recall = []
    test_f1 = []
    for line in fp:

        if state == STATE_EPOCH:
            epoch += [int(line.split(' ')[1])]
        elif state == STATE_TRAIN_ACC:
            train_acc += [float(line.split(' ')[2])]
        elif state == STATE_TRAIN_PREC:
            train_prec += [float(line.split(' ')[2])]
        elif state == STATE_TRAIN_RECALL:
            train_recall += [float(line.split(' ')[2])]
        elif state == STATE_TRAIN_F1:
            train_f1 += [float(line.split(' ')[2])]
        elif state == STATE_VALID_ACC:
            valid_acc += [float(line.split(' ')[2])]
        elif state == STATE_VALID_PREC:
            valid_prec += [float(line.split(' ')[2])]
        elif state == STATE_VALID_RECALL:
            valid_recall += [float(line.split(' ')[2])]
        elif state == STATE_VALID_F1:
            valid_f1 += [float(line.split(' ')[2])]
        elif state == STATE_TEST_ACC:
            test_acc += [float(line.split(' ')[2])]
        elif state == STATE_TEST_PREC:
            test_prec += [float(line.split(' ')[2])]
        elif state == STATE_TEST_RECALL:
            test_recall += [float(line.split(' ')[2])]
        elif state == STATE_TEST_F1:
            test_f1 += [float(line.split(' ')[2])]


        state = (state+1) % 14

    # Close opend file
    fp.close()


    ### get final result
    maximum = max(valid_acc)
    max_index = [i for i in range(len(valid_acc)) if valid_acc[i] == maximum]

    print "best validating accuracy = %f" % maximum
    for i in max_index:
        #print "max_index = %d" % i
        print "best epoch = %d" % epoch[i]
        print "final training accuracy, precision, recall, f1 = %s" % str((train_acc[i], train_prec[i], train_recall[i], train_f1[i]))
        print "final validating accuracy, precision, recall, f1 = %s" % str((valid_acc[i], valid_prec[i], valid_recall[i], valid_f1[i]))
        print "final test accuracy, precision, recall, f1 = %s" % str((test_acc[i], test_prec[i], test_recall[i], test_f1[i]))

    ##### just try best test
    #maximum = max(test_acc)
    #print "(cheat)best test accuracy = %f" % maximum
    
    print '***Since the server doesn\'t support the matplotlib, I comment the plotting things out.'

    """
    ### Since the server doesn't support the matplotlib, I comment this plotting things out.
    ### However, it works greatly if the lib is installed.

    # red dashes, blue squares and green triangles
    plt.plot(epoch, train_acc, 'r--', label='training') 
    plt.plot(epoch, valid_acc, 'bs', label='validating')
    #plt.plot(epoch, test_acc, 'g^', label='testing')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')

    #plt.show()
    plt.savefig(args.result_file+'.png')
    """



