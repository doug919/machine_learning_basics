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
import csv

from collections import OrderedDict

import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.font_manager import FontProperties



STATE_PARAMS = 0
STATE_FEATURE_TIME = 1
STATE_TRAINING_TIME = 2
STATE_TRAIN_ACC = 3
STATE_TRAIN_PREC = 4
STATE_TRAIN_RECALL = 5
STATE_TRAIN_F1 = 6
STATE_VALID_ACC = 7
STATE_VALID_PREC = 8
STATE_VALID_RECALL = 9
STATE_VALID_F1 = 10
STATE_TEST_ACC = 11
STATE_TEST_PREC = 12
STATE_TEST_RECALL = 13
STATE_TEST_F1 = 14
STATE_SEP_LINE = 15

def get_arguments(argv):
    parser = argparse.ArgumentParser(description='This program processes the stdout of hw3.py, collecting the results and generate csv.')
    parser.add_argument('result_folder', metavar='FOLDER_PATH', 
                        help='a file containing the results')
    parser.add_argument('out_csv', metavar='FILE_NAME',
                        help='output csv file name')
    parser.add_argument('-o', '--output_figure', metavar='FILE_NAME', default=None,
                        help='output png figure')
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

    # list files in the folder
    fnames = os.listdir(args.result_folder)

    csv_rows = []
    for fn in fnames:
        if fn.startswith("."):
            continue

        filepath = os.path.join(args.result_folder, fn)
        logger.info("file=%s" % filepath)

        with open(filepath) as f:
            one_exp = OrderedDict()
            state = STATE_PARAMS
            for line in f:
                if state == STATE_PARAMS:
                    one_exp['param'] = line.rstrip()
                elif state == STATE_FEATURE_TIME:
                    pass
                elif state == STATE_TRAINING_TIME:
                    pass
                elif state == STATE_TRAIN_ACC:
                    one_exp['train_acc'] = float(line.split(' ')[2])
                elif state == STATE_TRAIN_PREC:
                    one_exp['train_prec'] = float(line.split(' ')[2])
                elif state == STATE_TRAIN_RECALL:
                    one_exp['train_recall'] = float(line.split(' ')[2])
                elif state == STATE_TRAIN_F1:
                    one_exp['train_f1'] = float(line.split(' ')[2])
                elif state == STATE_VALID_ACC:
                    one_exp['valid_acc'] = float(line.split(' ')[2])
                elif state == STATE_VALID_PREC:
                    one_exp['valid_prec'] = float(line.split(' ')[2])
                elif state == STATE_VALID_RECALL:
                    one_exp['valid_recall'] = float(line.split(' ')[2])
                elif state == STATE_VALID_F1:
                    one_exp['valid_f1'] = float(line.split(' ')[2])
                elif state == STATE_TEST_ACC:
                    one_exp['test_acc'] = float(line.split(' ')[2])
                elif state == STATE_TEST_PREC:
                    one_exp['test_prec'] = float(line.split(' ')[2])
                elif state == STATE_TEST_RECALL:
                    one_exp['test_recall'] = float(line.split(' ')[2])
                elif state == STATE_TEST_F1:
                    one_exp['test_f1'] = float(line.split(' ')[2])

                state = (state+1) % 16

        logger.info("%s" % str(one_exp))
        csv_rows += [one_exp]

    with open(args.out_csv, 'w') as csvfile:
        fieldnames = csv_rows[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for one_exp in csv_rows:
            writer.writerow(one_exp)

    ### Since the server doesn't support the matplotlib, I comment this plotting things out.
    ### However, it works greatly if the lib is installed.
    """
    if args.output_figure is not None:
        valid_acc_stepsize01 = [one_exp['valid_acc'] for one_exp in csv_rows if one_exp.has_key('param') and float(one_exp['param'].split(' ')[2]) == 0.1]
        x_axis_stepsize01 = [float(one_exp['param'].split(' ')[3]) for one_exp in csv_rows if one_exp.has_key('param') and float(one_exp['param'].split(' ')[2]) == 0.1]

        valid_acc_stepsize03 = [one_exp['valid_acc'] for one_exp in csv_rows if one_exp.has_key('param') and float(one_exp['param'].split(' ')[2]) == 0.3]
        x_axis_stepsize03 = [float(one_exp['param'].split(' ')[3]) for one_exp in csv_rows if one_exp.has_key('param') and float(one_exp['param'].split(' ')[2]) == 0.3]

        valid_acc_stepsize05 = [one_exp['valid_acc'] for one_exp in csv_rows if one_exp.has_key('param') and float(one_exp['param'].split(' ')[2]) == 0.5]
        x_axis_stepsize05 = [float(one_exp['param'].split(' ')[3]) for one_exp in csv_rows if one_exp.has_key('param') and float(one_exp['param'].split(' ')[2]) == 0.5]

        valid_acc_stepsize07 = [one_exp['valid_acc'] for one_exp in csv_rows if one_exp.has_key('param') and float(one_exp['param'].split(' ')[2]) == 0.7]
        x_axis_stepsize07 = [float(one_exp['param'].split(' ')[3]) for one_exp in csv_rows if one_exp.has_key('param') and float(one_exp['param'].split(' ')[2]) == 0.7]

        valid_acc_stepsize09 = [one_exp['valid_acc'] for one_exp in csv_rows if one_exp.has_key('param') and float(one_exp['param'].split(' ')[2]) == 0.9]
        x_axis_stepsize09 = [float(one_exp['param'].split(' ')[3]) for one_exp in csv_rows if one_exp.has_key('param') and float(one_exp['param'].split(' ')[2]) == 0.9]

        #import pdb; pdb.set_trace()

        # red dashes, blue squares and green triangles
        plt.plot(x_axis_stepsize01, valid_acc_stepsize01, label='stepSize=0.1') 
        plt.plot(x_axis_stepsize03, valid_acc_stepsize03, label='stepSize=0.3') 
        plt.plot(x_axis_stepsize05, valid_acc_stepsize05, label='stepSize=0.5') 
        plt.plot(x_axis_stepsize07, valid_acc_stepsize07, label='stepSize=0.7') 
        plt.plot(x_axis_stepsize09, valid_acc_stepsize09, label='stepSize=0.9') 

        plt.xlabel('lmbd')
        plt.ylabel('validation accuracy')
        plt.legend(loc='lower right')

        #plt.show()
        plt.savefig(args.output_figure)
    

    """
    """
    if args.output_figure is not None:
        valid_acc_l1_f1 = [one_exp['valid_acc'] for one_exp in csv_rows if one_exp.has_key('param') and one_exp['param'].split(' ')[1] == 'l1' and one_exp['param'].split(' ')[4] == '1']
        x_axis_l1_f1 = [int(one_exp['param'].split(' ')[0]) for one_exp in csv_rows if one_exp.has_key('param') and one_exp['param'].split(' ')[1] == 'l1' and one_exp['param'].split(' ')[4] == '1']

        valid_acc_l1_f2 = [one_exp['valid_acc'] for one_exp in csv_rows if one_exp.has_key('param') and one_exp['param'].split(' ')[1] == 'l1' and one_exp['param'].split(' ')[4] == '2']
        x_axis_l1_f2 = [int(one_exp['param'].split(' ')[0]) for one_exp in csv_rows if one_exp.has_key('param') and one_exp['param'].split(' ')[1] == 'l1' and one_exp['param'].split(' ')[4] == '2']

        valid_acc_l1_f3 = [one_exp['valid_acc'] for one_exp in csv_rows if one_exp.has_key('param') and one_exp['param'].split(' ')[1] == 'l1' and one_exp['param'].split(' ')[4] == '3']
        x_axis_l1_f3 = [int(one_exp['param'].split(' ')[0]) for one_exp in csv_rows if one_exp.has_key('param') and one_exp['param'].split(' ')[1] == 'l1' and one_exp['param'].split(' ')[4] == '3']

        valid_acc_l2_f1 = [one_exp['valid_acc'] for one_exp in csv_rows if one_exp.has_key('param') and one_exp['param'].split(' ')[1] == 'l2' and one_exp['param'].split(' ')[4] == '1']
        x_axis_l2_f1 = [int(one_exp['param'].split(' ')[0]) for one_exp in csv_rows if one_exp.has_key('param') and one_exp['param'].split(' ')[1] == 'l2' and one_exp['param'].split(' ')[4] == '1']

        valid_acc_l2_f2 = [one_exp['valid_acc'] for one_exp in csv_rows if one_exp.has_key('param') and one_exp['param'].split(' ')[1] == 'l2' and one_exp['param'].split(' ')[4] == '2']
        x_axis_l2_f2 = [int(one_exp['param'].split(' ')[0]) for one_exp in csv_rows if one_exp.has_key('param') and one_exp['param'].split(' ')[1] == 'l2' and one_exp['param'].split(' ')[4] == '2']

        valid_acc_l2_f3 = [one_exp['valid_acc'] for one_exp in csv_rows if one_exp.has_key('param') and one_exp['param'].split(' ')[1] == 'l2' and one_exp['param'].split(' ')[4] == '3']
        x_axis_l2_f3 = [int(one_exp['param'].split(' ')[0]) for one_exp in csv_rows if one_exp.has_key('param') and one_exp['param'].split(' ')[1] == 'l2' and one_exp['param'].split(' ')[4] == '3']


        temp_tuplist = sorted(zip(valid_acc_l1_f1, x_axis_l1_f1), key=lambda x: x[1])
        valid_acc_l1_f1 = [x for x, y in temp_tuplist]
        x_axis_l1_f1 = [y for x, y in temp_tuplist]

        temp_tuplist = sorted(zip(valid_acc_l1_f2, x_axis_l1_f2), key=lambda x: x[1])
        valid_acc_l1_f2 = [x for x, y in temp_tuplist]
        x_axis_l1_f2 = [y for x, y in temp_tuplist]

        temp_tuplist = sorted(zip(valid_acc_l1_f3, x_axis_l1_f3), key=lambda x: x[1])
        valid_acc_l1_f3 = [x for x, y in temp_tuplist]
        x_axis_l1_f3 = [y for x, y in temp_tuplist]

        temp_tuplist = sorted(zip(valid_acc_l2_f1, x_axis_l2_f1), key=lambda x: x[1])
        valid_acc_l2_f1 = [x for x, y in temp_tuplist]
        x_axis_l2_f1 = [y for x, y in temp_tuplist]

        temp_tuplist = sorted(zip(valid_acc_l2_f2, x_axis_l2_f2), key=lambda x: x[1])
        valid_acc_l2_f2 = [x for x, y in temp_tuplist]
        x_axis_l2_f2 = [y for x, y in temp_tuplist]

        temp_tuplist = sorted(zip(valid_acc_l2_f3, x_axis_l2_f3), key=lambda x: x[1])
        valid_acc_l2_f3 = [x for x, y in temp_tuplist]
        x_axis_l2_f3 = [y for x, y in temp_tuplist]

        # red dashes, blue squares and green triangles
        plt.plot(x_axis_l1_f1, valid_acc_l1_f1, label='L1,F1') 
        plt.plot(x_axis_l1_f2, valid_acc_l1_f2, label='L1,F2') 
        plt.plot(x_axis_l1_f3, valid_acc_l1_f3, label='L1,F3') 
        plt.plot(x_axis_l2_f1, valid_acc_l2_f1, label='L2,F1') 
        plt.plot(x_axis_l2_f2, valid_acc_l2_f2, label='L2,F2') 
        plt.plot(x_axis_l2_f3, valid_acc_l2_f3, label='L2,F3') 

        plt.xlabel('lmbd')
        plt.ylabel('validation accuracy')
        plt.legend(loc='lower right')

        #plt.show()
        plt.savefig(args.output_figure)
    """
