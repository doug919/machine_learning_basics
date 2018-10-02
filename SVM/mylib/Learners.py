##
# Learners.py 
#
# CS 578 - HW1
# Fall 2015 @ Purdue University
#
# @author I-Ta Lee
#
# @date 09/10/2015
#

import math
import copy
import logging
import operator
import cPickle
import timeit
from collections import Counter
from Queue import Queue

import numpy as np

class LearnerBase(object):
    def __init__(self, **kwargs):
        loglevel = logging.ERROR if 'loglevel' not in kwargs else kwargs['loglevel']
        logging.basicConfig(format='[%(levelname)s][%(name)s] %(message)s', level=loglevel)
        self.logger = logging.getLogger(__name__+'.'+self.__class__.__name__) 

        # parameters
        self.model = None

    ##
    # get score
    #
    # @param X_test
    #               - feature vectors of testing data
    # @param y_test
    #               - labels of testing data
    # @param accuracy
    #               - (optional) get accuracy (Default: True)
    # @param precision
    #               - (optional) get precision (Default: False)
    # @param recall
    #               - (optional) get recall (Default: False)
    # @param f1
    #               - (optional) get f1 score (Default: False)
    # @return None
    #
    def score(self, X_test, y_test, **kwargs):
        do_accuracy = True if 'accuracy' not in kwargs else kwargs['accuracy']
        do_precision = False if 'precision' not in kwargs else kwargs['precision']
        do_recall = False if 'recall' not in kwargs else kwargs['recall']
        do_f1 = False if 'f1' not in kwargs else kwargs['f1']

        y_predict = self.predict(X_test)
        correct = (y_predict == y_test)
        error_sample_index = [i for i in range(correct.shape[0]) if correct[i] == False]
        self.logger.debug("error sample index = %s" % (str(error_sample_index)))

        ret = {}
        if do_accuracy:
            ret['accuracy'] = sum(correct) / float(y_test.shape[0])

        precision = LearnerBase.precision(y_predict, y_test)
        recall = LearnerBase.recall(y_predict, y_test)
        if do_precision:
            ret['precision'] = precision
        if do_recall:
            ret['recall'] = recall
        if do_f1:
            ret['f1'] = LearnerBase.f1_score(precision, recall)

        return ret

    ##
    # get precision
    #
    # @param y_predict
    #               - predicted labels of testing data
    # @param y_test
    #               - labels of testing data
    #
    # @return None
    #
    @staticmethod
    def precision(y_predict, y_test):
        tp = [i for i in range(len(y_test)) if y_test[i]>0 and y_predict[i]>0]
        fp = [i for i in range(len(y_test)) if y_test[i]<=0 and y_predict[i]>0]

        #assert (len(tp)+len(fp) != 0)
        if len(tp)+len(fp) != 0:
            precision = float(len(tp)) / (len(tp)+len(fp))
        else:
            logging.debug("Both true positives and false positives are zero, so we cannot calculate the precision.")
            precision = -1
        return precision

    ##
    # get recall
    #
    # @param y_predict
    #               - predicted labels of testing data
    # @param y_test
    #               - labels of testing data
    #
    # @return None
    #
    @staticmethod
    def recall(y_predict, y_test):
        tp = [i for i in range(len(y_test)) if y_test[i]>0 and y_predict[i]>0]
        fn = [i for i in range(len(y_test)) if y_test[i]>0 and y_predict[i]<=0]
        
        #assert (len(tp)+len(fn) != 0)

        if len(tp)+len(fn) != 0:
            recall = float(len(tp)) / (len(tp)+len(fn))
        else:
            logging.debug("Both true positives and false negatives are zero, so we cannot calculate the precision.")
            recall = -1

        
        return recall

    ##
    # get F-1 score
    #
    # @param precision
    #               - calculated by precision()
    # @param recall
    #               - calculated by recall()
    #
    # @return None
    #
    @staticmethod
    def f1_score(precision, recall):
        return (2.0 * precision * recall) / (precision + recall)

    ##
    # get dump pickle file
    #
    # @param filename
    #               - file name
    #
    # @return None
    #
    def dump(self, filename):
        cPickle.dump(self, open(filename, 'wb'))


class LinearSvm(LearnerBase):
    def __init__(self, **kwargs):
        super(LinearSvm, self).__init__(**kwargs)
        

    def fit(self, X, y, **kwargs):
        # params
        self.regularize = 'l1' if 'regularize' not in kwargs else kwargs['regularize']
        self.max_iterations = 10 if 'max_iterations' not in kwargs else kwargs['max_iterations'] 
        self.step_size = 0.1 if 'step_size' not in kwargs else kwargs['step_size'] 
        self.lmbd = 0.1 if 'lmbd' not in kwargs else kwargs['lmbd']

        assert self.regularize in ['l1', 'l2']
        assert self.max_iterations > 0
        assert self.step_size > 0 and self.step_size <= 1
        assert self.lmbd > 0 and self.lmbd <= 1

        self.model = self._fit(X, y)

    def _fit(self, X, y):
        # w_0 for bias
        w = np.zeros(X.shape[1])
        for it in range(self.max_iterations):
            self.logger.debug("iteration %d" % (it))
            gradient = self._l1_reg_gradient(X, y, w) if self.regularize == 'l1' else self._l2_reg_gradient(X, y, w)
            w = w + (self.step_size * gradient)

        return w

    def _l1_reg_gradient(self, X, y, w):
        n_dim = X.shape[1]
        gradient = np.zeros(n_dim)
        for sample_idx in range(X.shape[0]):
            X_cur = X[sample_idx]
            y_cur = y[sample_idx]
            if y_cur * np.dot(w, X_cur) <= 1:
                gradient = gradient + (y_cur * X_cur)
        
        temp_regularization = self.lmbd * np.array([1] * n_dim)
        temp_regularization[0] = 0      # bias should not be regularized

        return gradient - temp_regularization

    def _l2_reg_gradient(self, X, y, w):
        n_dim = X.shape[1]
        gradient = np.zeros(n_dim)
        for sample_idx in range(X.shape[0]):

            X_cur = X[sample_idx]
            y_cur = y[sample_idx]
            if y_cur * np.dot(w, X_cur) <= 1:
                gradient = gradient + (y_cur * X_cur)

        temp_regularization = self.lmbd * w
        temp_regularization[0] = 0     # bias should not be regularized

        return gradient - temp_regularization

    def predict_one(self, x):
        assert self.model is not None

        y_pred = -1 if np.dot(self.model, x) <= 0 else 1  
        
        return y_pred

    def predict(self, X_test):
        assert self.model is not None

        y_predict = []
        for x in X_test:        
            y_predict += [self.predict_one(x)]

        return np.array(y_predict)

