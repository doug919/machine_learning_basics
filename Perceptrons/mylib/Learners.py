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

        assert (len(tp)+len(fp) != 0)
        precision = float(len(tp)) / (len(tp)+len(fp))
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
        
        assert (len(tp)+len(fn) != 0)
        recall = float(len(tp)) / (len(tp)+len(fn))
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

class DualPerceptron(LearnerBase):
    def __init__(self, **kwargs):
        super(DualPerceptron, self).__init__(**kwargs)

    @staticmethod
    def linear_kernel(X):
        n = X.shape[0]
        k = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                k[i][j] = np.dot(X[i], X[j]) 
        return k

    def fit(self, X, y, max_iter, **kwargs):
        assert 'kernel' in kwargs
        kernel = kwargs['kernel']

        self.logger.debug("start training")
        start_time = timeit.default_timer()

        n = X.shape[0]
        alphas = np.zeros(n)

        for i in range(max_iter):
            self.logger.debug("start epoch %d" % i)
            DualPerceptron._fit(X, y, alphas, kernel)

        self.model = X, alphas
        self.logger.debug("training elapses: %f s" % (timeit.default_timer() - start_time))

    def fit_one_epoch(self, X, y, **kwargs):
        assert 'kernel' in kwargs
        kernel = kwargs['kernel']

        self.logger.debug("start training")
        start_time = timeit.default_timer()

        n = X.shape[0]
        alphas = np.zeros(n)    

        DualPerceptron._fit(X, y, alphas, kernel)

        self.model = X, alphas
        self.logger.debug("training elapses: %f s" % (timeit.default_timer() - start_time))

    @staticmethod
    def _fit(X, y, alphas, kernel):  
        for example_idx in range(X.shape[0]):
            X_cur = X[example_idx]
            y_cur = y[example_idx]
            y_pred = sum(alphas * kernel[example_idx])  
            if y_cur * y_pred <= 0:
                alphas[example_idx] += y_cur 

    def predict(self, X_test):
        assert self.model is not None
        y_predict = []
        X = self.model[0]
        alphas = self.model[1]

        for i in range(X_test.shape[0]):
            s = 0
            for j in range(X.shape[0]):
                s += (alphas[j] * np.dot(X[j], X_test[i]))
     
            pred = -1 if np.sign(s) <= 0 else 1
            y_predict += [pred]

        return np.array(y_predict)


class Perceptron(LearnerBase):
    def __init__(self, **kwargs):
        super(Perceptron, self).__init__(**kwargs)
        self.do_avg = False if 'average' not in kwargs else kwargs['average']
        self.logger.debug("do_avg = %d", self.do_avg)

    def fit(self, X, y, max_iter, **kwargs):
        learning_rate = 1.0 if 'learning_rate' not in kwargs else kwargs['learning_rate']

        self.logger.debug("start training")
        start_time = timeit.default_timer()

        # bias in the weights[0]
        weights = np.array([0] * X.shape[1])

        # if self.do_avg is true, we do Averaged Perceptron; else we do Perceptron
        if self.do_avg:
            self.total_weights = np.array([0] * X.shape[1]) if do_clear or self.model is None else self.total_weights
            self.total_count = [0] if do_clear or self.model is None else self.total_count
        else:
            self.total_weights = None
            self.total_count = [0]

        for i in range(max_iter):
            self.logger.debug("start epoch %d" % i)
            Perceptron._fit(X, y, weights, learning_rate, total_weights=self.total_weights, total_count=self.total_count)

        self.model = weights 
        self.logger.debug("training elapses: %f s" % (timeit.default_timer() - start_time))

    def fit_one_epoch(self, X, y, **kwargs):
        learning_rate = 1.0 if 'learning_rate' not in kwargs else kwargs['learning_rate']
        do_clear = False if 'clear' not in kwargs else kwargs['clear']

        self.logger.debug("start training")
        start_time = timeit.default_timer()

        # bias in the weights[0]
        weights = np.array([0] * X.shape[1]) if do_clear or self.model is None else self.model
        if self.do_avg:
            self.total_weights = np.array([0] * X.shape[1]) if do_clear or self.model is None else self.total_weights
            self.total_count = [0] if do_clear or self.model is None else self.total_count
        else:
            self.total_weights = None
            self.total_count = [0]
        
        Perceptron._fit(X, y, weights, learning_rate, total_weights=self.total_weights, total_count=self.total_count)

        self.model = weights 
        self.logger.debug("training elapses: %f s" % (timeit.default_timer() - start_time))

    @staticmethod
    def _fit(X, y, weights, learning_rate, **kwargs):   
        for example_idx in range(X.shape[0]):
            X_cur = X[example_idx]
            y_cur = y[example_idx]
            y_pred = np.sign(sum(weights * X_cur))  
            if y_cur * y_pred <= 0:
                weights += learning_rate*y_cur*X_cur

            if 'total_weights' in kwargs and kwargs['total_weights'] is not None:
                kwargs['total_weights'] += weights
                kwargs['total_count'][0] += 1

    def predict(self, X_test):
        assert self.model is not None
        y_predict = []

        for x in X_test:        
            y_predict += [self.predict_one(x)]
                
        return np.array(y_predict)

    def predict_one(self, x):
        if self.do_avg:
            model = self.total_weights / float(self.total_count[0])
        else:
            model = self.model    
        y_pred = -1 if np.sign(sum(model * x)) <= 0 else 1  
        return y_pred


class Winnow(LearnerBase):
    ##
    # a classifier of Winnow algorithm
    #
    # @param loglevel
    #               - log level for logging
    # @return None
    #
    def __init__(self, **kwargs):
        super(Winnow, self).__init__(**kwargs)

    ##
    # start training
    #
    # @param X
    #               - feature vectors of training data
    # @param y
    #               - labels of training data
    # @param max_iter
    #               - max iteration
    # @param alpha
    #               - (optional) promotion and demotion parameter alpha in winnow
    # @return None
    #
    def fit(self, X, y, max_iter, **kwargs):

        self.logger.debug("start training")
        start_time = timeit.default_timer()

        self.threshold = X.shape[1]    # threshold is number of features
        alpha = 2.0 if 'alpha' not in kwargs else kwargs['alpha']
        weights = np.array([1] * X.shape[1])

        for epoch in range(max_iter):
            self.logger.debug("start epoch %d" % epoch)
            Winnow._fit(X, y, weights, self.threshold, alpha)
            import pdb; pdb.set_trace() 

        self.model = weights
        self.logger.debug("training elapses: %f s" % (timeit.default_timer() - start_time))

    def fit_one_epoch(self, X, y, **kwargs):
        self.logger.debug("start training")
        start_time = timeit.default_timer()

        self.threshold = X.shape[1]    # threshold is number of features
        alpha = 2.0 if 'alpha' not in kwargs else kwargs['alpha']
        do_clear = False if 'clear' not in kwargs else kwargs['clear']
        weights = np.array([1] * X.shape[1]) if do_clear or self.model is None else self.model

        Winnow._fit(X, y, weights, self.threshold, alpha)

        self.model = weights
        self.logger.debug("training elapses: %f s" % (timeit.default_timer() - start_time))

    @staticmethod
    def _fit(X, y, weights, threshold, alpha):
        for example_idx in range(X.shape[0]):
            X_cur = X[example_idx]
            y_cur = y[example_idx]
            y_pred = 1 if sum(weights * X_cur) >= threshold else -1

            if y_cur*y_pred <= 0:
                act_idx = [i for i in range(X_cur.shape[0]) if X_cur[i] > 0]
                if y_cur > 0:
                    weights[act_idx] *= alpha
                else:
                    weights[act_idx] /= alpha

    ##
    # get predicting labels
    #
    # @param X_test
    #               - feature vectors of testing data
    #
    # @return None
    #
    def predict(self, X_test):
        assert self.model is not None
        y_predict = []

        for x in X_test:   
            y_predict += [self.predict_one(x)]
                
        return np.array(y_predict)

    def predict_one(self, x):
        y_pred = 1 if sum(self.model * x) >= self.threshold else -1  
        return y_pred

