import sys
import logging
import time
import json

from mylib.Features import FeatureSet 
from mylib import Learners
from mylib import Utils

# import current settings as a global variable in this module
g_setting = Utils.Setting("settings.json")


# golden area
def parseArgs(args):
    """ 
        Parses arguments vector, looking for switches of the form -key {optional value}.
        For example:
        parseArgs([ 'template.py', '-i', '10', '-r', 'l1', '-s', '0.4', '-l', '0.5', '-f', '1' ]) = {'-i':'10', '-r':'l1', '-s:'0.4', '-l':'0.5', '-f':1 }
    """
    args_map = {}
    curkey = None
    for i in xrange(1, len(args)):
        if args[i][0] == '-':
            args_map[args[i]] = True
            curkey = args[i]
        else:
            assert curkey
            args_map[curkey] = args[i]
            curkey = None
    return args_map

def validateInput(args):
    args_map = parseArgs(args)

    maxIterations = 10 # the maximum number of iterations. should be a positive integer
    regularization = 'l1' # 'l1' or 'l2'
    stepSize = 0.1 # 0 < stepSize <= 1
    lmbd = 0.1 # 0 < lmbd <= 1
    featureSet = 1 # 1: original attribute, 2: pairs of attributes, 3: both

    if '-i' in args_map:
      maxIterations = int(args_map['-i'])
    if '-r' in args_map:
      regularization = args_map['-r']
    if '-s' in args_map:
      stepSize = float(args_map['-s'])
    if '-l' in args_map:
      lmbd = float(args_map['-l'])
    if '-f' in args_map:
      featureSet = int(args_map['-f'])

    assert maxIterations > 0
    assert regularization in ['l1', 'l2']
    assert stepSize > 0 and stepSize <= 1
    assert lmbd > 0 and lmbd <= 1
    assert featureSet in [1, 2, 3]
    
    return [maxIterations, regularization, stepSize, lmbd, featureSet]

# implementation starts
def getFeatureSet(featureSet, load_feature=None, dump_feature=None):
    """
        load_feature: feature file name
    """
    global g_setting

    ### feature extraction
    feature_set = FeatureSet(loglevel=g_setting.get_loglevel())

    # if load_feature_path is specified, we ignore the unigram/bigram option and just load the feature file
    load_feature_path = g_setting.get_load_feature_path()
    if load_feature_path != "":
        # load from file
        feature_set.load(load_feature_path)
    else:
        # feature extraction
        feature_select = {
            1: [FeatureSet.FEATURE_UNIGRAM],
            2: [FeatureSet.FEATURE_BIGRAM],
            3: [FeatureSet.FEATURE_UNIGRAM, FeatureSet.FEATURE_BIGRAM],
        } [featureSet]
        raw_path = {}
        raw_path['train'] = g_setting.get_training_data_path()
        raw_path['validate'] = g_setting.get_validating_data_path()
        raw_path['test'] = g_setting.get_testing_data_path()
        feature_set.extract_feature(raw_path, 
                                    feature_select, 
                                    unigram_tf_threshold=g_setting.get_unigram_term_freq_threshold(), 
                                    bigram_tf_threshold=g_setting.get_bigram_term_freq_threshold())

    # dump file if specified
    dump_feature_path = g_setting.get_dump_feature_path()
    if dump_feature_path != "":
        feature_set.dump(dump_feature_path)

    return feature_set.X_train, feature_set.y_train, feature_set.X_validate, feature_set.y_validate, feature_set.X_test, feature_set.y_test

## Gradient Descent Algorithm
def GD(maxIterations, regularization, stepSize, lmbd, featureSet):
    """

        parameters:
            maxIterations:      interger, number of iterations 
            regularization:     'l1' or 'l2'
            stepSize:           real number
            lmbd:               real number, hyperparameter
            featureSet:         1, 2, 3 for unigram, bigram, both

        return:
            accuracy, precision, recall, f1 
    """
    global g_setting

    # get the selected feature tables
    feature_extraction_start_time = time.time()
    X_train, y_train, X_validate, y_validate, X_test, y_test = getFeatureSet(featureSet)
    print "feature extraction takes %f seconds." % (time.time() - feature_extraction_start_time)
    
    training_start_time = time.time()
    learner = Learners.LinearSvm(loglevel=g_setting.get_loglevel())
    learner.fit(X_train, y_train, regularize=regularization, max_iterations=maxIterations, step_size=stepSize, lmbd=lmbd)
    print "training takes %f seconds." % (time.time() - training_start_time)

    train_result = learner.score(X_train, y_train, 
                                 accuracy=True, 
                                 precision=True, 
                                 recall=True, 
                                 f1=True)
    print "training accuracy: %f" % (train_result['accuracy'])
    print "training precision: %f" % (train_result['precision'])
    print "training recall: %f" % (train_result['recall'])
    print "training f1: %f" % (train_result['f1'])

    validate_result = learner.score(X_validate, y_validate, 
                                    accuracy=True, 
                                    precision=True, 
                                    recall=True, 
                                    f1=True)
    print "validating accuracy: %f" % (validate_result['accuracy'])
    print "validating precision: %f" % (validate_result['precision'])
    print "validating recall: %f" % (validate_result['recall'])
    print "validating f1: %f" % (validate_result['f1'])

    test_result = learner.score(X_test, y_test, 
                                accuracy=True, 
                                precision=True, 
                                recall=True, 
                                f1=True)
    print "testing accuracy: %f" % (test_result['accuracy'])
    print "testing precision: %f" % (test_result['precision'])
    print "testing recall: %f" % (test_result['recall'])
    print "testing f1: %f" % (test_result['f1'])
    print '--------------------------------------------------------------'


# main
# ----
# The main program loop
# You should modify this function to run your experiments

def main():
    arguments = validateInput(sys.argv)
    maxIterations, regularization, stepSize, lmbd, featureSet = arguments
    print maxIterations, regularization, stepSize, lmbd, featureSet

    GD(maxIterations, regularization, stepSize, lmbd, featureSet)

if __name__ == '__main__':
    main()
