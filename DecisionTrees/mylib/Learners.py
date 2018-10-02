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
import pickle
from collections import Counter
from Queue import Queue

import numpy as np

import Tree

class ID3:
	##
	# a classifier of ID3 algorithm
	#
	# @param loglevel
	#				- log level for logging
	# @param int_range
	#				- a integer list, like [1,2,3,4,5,...]
	# @param use_threshold
	#				- True/False for using the threshold splitting
	# @param max_depth
	#				- a integer for maximum depth limitation
	# @param default_label
	#				- 1/0 default label when get_most_common_label has multiple labels counting the same 
	# @return None
	#
	def __init__(self, **kwargs):
		loglevel = logging.ERROR if 'loglevel' not in kwargs else kwargs['loglevel']
		logging.basicConfig(format='[%(levelname)s][%(name)s] %(message)s', level=loglevel)
		self.logger = logging.getLogger(__name__+'.'+self.__class__.__name__) 

		# parameters
		self.model = None
		self.int_range = None if 'int_range' not in kwargs else kwargs['int_range']
		self.use_threshold = False if 'use_threshold' not in kwargs else kwargs['use_threshold']
		self.max_depth = -1 if 'max_depth' not in kwargs else kwargs['max_depth']
		self.default_label = 1 if 'default_label' not in kwargs else kwargs['default_label']

	##
	# start training
	#
	# @param X
	#				- feature vectors of training data
	# @param y
	#				- labels of training data
	# @return None
	#
	def fit(self, X, y, **kwargs):
		attrs = set(range(X.shape[1]))
		self.model = self.create_tree(X, y, attrs, 0)
		self.model.training_samples = range(y.shape[0])

	##
	# recursive function for growing trees
	#
	# @param X
	#				- feature vectors of training data
	# @param y
	#				- labels of training data
	# @param attrs
	#				- attributes that have not been used
	# @param depth
	#				- current tree depth
	#
	# @return tree node
	#
	def create_tree(self, X, y, attrs, depth):
		self.logger.debug("---------------------------------------")
		self.logger.debug("depth = %d" % (depth))
		self.logger.debug("attrs = %s" % (str(attrs)))
		self.logger.debug("X.shape = %s" % (str(X.shape)))
		self.logger.debug("y.shape = %s" % (str(y.shape)))
		self.logger.debug("---------------------------------------")

		label_count = Counter(y)
		labels = label_count.keys()
		
		if len(attrs) == 0:	# recursive terminate
			most_common_label = self.get_most_common_label(label_count)
			self.logger.debug("create leaf(attrs is empty), label = %d" % (most_common_label))
			root_node = Tree.LeafNode(most_common_label)
		elif len(labels) == 1:	# recursive terminate
			self.logger.debug("create leaf(one type labels), label = %d" % (labels[0]))
			root_node = Tree.LeafNode(labels[0])
		elif self.max_depth > -1 and depth >= self.max_depth:	# recursive terminate
			most_common_label = self.get_most_common_label(label_count)
			self.logger.debug("create leaf(max depth), label = %d" % (most_common_label))
			root_node = Tree.LeafNode(most_common_label)
		else:
			best_attr = self.get_best_attr(X, y, attrs)

			self.logger.debug("best_attr = %d" % (best_attr))
			root_node = Tree.Node(best_attr)

			if self.use_threshold:
				self.divide_by_threshold(root_node, best_attr, X, y, attrs, depth)
			else:
				self.divide_by_value(root_node, best_attr, X, y, attrs, depth)

		return root_node

	##
	# get the best threshold
	#
	# @param best_attr_vals
	#				- values of selected attributes
	# @param y
	#				- labels of training data
	# @param attrs
	#				- attributes that have not been used
	#
	# @return best threshold value
	#
	def get_best_threshold(self, best_attr_vals, y):
		
		samples = [(best_attr_vals[i], y[i]) for i in range(best_attr_vals.shape[0])]
		samples.sort(key=lambda x: x[0])

		thresholds = []
		current = samples[0]

		for i in range(len(samples)):
			if samples[i][1] != current[1]:
				thresholds += [(samples[i][0]+samples[i-1][0])/2.0]
				current = samples[i]
				
		thresholds = set(thresholds)

		# calculate information gain for each threshold
		info_gains = {}
		for thr in thresholds:
			group_left_idx = [i for i in range(best_attr_vals.shape[0]) if best_attr_vals[i] <= thr]
			group_right_idx = [i for i in range(best_attr_vals.shape[0]) if best_attr_vals[i] > thr]
			info_gains[thr] = self.information_gain(y, [group_left_idx, group_right_idx])

		return max(info_gains.iteritems(), key=operator.itemgetter(1))[0]

	##
	# threshold splitting
	#
	# @param root_node
	#				- current node
	# @param best_attr
	#				- index of selected attributes
	# @param X
	#				- feature vectors of training data
	# @param y
	#				- labels of training data
	# @param attrs
	#				- attributes that have not been used
	# @param depth
	#				- current depth
	#
	# @return None
	#
	def divide_by_threshold(self, root_node, best_attr, X, y, attrs, depth):
		best_attr_vals = X[:, best_attr]

		best_threshold = self.get_best_threshold(best_attr_vals, y)
		root_node.set_threshold(best_threshold)
		self.logger.debug("best_threshold = %f" % (best_threshold))
		
		group_left_idx = [i for i in range(best_attr_vals.shape[0]) if best_attr_vals[i] <= best_threshold]
		group_right_idx = [i for i in range(best_attr_vals.shape[0]) if best_attr_vals[i] > best_threshold]

		for key, idx in [(True, group_left_idx), (False, group_right_idx)]:

			if len(idx) == 0:
				label_count = Counter(y)
				most_common_label = self.get_most_common_label(label_count)
				self.logger.debug("create_subleaf key = %d, label = %d" % (key, most_common_label))
				child = Tree.LeafNode(most_common_label)
			else:
				self.logger.debug("create_subtree key = %d" % (key))
				# recursive 
				child = self.create_tree(X[idx, :], 
										 y[idx], 
										 attrs - {best_attr}, 
										 depth + 1)

			child.training_samples = copy.copy(idx)	# records the samples inside
			root_node.add_child(key, child)

	##
	# value splitting
	#
	# @param root_node
	#				- current node
	# @param best_attr
	#				- index of selected attributes
	# @param X
	#				- feature vectors of training data
	# @param y
	#				- labels of training data
	# @param attrs
	#				- attributes that have not been used
	# @param depth
	#				- current depth
	#
	# @return None
	#
	def divide_by_value(self, root_node, best_attr, X, y, attrs, depth):
		label_count = Counter(y)
		best_attr_vals = X[:, best_attr]

		if self.int_range:
			best_attr_val_set = self.int_range
		else:
			best_attr_val_set = np.unique(best_attr_vals)

		for v in best_attr_val_set:
			subsample_idx = [i for i in range(best_attr_vals.shape[0]) if best_attr_vals[i] == v] 
			self.logger.debug("#samples = %s" % str(Counter(y[subsample_idx])))
			if len(subsample_idx) == 0:
				most_common_label = self.get_most_common_label(label_count)
				self.logger.debug("create_subleaf v = %d, label = %d" % (v, most_common_label))
				child = Tree.LeafNode(most_common_label)
			else:
				self.logger.debug("create_subtree v = %d" % (v))
				child = self.create_tree(X[subsample_idx, :], 
										 y[subsample_idx], 
										 attrs - {best_attr}, 
										 depth + 1)

			child.count_labels = Counter(y[subsample_idx])	
			root_node.add_child(v, child)

	##
	# majority vote
	#
	# @param label_count
	#				- dictionary of Counter of labels
	# @return most common label
	#
	def get_most_common_label(self, label_count):
		most_common_label = max(label_count.iteritems(), key=operator.itemgetter(1))[0]
		even_label = [key for key, val in label_count.iteritems() if key != most_common_label and val == label_count[most_common_label]]
		if len(even_label) > 0:
			most_common_label = self.default_label
		self.logger.debug("most_common_label = %d" % (most_common_label))
		return most_common_label

	##
	# calculate information gain and get the largest one
	#
	# @param X
	#				- feature vectors of training data
	# @param y
	#				- labels of training data
	# @param attrs
	#				- attributes that have not been used
	#
	# @return index of the best attributes
	#
	def get_best_attr(self, X, y, attrs):
		info_gains = {}
		for attr in attrs:
			attr_vals = X[:, attr]
			attr_val_set = np.unique(attr_vals)
			idx_groups = []
			for v in attr_val_set:
				idx_groups += [[i for i in range(attr_vals.shape[0]) if attr_vals[i] == v]]

			info_gains[attr] = ID3.information_gain(y, idx_groups)

		self.logger.debug("info_gains = %s" % (str(info_gains)))
		return max(info_gains.iteritems(), key=operator.itemgetter(1))[0]

	##
	# calculate information gain 
	#
	# @param y
	#				- labels of training data
	# @param idx_groups
	#				- indexes of branching groups
	#
	# @return index of the best attributes
	#
	@staticmethod
	def information_gain(y, idx_groups):
		# entire entropy
		label_probs = ID3.get_label_probs(y)
		n = y.shape[0]
		s_ent = ID3.entropy(label_probs)

		# expected entropy for partitioning
		e_ent = 0.0
		
		for idxs in idx_groups:
			label_probs = ID3.get_label_probs(y[idxs])
			n_v = y[idxs].shape[0]
			e_ent += ((n_v / float(n)) * ID3.entropy(label_probs))
		return s_ent - e_ent

	##
	# calculate entropy 
	#
	# @param label_probs
	#				- dictionary of probabilities of each label
	#
	# @return entropy
	#
	@staticmethod
	def entropy(label_probs):
		ent = 0.0
		for p in label_probs:
			ent -= (p * math.log(p, 2))
		return ent

	##
	# get probabilities of each label
	#
	# @param y
	#				- labels of training data
	#
	# @return dictionary of probabilities of each label
	#
	@staticmethod
	def get_label_probs(y):
		label_count = Counter(y)
		n = y.shape[0]
		return [c/float(n)  for c in label_count.values()] 

	##
	# print out the tree
	#
	# @return None
	#
	def print_tree(self):
		Tree.print_tree(self.model)

	##
	# tree pruning
	#
	# @param X
	#				- feature vectors of training data
	# @param y
	#				- labels of training data
	# @param X_validation
	#				- feature vectors of validating data
	# @param y_validation
	#				- labels of validating data
	#
	# @return None
	#
	def prune_tree(self, X, y, X_validation, y_validation):
		self.X_validation = X_validation
		self.y_validation = y_validation
		self._prune(None, None, self.model, X, y)
			
	##
	# tree pruning subfunction
	#
	# @param parent
	#				- parent node
	# @param cond
	#				- key of parent node dictionary
	# @param current_node
	#				- current node
	# @param X
	#				- feature vectors of training data
	# @param y
	#				- labels of training data
	#
	# @return None
	#
	def _prune(self, parent, cond, current_node, X, y):
		if not current_node.is_leaf:
			for key, val in current_node.branches.iteritems():
				subsample_idx = [i for i in range(X.shape[0]) if current_node.get_key_by_value(X[i][current_node.attr]) == key]
				self._prune(current_node, key, val, X[subsample_idx, :], y[subsample_idx])

			if parent is not None and cond is not None:
				# get pre-prune accuracy
				orig_accuracy = self.score(self.X_validation, self.y_validation)

				# try replace this subtree with a leaf
				label_count = Counter(y)
				most_common_label = self.get_most_common_label(label_count)
				parent.branches[cond] = Tree.LeafNode(most_common_label)

				# get post-prune accuracy
				new_accuracy = self.score(self.X_validation, self.y_validation)
				self.logger.debug("orig_accuracy = %f, new_accuracy = %f" % (orig_accuracy, new_accuracy))

				# if worse than original one, replace back
				if new_accuracy < orig_accuracy:
					parent.branches[cond] = current_node
				else:
					self.logger.debug("pruned")

	##
	# get predicting labels
	#
	# @param X_test
	#				- feature vectors of testing data
	#
	# @return None
	#
	def predict(self, X_test):
		y_predict = []
		for x in X_test:
			current_node = self.model
			while True:
				if current_node.is_leaf:
					y_predict += [current_node.label]
					break
				else:
					if self.use_threshold:
						current_node = current_node.branches[current_node.get_key_by_value(x[current_node.attr])]
					else:
						current_node = current_node.branches[x[current_node.attr]]

		return np.array(y_predict)

	##
	# get accuracy
	#
	# @param X_test
	#				- feature vectors of testing data
	# @param y_test
	#				- labels of testing data
	#
	# @return None
	#
	def score(self, X_test, y_test):
		y_predict = self.predict(X_test)
		correct = (y_predict == y_test)
		error_sample_index = [i for i in range(correct.shape[0]) if correct[i] == False]
		self.logger.debug("error sample index = %s" % (str(error_sample_index)))

		return sum(correct) / float(y_test.shape[0])

	##
	# get F-1 score
	#
	# @param X_test
	#				- feature vectors of testing data
	# @param y_test
	#				- labels of testing data
	#
	# @return None
	#
	def f1_score(self, X_test, y_test):
		y_predict = self.predict(X_test)
		tp = [i for i in range(len(y_test)) for yp in y_predict if y_test[i]==1 and yp==1]
		fp = [i for i in range(len(y_test)) for yp in y_predict if y_test[i]==0 and yp==1]
		fn = [i for i in range(len(y_test)) for yp in y_predict if y_test[i]==1 and yp==0]

		recall = float(len(tp)) / (len(tp)+len(fn))
		precision = float(len(tp)) / (len(tp)+len(fp))

		return (2 * precision * recall) / (precision + recall)

	##
	# get dump pickle file
	#
	# @param filename
	#				- file name
	#
	# @return None
	#
	def dump(self, filename):
		pickle.dump(self, filename)

