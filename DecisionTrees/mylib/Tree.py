##
# Utils.py
# 
# CS 578 - HW1
# Fall 2015 @ Purdue University
#
# @author I-Ta Lee
#
# @date 09/10/2015
#

##
# print out the tree
#
# @param root_node
#				- root of tree
# @param depth
#				- the depth of the root
# @return None
#
def print_tree(root_node, depth=0):
	if root_node.is_leaf:
		print "\t" * depth, "(%d)" % (root_node.label)
	else:
		print "\t" * depth, "[attr = %d]" % (root_node.attr)
		if depth > 0:
			if root_node.count_labels.has_key(1):
				print "\t" * depth, "[+: %d]" % (root_node.count_labels[1])
			if root_node.count_labels.has_key(0):
				print "\t" * depth, "[-: %d]" % (root_node.count_labels[0])
		if root_node.threshold is not None:
			print "\t" * depth, "[val <= %.2f]" % (root_node.threshold)

		for key, val in root_node.branches.iteritems():
			print "\t" * depth, key
			print_tree(val, depth+1)


class Node:
	##
	# This class represents a tree node
	#
	# @param attr
	#				- selected node attribute
	# @return None
	#
	def __init__(self, attr):
		self.is_leaf = False
		self.attr = attr
		self.threshold = None
		self.branches = {}			# we use dictionary data structure to build the tree
		self.count_labels = {}

	##
	# set the threshold value for the node
	#
	# @param threshold
	#				- selected threshold value
	# @return None
	#
	def set_threshold(self, threshold):
		self.threshold = threshold

	##
	# for threshold splitting we need a conversion from 
	# the conditions of threshold comparisons to the
	# dictionary key
	#
	# @param val
	#				- value to be compared to the threshold
	# @return None
	#
	def get_key_by_value(self, val):
		if self.threshold is None:
			key = val
		else:
			key = (val <= self.threshold)
		return key

	##
	# add a child to this node
	#
	# @param key_condition
	#				- dictionary key
	# @param new_node
	#				- children node
	# @return None
	#
	def add_child(self, key_condition, new_node):
		self.branches[key_condition] = new_node


class LeafNode:
	##
	# This class represents a leaf node
	#
	# @param label
	#				- label of leaf node
	# @return None
	#
	def __init__(self, label):
		self.is_leaf = True
		self.label = label







