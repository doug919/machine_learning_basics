##
# Utils.py
# 
# CS 578 - HW2
# Fall 2015 @ Purdue University
#
# @author I-Ta Lee
#
# @date 09/25/2015
#

import csv

##
# utility function for parsing the input argument to a integer list
#
# @param astr
#				- a string like "1-10"
# @return a list of the specified integers
#
def parse_range(astr):
    result = set()
    for part in astr.split(','):
        x = part.split('-')
        result.update(range(int(x[0]), int(x[-1]) + 1))
    return sorted(result)

##
# utility function for parsing the input arguments to a list
#
# @param astr
#				- a string like "1,2,10"
# @return a list of the specified values
#
def parse_list(astr):
    result = set()
    for part in astr.split(','):
        result.add(float(part))
    return sorted(result)


def load_raw(path):
	raw = []
	y = []
	with open(path, 'r') as csvfile:
		r = csv.reader(csvfile)
		for row in r:
			raw += [row[0]]
			y += [row[1]]

	return raw, y
