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

