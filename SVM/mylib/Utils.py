##
# Utils.py
# 
# CS 578 - HW3
# Fall 2015 @ Purdue University
#
# @author I-Ta Lee
#
# @date 11/05/2015
#

import csv
import json
import logging

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

def dump_dict_to_csv(file_name, data):
    import csv
    w = csv.writer(open(file_name, 'w'))
    for key, val in data.items():
        w.writerow([key, val])

def dump_list_to_csv(file_name, data):
    import csv
    w = csv.writer(open(file_name, 'w'))
    for row in data:
        w.writerow(row)


class Setting:
    def __init__(self, json_filename):
        with open(json_filename) as json_file:
            self.setting = json.load(json_file)

    def get_loglevel(self):
        ret = None
        if self.setting["loglevel"] == "ERROR":
            ret = logging.ERROR
        elif self.setting["loglevel"] == "DEBUG":
            ret = logging.DEBUG
        elif self.setting["loglevel"] == "INFO":
            ret = logging.INFO

        assert ret != None
        return ret

    def get_training_data_path(self):
        return self.setting["training_data_path"]

    def get_validating_data_path(self):
        return self.setting["validating_data_path"]

    def get_testing_data_path(self):
        return self.setting["testing_data_path"]

    def get_unigram_term_freq_threshold(self):
        return self.setting["unigram_term_freq_threshold"]

    def get_bigram_term_freq_threshold(self):
        return self.setting["bigram_term_freq_threshold"]

    def get_load_feature_path(self):
        return self.setting["load_feature_path"]

    def get_dump_feature_path(self):
        return self.setting["dump_feature_path"]

