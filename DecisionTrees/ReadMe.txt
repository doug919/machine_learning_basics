usage: hw1.py [-h] [-p] [-t] [-m DEPTH] [-f DEFAULT_LABEL] [-r RANGE] [-e]
              [-v] [-d]
              TRAINING_DATA VALIDATION_DATA TESTINIG_DATA

CS180 HW1 Decision Tree by I-Ta Lee

positional arguments:
  TRAINING_DATA         training data in csv format
  VALIDATION_DATA       validation data in csv format
  TESTINIG_DATA         testing data in csv format

optional arguments:
  -h, --help            show this help message and exit
  -p, --do_prune        reduced error pruning (DEFAULT: False)
  -t, --use_threshold   use threshold to partition tree (DEFAULT: False)
  -m DEPTH, --max_depth DEPTH
                        maximum depth of decision trees (DEFAULT: INFINITE)
  -f DEFAULT_LABEL, --default_label DEFAULT_LABEL
                        default label used when the majority vote has the same
                        number of votes (DEFAULT: 1)
  -r RANGE, --int_range RANGE
                        integer ranger of data. This follows the format "x-y",
                        and it means range(x, y+1) (DEFAULT: 1-10, which is
                        equal to range(1, 11)
  -e, --print_tree      print tree (DEFAULT: False)
  -v, --verbose         show info messages
  -d, --debug           show debug messages


Examples:
	### ID3 algorithm partitioning with discrete attribute value
	> python hw1.py data/training.csv data/validating.csv data/testing.csv
	training accuracy = 0.983333
	validation accuracy = 0.800000
	testing accuracy = 0.850000 

	### ID3 algorithm partitioning with discrete attribute value and using post-pruning
	> python hw1.py -p data/training.csv data/validating.csv data/testing.csv
	training accuracy = 0.983333
	validation accuracy = 0.800000
	testing accuracy = 0.850000
	post-pruning training accuracy = 0.900000
	post-pruning validation accuracy = 0.900000
	post-pruning testing accuracy = 0.850000

	### ID3 algorithm partitioning with threshold value (continuous value)
	> python hw1.py -t data/training.csv data/validating.csv data/testing.csv
	training accuracy = 0.983333
	validation accuracy = 0.600000
	testing accuracy = 0.750000
	
	### ID3 algorithm partitioning with threshold value (continuous value) and using post-pruning
	> python hw1.py -t -p data/training.csv data/validating.csv data/testing.csv
	training accuracy = 0.983333
	validation accuracy = 0.600000
	testing accuracy = 0.750000
	post-pruning training accuracy = 0.900000
	post-pruning validation accuracy = 0.900000
	post-pruning testing accuracy = 0.850000

	### ID3 algorithm partitioning with discrete attribute value and setting maximum depth to 2
	> python hw1.py -m 2 data/training.csv data/validating.csv data/testing.csv
	training accuracy = 0.966667
	validation accuracy = 0.800000
	testing accuracy = 0.850000

	### ID3 algorithm partitioning with discrete attribute value and setting maximum depth to 3
	> python hw1.py -m 3 data/training.csv data/validating.csv data/testing.csv
	training accuracy = 0.983333
	validation accuracy = 0.800000
	testing accuracy = 0.850000

	### ID3 algorithm partitioning with threshold value (continuous value) and setting maximum depth to 3
	> python hw1.py -m 3 -t data/training.csv data/validating.csv data/testing.csv
	training accuracy = 0.900000
	validation accuracy = 0.900000
	testing accuracy = 0.850000

	### ID3 algorithm partitioning with threshold value (continuous value) and setting maximum depth to 4
	> python hw1.py -m 4 -t data/training.csv data/validating.csv data/testing.csv
	training accuracy = 0.933333
	validation accuracy = 0.650000
	testing accuracy = 0.850000

	### ID3 algorithm partitioning with threshold value (continuous value) and setting maximum depth to 4 and pruning
	> python hw1.py -m 4 -t -p data/training.csv data/validating.csv data/testing.csv
	training accuracy = 0.933333
	validation accuracy = 0.650000
	testing accuracy = 0.850000
	post-pruning training accuracy = 0.900000
	post-pruning validation accuracy = 0.900000
	post-pruning testing accuracy = 0.850000




