usage: hw2.py [-h] [-a ALGO] [-i MAXITERATION] [-f FEATURE] [-l LEARNINGRATE]
              [-u THRESHOLD] [-b THRESHOLD] [-g] [-w FILE_NAME] [-r FILE_NAME]
              [-v] [-d]
              TRAINING_DATA VALIDATING_DATA TESTINIG_DATA

CS180 HW2 Winnow and Perceptron by I-Ta Lee

positional arguments:
  TRAINING_DATA         training data in csv format
  VALIDATING_DATA       validation data in csv format
  TESTINIG_DATA         testing data in csv format

optional arguments:
  -h, --help            show this help message and exit
  -a ALGO, --algorithm ALGO
                        choose algorithm: 1. perceptron, 2. winnow, 3. dual
                        perceptron (DEFAULT: 1)
  -i MAXITERATION, --max_iteration MAXITERATION
                        the maximum number of iterations. should be a positive
                        integer (DEFAULT: 10)
  -f FEATURE, --feature_select FEATURE
                        choose feature set: 1. unigram, 2. bigram 3. both
                        (DEFAULT: 1)
  -l LEARNINGRATE, --learning_rate LEARNINGRATE
                        learning rate (DEFAULT: 1.0)
  -u THRESHOLD, --unigram_tf_threshold THRESHOLD
                        unigram term frequency threshold for feature selection
                        (DEFAULT: >=3)
  -b THRESHOLD, --bigram_tf_threshold THRESHOLD
                        bigram term frequency threshold for feature selection
                        (DEFAULT: >=2)
  -g, --avg_perceptron  use averaged perceptron instead of perceptron (-a
                        should be 1)
  -w FILE_NAME, --dump_feature FILE_NAME
                        dump the selected feature set (DEFAULT: None)
  -r FILE_NAME, --load_feature FILE_NAME
                        file path for the feature set to read; will ignore the
                        feature selected by -f (DEFAULT: None)
  -v, --verbose         show info messages
  -d, --debug           show debug messages

Example Usage

  ##### Basic Scenarios #####
  #
  ### run winnow algorithm with unigram feature for 100 iterations(epochs).
  python hw2.py -a 2 -i 100 -f 1 data/train.csv data/validation.csv data/test.csv

  ### run perceptron algorithm with bigram feature for 1000 iterations
  python hw2.py -a 1 -i 1000 -f 2 data/train.csv data/validation.csv data/test.csv

  ### run averaged perceptron algorithm with bigram feature for 1000 iterations (just add -g)
  python hw2.py -a 1 -i 1000 -f 2 -g data/train.csv data/validation.csv data/test.csv

  ### run dual perceptron algorithm with unigram+bigram feature for 50 iterations
  python hw2.py -a 3 -i 50 -f 3 data/train.csv data/validation.csv data/test.csv



  ##### To collect result and plot figures ######
  ### our main program will output all training/validating/testing results for each iteration in stdout
  ### If I want to get the best validating hyperparameters, which is the epoch, 
  ### I have toredirect the result into a file then use another collect_and_plot_result.py to parse and 
  ### select the best hyperparameter and its testing results

  ### The command should be like this:
  python hw2.py -a 2 -i 100 -f 1 data/train.csv data/validation.csv data/test.csv > result_winnow_unigram_100
  python parse_and_plot_result.py result_winnow_unigram_100

  ### output would be like this
  best validating accuracy = 0.656660
  best epoch = 6
  final training accuracy, precision, recall, f1 = (0.776024, 0.771463, 0.786604, 0.77896)
  final validating accuracy, precision, recall, f1 = (0.65666, 0.658192, 0.654494, 0.656338)
  final test accuracy, precision, recall, f1 = (0.657598, 0.656039, 0.644824, 0.650383)
  ***Since the server doesn't support the matplotlib, I comment the plotting things out.



  ##### Extra Functions #####
  #
	### you can output a pickle file for your unigram/bigram/both features to avoid repeatedly conducting feature extraction for different experiments
	### run perceptron with unigram and at the same time output the unigram feature as pickle file
	python hw2.py -a 1 -i 100 -f 1 -w unigram.pkl data/train.csv data/validation.csv data/test.csv

  ### run perceptron with unigram loaded from a pickle file (this will ignore -f option and will not read the raw data files)
  python hw2.py a 1 -i 100 -r unigram.pkl data/train.csv data/validation.csv data/test.csv

	




