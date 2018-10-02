"""
CS578 - HW2, Fall 2015, Purdue University
Feature Extraction
@package mylib
@author I-Ta Lee
@date 09/28/2015
"""

# python lib
import logging
import cPickle
import collections
import timeit

# third-party lib
import numpy as np
from scipy import sparse

# my lib
from . import Utils

STOP_WORD_LIST = ['a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also', 'am', 'among',
				  'an', 'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'but', 
				  'by', 'can', 'could', 'dear', 'did', 'do', 'does', 'either', 'else', 
				  'ever', 'every', 'for', 'from', 'get', 'got', 'had', 'has', 'have', 'he', 
				  'her', 'hers', 'him', 'his', 'how', 'however', 'i', 'if', 'in', 'into', 
				  'is', 'it', 'its', 'just', 'least', 'let', 'like', 'likely', 'may', 'me', 
				  'might', 'most', 'must', 'my', 'of', 'off', 
				  'often', 'on', 'only', 'or', 'other', 'our', 'own', 'rather', 'said', 'say', 
				  'says', 'she', 'should', 'since', 'so', 'some', 'than', 'that', 'the', 'their', 
				  'them', 'then', 'there', 'these', 'they', 'this', 'tis', 'to', 'too', 'twas', 
				  'us', 'wants', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 
				  'who', 'whom', 'why', 'will', 'with', 'would', 'yet', 'you', 'your']


class UniGram:
	"""
	a class performs unigram bag-of-word feature extraction
	@package mylab.Features
	"""
	def __init__(self, **kwargs):
		"""
		UniGram Constructor
		@return None
		"""
		self.model = None

	def fit(self, raw_file, term_freq_threshold):
		"""
		interface for constructing bag-of-word model, stored in self.model, from raw_file
		@param raw_file a csv file containing raw data and labels
		@return None
		"""
		docs, y = Utils.load_raw(raw_file)
		self._fit(docs, term_freq_threshold)

	def _fit(self, docs, term_freq_threshold):
		"""
		construct bag-of-word model, which is stored in self.model
		@param docs a list of document data
		@return None
		"""
		words = collections.OrderedDict()
		for doc in docs:
			doc_splited = doc.split(' ')
			for word in doc_splited:
				# case insensitive
				word_lower = word.lower()
				
				# ignore empty string
				if word_lower == '':
					continue

				# ignore stop words
				if word_lower in STOP_WORD_LIST:
					continue
					
				if not words.has_key(word_lower):
					words[word_lower] = 1
				else:
					words[word_lower] += 1

		self.model = [key for key, val in words.iteritems() if val >= term_freq_threshold]

	def transform(self, raw_file):
		"""
		interface for generating bag-of-word feature vectors from raw_file
		@param raw_file a csv file containing raw data and labels
		@return (X, y) feature vectors and labels in numpy array
		"""
		docs, y = Utils.load_raw(raw_file)
		return self._transform(docs, y)

	def _transform(self, docs, y_orig):
		"""
		generate bag-of-word feature vectors 
		@param docs a list of document data
		@param y_orig labels represented in string '+' or '-'
		@return (X, y) feature vectors and labels in numpy array
		"""
		assert self.model is not None

		ndoc = len(docs)
		nfeature = len(self.model)
		X = np.zeros((ndoc, nfeature+1)) 	# add bias
		y = np.zeros(ndoc)

		for i in range(ndoc):
			bag_of_unigram = collections.OrderedDict.fromkeys(self.model, 0)
			doc_splited = docs[i].split(' ')
			for word in doc_splited:

				word_lower = word.lower()
				if bag_of_unigram.has_key(word_lower):
					bag_of_unigram[word_lower] += 1
				else:
					# filtered words
					pass
			
			X[i] = np.array([1] + bag_of_unigram.values())

			assert y_orig[i] == '+' or y_orig[i] == '-'
			y[i] = 1 if y_orig[i] == '+' else -1

		return X, y

	def fit_transform(self, raw_file, term_freq_threshold):
		"""
		interface for constructing bag-of-word model and generating feature vectors from raw_file
		@param raw_file a csv file containing raw data and labels
		@return (X, y) feature vectors and labels in numpy array
		"""
		docs, y = Utils.load_raw(raw_file)
		self._fit(docs, term_freq_threshold)
		return self._transform(docs, y)


class BiGram:
	"""
	a class performs bigram bag-of-word feature extraction
	@package mylab.Features
	"""
	def __init__(self, **kwargs):
		"""
		BiGram Constructor
		@return None
		"""
		self.model = None

	def fit(self, raw_file, term_freq_threshold):
		"""
		interface for constructing bag-of-bigram, stored in self.model, model from raw_file
		@param raw_file a csv file containing raw data and labels
		@return None
		"""
		docs, y = Utils.load_raw(raw_file)
		self._fit(docs, term_freq_threshold)

	def _fit(self, docs, term_freq_threshold):
		"""
		construct bag-of-bigram model, which is stored in self.model
		@param docs a list of document data
		@return None
		"""
		phases = collections.OrderedDict()
		for doc in docs:
			doc_splited = doc.split(' ')

			#ignore stop words and case insensitive
			doc_splited = [d.lower() for d in doc_splited if d.lower() not in STOP_WORD_LIST]

			for i in range(len(doc_splited)-1):
				# ignore empty string
				if doc_splited[i] == '' or doc_splited[i+1] == '':
					continue

				phase = ' '.join((doc_splited[i], doc_splited[i+1]))
				
				if not phases.has_key(phase):
					phases[phase] = 1
				else:
					phases[phase] += 1

		self.model = [key for key, val in phases.iteritems() if val >= term_freq_threshold]
		print len(self.model)

	def transform(self, raw_file):
		"""
		interface for generating bag-of-bigram feature vectors from raw_file
		@param raw_file a csv file containing raw data and labels
		@return (X, y) feature vectors and labels in numpy array
		"""
		docs, y = Utils.load_raw(raw_file)
		return self._transform(docs, y)

	def _transform(self, docs, y_orig):
		"""
		generate bag-of-bigram feature vectors 
		@param docs a list of document data
		@param y_orig labels represented in string '+' or '-'
		@return (X, y) feature vectors and labels in numpy array
		"""
		assert self.model is not None

		ndoc = len(docs)
		nfeature = len(self.model)
		X = np.zeros((ndoc, nfeature+1))	# add bias
		y = np.zeros(ndoc)

		for i in range(ndoc):
			bag_of_bigram = collections.OrderedDict.fromkeys(self.model, 0)
			doc_splited = docs[i].split(' ')

			# ignore stop words and case insensitive
			doc_splited = [d.lower() for d in doc_splited if d.lower() not in STOP_WORD_LIST]

			for j in range(len(doc_splited)-1):
				if doc_splited[j] == '' or doc_splited[j+1] == '':
					continue

				phase = ' '.join((doc_splited[j], doc_splited[j+1]))
 
				if bag_of_bigram.has_key(phase):
					bag_of_bigram[phase] += 1
				else:
					# filtered words
					pass

			X[i] = np.array([1] + bag_of_bigram.values())

			assert y_orig[i] == '+' or y_orig[i] == '-'
			y[i] = 1 if y_orig[i] == '+' else -1

		return X, y

	def fit_transform(self, raw_file, term_freq_threshold):
		"""
		interface for constructing bag-of-bigram model and generating feature vectors from raw_file
		@param raw_file a csv file containing raw data and labels
		@return (X, y) feature vectors and labels in numpy array
		"""
		docs, y = Utils.load_raw(raw_file)
		self._fit(docs, term_freq_threshold)
		return self._transform(docs, y)


class FeatureSet:
	"""
	a container class collects feature sets
	@package mylab.Features
	"""

	"""
	constants, types of feature sets supported
	"""
	FEATURE_UNIGRAM = 1
	FEATURE_BIGRAM = 2

	def __init__(self, **kwargs):
		"""
		FeatureSet Constructor
		@param loglevel (optional) logging level
		@return None
		"""
		# parameters
		loglevel = logging.ERROR if 'loglevel' not in kwargs else kwargs['loglevel']
		logging.basicConfig(format='[%(levelname)s][%(name)s] %(message)s', level=loglevel)
		self.logger = logging.getLogger(__name__+'.'+self.__class__.__name__) 

	def extract_feature(self, raw_path, feature_select=[FEATURE_UNIGRAM], **kwargs):
		"""
		do feature extraction; extracted features and labels would be stored in self.{X/y}_{train/validate/test}; 
		multiple feature sets will be concatenated horizontally in self.X_{train/validate/test}; 
		and the self.dim will be a list of integers indicating the dimension of each feature set
		@param raw_path a dictionary containing keys: 'train', 'validate', 'test', and corresponding values which are "key"'s file paths
		@param feature_select a list of feature type selections (Default: [FEATURE_UNIGRAM])
		@return None
		"""
		unigram_tf_threshold = 3 if 'unigram_tf_threshold' not in kwargs else kwargs['unigram_tf_threshold']
		bigram_tf_threshold = 2 if 'bigram_tf_threshold' not in kwargs else kwargs['bigram_tf_threshold']
		self.X_train = None
		self.X_validate = None
		self.X_test = None
		self.y_train = None
		self.y_validate = None
		self.y_test = None
		self.dim = []
		self.feature_select = feature_select

		self.logger.info("start feature extraction")
		start_time = timeit.default_timer()
		for fsel in feature_select:
			fset = {
				FeatureSet.FEATURE_UNIGRAM: UniGram(),
				FeatureSet.FEATURE_BIGRAM: BiGram(),
			} [fsel]
			tf_threshold = {
				FeatureSet.FEATURE_UNIGRAM: unigram_tf_threshold,
				FeatureSet.FEATURE_BIGRAM: bigram_tf_threshold,
			} [fsel]

			if self.X_train is None:
				self.X_train, self.y_train = fset.fit_transform(raw_path['train'], tf_threshold)
				self.X_validate, self.y_validate = fset.transform(raw_path['validate'])
				self.X_test, self.y_test = fset.transform(raw_path['test'])
				self.dim += [self.X_train.shape[1]]
			else:
				X_tmp, y_tmp = fset.fit_transform(raw_path['train'], tf_threshold)	
				self.X_train = np.concatenate((self.X_train, X_tmp), axis=1)
				assert (y_tmp == self.y_train).all()

				X_tmp, y_tmp = fset.transform(raw_path['validate'])
				self.X_validate = np.concatenate((self.X_validate, X_tmp), axis=1)
				assert (y_tmp == self.y_validate).all()

				X_tmp, y_tmp = fset.transform(raw_path['test'])
				self.X_test = np.concatenate((self.X_test, X_tmp), axis=1)
				assert (y_tmp == self.y_test).all()

				self.dim += [self.X_train.shape[1] - sum(self.dim)]

		self.logger.info("feature extraction elapses: %f s" % (timeit.default_timer() - start_time))
		self.logger.info("X_train.shape = %s" % (str(self.X_train.shape)))
		self.logger.info("X_validate.shape = %s" % (str(self.X_validate.shape)))
		self.logger.info("X_test.shape = %s" % (str(self.X_test.shape)))
		assert sum(self.dim) == self.X_train.shape[1]
		assert sum(self.dim) == self.X_validate.shape[1]
		assert sum(self.dim) == self.X_test.shape[1]

	def load(self, file_name):
		"""
		load features from a pickle file into this object
		@param file_name file name of the pickle file
		@return None
		"""
		self.logger.debug("load features from file: %s" % (file_name))
		start_time = timeit.default_timer()
		tmp_obj = cPickle.load(open(file_name, "rb"))
		self.feature_select = tmp_obj['feature_select']
		self.dim = tmp_obj['dim']

		# load from sparse matrix
		self.X_train = tmp_obj['X_train'].toarray()
		self.y_train = tmp_obj['y_train'].toarray()
		self.y_train = np.reshape(self.y_train, self.y_train.shape[1])

		self.X_validate = tmp_obj['X_validate'].toarray()
		self.y_validate = tmp_obj['y_validate'].toarray()
		self.y_validate = np.reshape(self.y_validate, self.y_validate.shape[1])

		self.X_test = tmp_obj['X_test'].toarray()
		self.y_test = tmp_obj['y_test'].toarray()
		self.y_test = np.reshape(self.y_test, self.y_test.shape[1])

		self.logger.info("load file elapses: %f s" % (timeit.default_timer() - start_time))
		self.logger.info("X_train.shape = %s" % (str(self.X_train.shape)))
		self.logger.info("X_validate.shape = %s" % (str(self.X_validate.shape)))
		self.logger.info("X_test.shape = %s" % (str(self.X_test.shape)))
	
	def dump(self, file_name):
		"""
		dump features to a pickle file from this object
		@param file_name file name of the pickle file
		@return None
		"""
		start_time = timeit.default_timer()
		tmp_obj = {}
		tmp_obj['feature_select'] = self.feature_select
		tmp_obj['dim'] = self.dim

		# save as sparse matrix
		tmp_obj['X_train'] = sparse.csr_matrix(self.X_train)
		tmp_obj['y_train'] = sparse.csr_matrix(self.y_train)
		tmp_obj['X_validate'] = sparse.csr_matrix(self.X_validate)
		tmp_obj['y_validate'] = sparse.csr_matrix(self.y_validate)
		tmp_obj['X_test'] = sparse.csr_matrix(self.X_test)
		tmp_obj['y_test'] = sparse.csr_matrix(self.y_test)
		
		self.logger.info("dumping features to file: %s" % (file_name))
		cPickle.dump(tmp_obj, open(file_name, "wb"))
		self.logger.info("dump file elapses: %f s" % (timeit.default_timer() - start_time))



