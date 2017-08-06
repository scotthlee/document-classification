'''Scott's script for training NB-SVMs a la Wang and Manning (2012)'''
import argparse
import pandas as pd
import numpy as np

from tools import *
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold

# Calculates the log-count ratio r
def log_count_ratio(pos_text, neg_text, alpha=1):
    p = np.add(alpha, np.sum(pos_text, axis=0))
    q = np.add(alpha, np.sum(neg_text, axis=0))
    p_norm, q_norm = np.sum(p), np.sum(q)
    p_ratio = np.true_divide(p, p_norm)
    q_ratio = np.true_divide(q, q_norm)
    r = np.log(np.true_divide(p_ratio, q_ratio))
    return r

# Returns interpolated weights for constructing the NB-SVM
def interpolate(w, beta):
	return ((1 - beta) * (np.true_divide(np.linalg.norm(w, ord=1), w.shape[1]))) + (beta * w)

# Finds the interpolation paramater beta that yields the highest accuracy
def tune_beta(x, y, w, b, betas):
    n = len(betas)
    results = np.zeros([n, 2])
    results[:,0] = betas
    for i in range(0, n):
        int_weights = interpolate(w, betas[i])
        results[i, 1] = accuracy(x, y, int_weights, b)
    return results

# Main class for the NB-SVM
class NBSVM:
	def __init__(self, C=0.1, beta=0.25):
		self.__name__ = 'nbsvm'
		self.coef_ = []
		self.int_coef_ = []
		self.r = 0.0
		self.bias = 0.0
		self.nb_bias = 0.0
		self.beta = beta
		self.C = C
		
	# Fits the model to the data and does the interpolation
	def fit(self, x, y):
		# Convert non-binary features to binary
		bin_x = tfidf_to_counts(x)
		
		# Calculating the log-count ratio
		X_pos = bin_x[np.where(y == 1)]
		X_neg = bin_x[np.where(y == 0)]
		self.r = log_count_ratio(X_pos, X_neg)
		X = np.multiply(self.r, bin_x)
		
		# Training linear SVM with NB features but no interpolation
		svm = LinearSVC(C=self.C)
		svm.fit(X, y)
		self.coef_ = svm.coef_
		self.int_coef_ = interpolate(self.coef_, self.beta)
		self.bias = svm.intercept_
	
	# Scores the interpolated model
	def score(self, x, y):
		bin_x = tfidf_to_counts(x)
		X = np.multiply(self.r, bin_x)
		return accuracy(X, y, self.int_coef_, self.bias)
	
	# Returns binary class predictions	
	def predict(self, x):
		bin_x = tfidf_to_counts(x)
		X = np.multiply(self.r, bin_x)
		return np.squeeze(linear_prediction(X, self.int_coef_, self.bias))
		
	# Returns predicted probabilities using Platt scaling
	def predict_proba(self, x, y):
		X = tfidf_to_counts(x)
		return platt_scale(X, y, self)

# Class for the MNB classifier
class TextMNB:
	def __init__(self):
		self.__name__ = 'mnb'
		
		#attributes for the model
		self.r = 0.0
		self.bias = 0.0
		self.nb_bias = 0.0
			
	def fit(self, x, y, verbose=True):
		#setting data attributes for the model instance
		X = tfidf_to_counts(x)
		
		#splitting by target class so we can calculate the log-count ratio
		X_pos = X[np.where(y == 1)]
		X_neg = X[np.where(y == 0)]
		self.r = log_count_ratio(X_pos, X_neg)
		
		#setting the npos and nneg variables
		n_pos = X_pos.shape[0]
		n_neg = X_neg.shape[0]
		
		#getting the bais for the MNB model
		self.nb_bias = np.log(np.true_divide(n_pos, n_neg))
		
	#trains, tests, and assesses the performance of the model
	def score(self, x, y):
		X = tfidf_to_counts(x)
		acc = accuracy(X, y, self.r, self.nb_bias)
		return acc
		
	def predict(self, x):
		X = tfidf_to_counts(x)		
		return np.squeeze(linear_prediction(X, self.r, self.nb_bias))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	# Positional arguments
	parser.add_argument('data', help='path for the input data')
	parser.add_argument('x_name', help='name of the column holding the text')
	parser.add_argument('y_name', help='name of the column holding the target values')

	# Optional arguments for tuning
	parser.add_argument('-lm', '--limit_features', type=bool, default=True, help='limit the number of features for the SVM? (yes or no)')
	parser.add_argument('-ft', '--features', type=int, default=35000, help='number of features for the SVM, if limited')
	parser.add_argument('-ng', '--ngrams', type=int, default=2, help='max ngram size')
	parser.add_argument('-sm', '--split_method', default='train-test', help='split the data by var(iable), train-test, or cross-val')
	parser.add_argument('-sv', '--split_variable', help='which variable to use for splitting')
	parser.add_argument('-tv', '--test_value', help='which value of --split_variable to use for testing')
	parser.add_argument('-vb', '--verbose', default=True, help='should functions print updates as they go?')
	args = parser.parse_args()

	# Loading and processing the data
	df = pd.read_csv(args.data)
	d = TextData()
	if args.limit_features:
		d.process(df, args.x_name, args.y_name, max_features=args.features, verbose=args.verbose)
	else:
		d.process(df, args.x_name, args.y_name, max_features=None, verbose=args.verbose)
	
	# Getting the training and test sets	
	d.split(args.split_method, args.split_variable, args.test_value)
	
	# Running the models
	mod = NBSVM(C=args.c_parameter, beta=args.beta)
	mod.fit(d.X_train, d.y_train, verbose=args.verbose)
	svm_acc = mod.score(d.X_test, d.y_test)
	print "\nResults:"
	print "NBSVM accuracy is %0.4f" %svm_acc
	
	mnb = TextMNB()
	mnb.fit(d.X_train, d.y_train, verbose=args.verbose)
	mnb_acc = mnb.score(d.X_test, d.y_test)
	print "MNB accuracy is %0.4f" %mnb_acc	

