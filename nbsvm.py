"""Scott's script for training SVMs a la Wang and Manning (2012)"""
import argparse
import pandas as pd
import numpy as np
import sklearn
from generic import *
from sklearn.svm import SVC, LinearSVC

#calculates the log-count ratio r
def log_count_ratio(pos_text, neg_text, alpha=1):
    p = np.add(alpha, np.sum(pos_text, axis=0))
    q = np.add(alpha, np.sum(neg_text, axis=0))
    p_norm, q_norm = np.sum(p), np.sum(q)
    p_ratio = np.true_divide(p, p_norm)
    q_ratio = np.true_divide(q, q_norm)
    r = np.log(np.true_divide(p_ratio, q_ratio))
    return r

def nb_svm(x, y, w, b, C=1):
    wt = w.transpose()
    y = y.reshape(y.shape[0], 1)
    l2_loss = np.square(np.maximum(0, 1 - y * (np.matmul(x, wt) + b)))
    return np.matmul(w, wt) + C * np.sum(l2_loss)

#returns interpolated weights for constructing the nb-svm
def interpolate(w, beta):
	return ((1 - beta) * (np.sum(w) / w.shape[1])) + (beta * w)

#finds the interpolation paramater beta that yields the highest accuracy
def tune_beta(x, y, w, b, betas):
    n = len(betas)
    results = np.zeros([n, 2])
    results[:,0] = betas
    for i in range(0, n):
        int_weights = interpolate(w, betas[i])
        results[i, 1] = accuracy(x, y, int_weights, b)
    return results

#tries to find the best C parameter for the SVM; probably not very useful
def tune_C(x_tr, y_tr, x_te, y_te, c_params, beta=0.25, interpolate=False):
    out = pd.DataFrame(np.zeros([len(c_params), 11]), columns=diag_names)
    i = 0
    for c_param in c_params:
        clf = LinearSVC(C=c_param).fit(x_tr, y_tr)
        w, b = clf.coef_, clf.intercept_
        if interpolate:
            w = interpolate(clf.coef_, beta)
            b = nb_bias
        out.iloc[i,:] = np.array(diagnostics(x_te, y_te, w, b))
        i += 1
    out.iloc[:,0] = c_params
    return out

#class for the MNB classifier
class TextMNB:
	def __init__(self):
		self.__name__ = 'mnb'
		self.X, self.y = [], []
		self.X_train, self.X_test = [], []
		self.y_train, self.y_test = [], []
		self.X_train_pos, self.X_train_neg = [], []
		self.X_train_nb, self.X_test_nb = [], []
		
		#attributes for the model
		self.r = 0.0
		self.bias = 0.0
		self.nb_bias = 0.0
			
	def fit(self, X, y, verbose=True):
		#setting data attributes for the model instance
		self.X_train, self.y_train = X, y
		
		#splitting by target class so we can calculate the log-count ratio
		self.X_train_pos = self.X_train[np.where(self.y_train == 1)]
		self.X_train_neg = self.X_train[np.where(self.y_train == 0)]
		
		self.r = log_count_ratio(self.X_train_pos, self.X_train_neg)
		self.X_train_nb = np.multiply(self.r, self.X_train)
		
		#setting the npos and nneg variables
		n_pos = self.X_train_pos.shape[0]
		n_neg = self.X_train_neg.shape[0]
		
		#getting the bais for the MNB model
		self.nb_bias = np.log(np.true_divide(n_pos, n_neg))
		
		#training the SVM with NB features but no interpolation			
		
	#trains, tests, and assesses the performance of the model
	def score(self, X, y, verbose=False):
		#setting data attributes for the model instance
		self.X_test, self.y_test = X, y
				
		#scoring the model
		acc = accuracy(self.X_test, self.y_test, self.r, self.nb_bias)
		
		#finding the best interpolation parameter given the data
		return acc
		
	def predict(self, verbose=True):		
		return linear_prediction(self.X_test, self.r, self.nb_bias).reshape(self.y_test.shape)

#main class for the NB-SVM
class TextNBSVM:
	def __init__(self):
		#setting attributes for the data
		self.__name__ = 'nbsvm'
		self.X, self.y = [], []
		self.X_train, self.X_test = [], []
		self.y_train, self.y_test = [], []
		self.X_train_pos, self.X_train_neg = [], []
		self.X_train_nb, self.X_test_nb = [], []
		
		#setting attributes for the NBSVM
		self.coef_ = []
		self.int_coef_ = []
		self.r = 0.0
		self.bias = 0.0
		self.nb_bias = 0.0
		self.beta = 0.25
		
	#loads the data object and saves the train/test sets as instance attributes
	def fit(self, X, y, verbose=True):
		#setting data attributes for the model instance
		X = tfidf_to_counts(X)
		self.X_train, self.y_train = X, y
		
		#splitting by target class so we can calculate the log-count ratio
		self.X_train_pos = self.X_train[np.where(self.y_train == 1)]
		self.X_train_neg = self.X_train[np.where(self.y_train == 0)]
		
		self.r = log_count_ratio(self.X_train_pos, self.X_train_neg)
		self.X_train_nb = np.multiply(self.r, self.X_train)
		
		#setting the npos and nneg variables
		n_pos = self.X_train_pos.shape[0]
		n_neg = self.X_train_neg.shape[0]
		
		#getting the bais for the MNB model
		self.nb_bias = np.log(np.true_divide(n_pos, n_neg))
		
		#training the SVM with NB features but no interpolation
		if verbose:		
			print "Training the NB-SVM..."
		
		nbsvm = LinearSVC()
		nbsvm.fit(self.X_train_nb, self.y_train)
		self.coef_ = nbsvm.coef_
		self.int_coef_ = interpolate(self.coef_, self.beta)
		self.bias = nbsvm.intercept_

	#trains, tests, and assesses the performance of the model
	def score(self, X, y):
		#setting data attributes for the model instance
		X = tfidf_to_counts(X)
		self.X_test, self.y_test = X, y
		self.X_test_nb = np.multiply(self.r, self.X_test)

		#finding the best interpolation parameter given the data
		int_accs = tune_beta(self.X_test_nb, self.y_test, self.coef_, self.bias, np.arange(0, 1.025, .025))
		inter_acc = int_accs[np.argsort(int_accs[:,1])[-1], 1]
		best_beta = int_accs[np.argsort(int_accs[:,1])[-1], 0]
		self.int_coef_ = interpolate(self.coef_, best_beta)
		
		self.beta = best_beta
		return inter_acc
	
	#returns binary class predictions	
	def predict(self, X, verbose=True):
		X = tfidf_to_counts(X)
		X = np.multiply(self.r, X)		
		return linear_prediction(X, interpolate(self.coef_, self.beta), self.bias).reshape(X.shape[0])
		
	#returns predicted probabilities using Platt scaling
	def predict_proba(self, X, y):
		X = tfidf_to_counts(X)
		return platt_scale(X, y, self)
		
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	#positional arguments
	parser.add_argument('data', help='path for the input data')
	parser.add_argument('x_name', help='name of the column holding the text')
	parser.add_argument('y_name', help='name of the column holding the target values')

	#optional arguments for tuning
	parser.add_argument('-lm', '--limit_features', type=bool, default=True, help='limit the number of features for the SVM? (yes or no)')
	parser.add_argument('-ft', '--features', type=int, default=35000, help='number of features for the SVM, if limited')
	parser.add_argument('-ng', '--ngrams', type=int, default=2, help='max ngram size')
	parser.add_argument('-sm', '--split_method', default='train-test', help='split the data by var(iable), train-test, or cross-val')
	parser.add_argument('-sv', '--split_variable', help='which variable to use for splitting')
	parser.add_argument('-tv', '--test_value', help='which value of --split_variable to use for testing')
	parser.add_argument('-vb', '--verbose', default=True, help='should functions print updates as they go?')
	args = parser.parse_args()

	#loading and processing the data
	df = pd.read_csv(args.data)
	d = TextData()
	if args.limit_features:
		d.process(df, args.x_name, args.y_name, max_features=args.features, verbose=args.verbose)
	else:
		d.process(df, args.x_name, args.y_name, max_features=None, verbose=args.verbose)
	
	#getting the training and test sets	
	d.split(args.split_method, args.split_variable, args.test_value)
	
	#running the models
	mod = TextNBSVM()
	mod.fit(d.X_train, d.y_train, verbose=args.verbose)
	svm_acc = mod.score(d.X_test, d.y_test, verbose=args.verbose)
	print "\nResults:"
	print "NBSVM accuracy is %0.4f" %svm_acc
	
	mnb = TextMNB()
	mnb.fit(d.X_train, d.y_train, verbose=args.verbose)
	mnb_acc = mnb.score(d.X_test, d.y_test, verbose=args.verbose)
	print "MNB accuracy is %0.4f" %mnb_acc	

