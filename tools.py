"""Functions for training and tuning supervised classification models"""
import math
import autograd.numpy as np
import sklearn
import pandas as pd
import inspect
import autograd
import scipy

from copy import deepcopy
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from autograd import jacobian
from sklearn.svm import SVC, LinearSVC
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, KFold

"""Stuff for preparing the data"""
#divides the data into training and testing samples using a column variable
def split_by_var(x, y, full_set, split_var, test_val):
	v = full_set[split_var].astype(str)
	train_indices = v[~v.isin([test_val])].index.tolist()
	test_indices = v[v.isin([test_val])].index.tolist()
	X_train = x[train_indices, :]
	X_test = x[test_indices, :]
	y_train = y[train_indices]
	y_test = y[test_indices]
	return [X_train, X_test, y_train, y_test]

#class for holding the split training and test data
class TextData:
	def __init__(self):
	#setting attributes for the data
		self.data = pd.DataFrame()
		self.X, self.y = [], []
		self.X_train, self.X_test = [], []
		self.y_train, self.y_test = [], []

	#wrappers for saving data frame to the RF object for when process() isn't used
	def set_data(self, df):
		self.data = df
		return

	def set_xy(self, x, y):
		self.X = deepcopy(x)
		self.y = deepcopy(y)
		return

	def set_y(self, y):
		self.y = deepcopy(y)
		return

	#another wrapper for the vectorization functions; optional, and will take a while
	def process(self, df, x_name, y_name=None, ngrams=2, max_features=35000, method='counts', binary=True, sparse=False):
		#choosing the particular flavor of vectorizer
		if method == 'counts':
			vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1, ngrams), decode_error='replace', binary=binary)
		elif method == 'tfidf':
			vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, ngrams), decode_error='replace')

		#fitting the vectorizer and converting the counts to an array
		full_fit = vectorizer.fit_transform(df[x_name])
		full_counts = full_fit.toarray()
		self.vocabulary_ = vectorizer.vocabulary_

		#passing the attributes up to the class instance
		self.data = df
		if sparse:
			full_counts = csr_matrix(full_counts)
		self.X = full_counts
		if y_name != None:
			self.y = np.array(df[y_name])
		return

	#splits the data into training and test sets; either called from process()
	#or on its own when your text is already vectorized and divided into x and y
	def split(self, method='train-test', split_var=None, test_val=None, seed=None):
		if method == 'train-test':
			self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=seed)
		elif method == 'var':
			self.X_train, self.X_test, self.y_train, self.y_test = split_by_var(self.X, self.y, data,
												split_var, test_val)
		elif method == 'bootstrap':
			index_range = np.arange(0, len(self.y))
			indices = np.random.choice(index_range, len(self.y), replace=True)
			X, y = self.X[indices, :], self.y[indices]
			self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, random_state=seed)
		return

#wrapper for inspect.getargspec; helps navigate my less-than-ideal naming conventions
def get_args(foo):
	return inspect.getargspec(foo)

"""General ML functions"""
#produces all the accuracy statistics
def diagnostics(guesses, targets):
	out = pd.DataFrame(np.zeros([1, 5]), columns=['se', 'sp', 'ppv', 'f1', 'acc'])
	out['se'] = sens(guesses, targets)
	out['sp'] = spec(guesses, targets)
	out['ppv'] = ppv(guesses, targets)
	out['f1'] = f1(guesses, targets)
	out['acc'] = acc(guesses, targets)
	return out

def model_diagnostics(mod, X, y, type='single'):
	if type == 'ensemble':
		g = mod.predict(X, y)
	else:
		g = mod.predict(X)
	return diagnostics(g, y)

#functions for accuracy statistics
def sens(guesses, targets):
	tp = np.sum(np.logical_and(guesses==1, targets==1))
	fn = np.sum(np.logical_and(guesses==0, targets==1))
	return np.true_divide(tp, tp + fn)

def spec(guesses, targets):
	tn = np.sum(np.logical_and(guesses==0, targets==0))
	fp = np.sum(np.logical_and(guesses==1, targets==0))
	return np.true_divide(tn, tn + fp)

def ppv(guesses, targets):
	tp = np.sum(np.logical_and(guesses==1, targets==1))
	fp = np.sum(np.logical_and(guesses==1, targets==0))
	return np.true_divide(tp, tp + fp)

def f1(guesses, targets):
	recall = sens(guesses, targets)
	precision = ppv(guesses, targets)
	return np.true_divide(2 * (precision * recall), precision + recall)

def acc(guesses, targets):
	return np.true_divide(np.sum(guesses == targets), len(guesses))

#returns the predicted outputs based on inputs, training weights, and training bias
#exp=True will exponentiate the predicted values, transforming to [0, 1]
def linear_prediction(x, w, b, neg=0, binary=True):
	guesses = np.matmul(x, w.transpose()) + b
	if binary:
		prediction = np.array(np.sign(guesses), dtype=int)
        	if neg == 0:
            		prediction[prediction == -1] = 0
	else:
		prediction = guesses
	return prediction

#returns the accuracy of a classifier based on inputs, outputs, training weights, and training bias
def accuracy(x, y, w, b):
    guess = linear_prediction(x, w, b)
    return np.true_divide(np.sum(guess.reshape(y.shape) == y), x.shape[0])

#converts tf-idf matrices to binary count matrices
def tfidf_to_counts(data):
	data = deepcopy(data)
	data[np.where(data > 0)] = 1
	return data

"""Functions for implementing Platt scaling with SVMs"""
#simple function for getting t from y (for Platt scaling)
def y_to_t(y):
	#quick type change, just in case
	y = deepcopy(np.array(y))

	#calculating t, which will replace 1/0 in y
	n_pos = np.sum(y == 1)
	n_neg = np.sum(y == 0)
	t_pos = np.true_divide(n_pos + 1, n_pos + 2)
	t_neg = np.true_divide(1, n_neg + 2)

	#replacing values in y with the appropriate t
	y[np.where(y == 1)] = t_pos
	y[np.where(y == 0)] = t_neg
	return y

#calculates cross-entropy using the sigmoid (for Platt scaling)
def platt_loss(vals, preds, y):
	A = vals[0]
	B = vals[1]
	p = platt_probs(A, B, preds)
	loss = -np.sum(y*np.log(p) + (1 - y)*np.log(1 - p))
	return loss

#calculates the sigmoid transformation of the linear predictions
def platt_probs(A, B, preds):
	p =  np.true_divide(1, (1 + np.exp(A*preds + B)))
	p = p.reshape(p.shape[0], )
	return p

#uses gradient descent to scale the
def platt_scale(x, y, mod, max_iter=1000, step=.001):
	#getting variables for the Platt scaling
	t = y_to_t(y)
	n_pos = np.sum(y == 1)
	n_neg = np.sum(y == 0)
	A = 0.0
	B = np.log(np.true_divide(n_neg + 1, n_pos + 1))

	#getting the predictions
	if type(mod).__name__ != 'LinearSVC':
		#mnb-ifying the input
		X = np.multiply(mod.r, x)
		preds = linear_prediction(X, mod.int_coef_, mod.bias, binary=False)
	else:
		X = deepcopy(x)
		preds = linear_prediction(X, mod.coef_, mod.intercept_, binary=False)

	#minimizing A and B via gradient descent
	vals = np.array([A, B])
	gradient = jacobian(platt_loss)
	for i in range(max_iter):
		vals -= gradient(vals, preds, y)*step

	#returning the
	A = vals[0]
	B = vals[1]
	probs = platt_probs(A, B, preds)
	return probs