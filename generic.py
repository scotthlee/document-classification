
"""Functions for training and tuning supervised classification models on text data"""

import math
import numpy as np
import sklearn
import pandas as pd
import inspect
from sklearn.svm import SVC, LinearSVC
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

	#wrappers for saving data frame to the RF object for when self.process() isn't used
	def set_data(self, df):
		self.data = df
		return
	
	def set_xy(self, x, y):
		self.X = x
		self.y = y
		return
	
	#another wrapper for the vectorization functions; optional, and will take a while
	def process(self, df, x_name, y_name, ngrams=2, max_features=35000, method='counts', binary=True, verbose=False):
		if verbose:		
			print "Vectorizing the corpus..."

		#choosing the particular flavor of vectorizer
		if method == 'counts':
			vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1, ngrams), decode_error='replace', binary=binary)
		elif method == 'tfidf':
			vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, ngrams), decode_error='replace')
		
		#fitting the vectorizer and converting the counts to an array
		full_fit = vectorizer.fit_transform(df[x_name])
		full_counts = full_fit.toarray()
		
		#passing the attributes up to the class instance
		self.data = df
		self.X = full_counts
		self.y = np.array(df[y_name])
		return	
		
	#splits the data into training and test sets; either called from self.process()
	#or on its own when your text is already vectorized and divided into x and y
	def split(self, split_method='train-test', split_var=None, test_val=None):
		if split_method == 'train-test':
			self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)
		elif split_method == 'var':
			self.X_train, self.X_test, self.y_train, self.y_test = split_by_var(self.X, self.y, self.data, 
												split_var, test_val)
		return 

"""General ML functions"""      
#returns the predicted outputs based on inputs, training weights, and training bias
#exp=True will exponentiate the predicted values, transforming to [0, 1]
def linear_prediction(x, w, b, neg=0, exp=False):
    guesses = np.matmul(x, w.transpose()) + b
    if exp:
        prediction = np.true_divide(1, 1 + np.exp(-1 * (guesses)))
    else:
        prediction = np.array(np.sign(guesses), dtype=int)
        if neg == 0:
            prediction[prediction == -1] = 0    
    return prediction

#returns the accuracy of a classifier based on inputs, outputs, training weights, and training bias
def accuracy(x, y, w, b):
    guess = linear_prediction(x, w, b)
    return np.true_divide(np.sum(guess.reshape(y.shape) == y), x.shape[0])

