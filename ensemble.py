'''A simple script for making ensemble classifiers with random forests and NBSVMs'''
import pandas as pd
import numpy as np
import sklearn
import generic
import nbsvm
import rf

from scipy.stats import gmean
from generic import *
from nbsvm import *
from rf import *

#holder class for the RF-NBSVM ensemble
class RF_SVM:
	def __init__(self):
		self.rf = []
		self.nbsvm = []
		self.data = []
	
	#fitting the separate models; input is a TextData instance
	def fit(self, data):
		#importing the data; X should be a TF-IDF matrix
		self.data = data
		
		#fitting the models
		forest = TextRF()
		svm = TextNBSVM()
		forest.fit(data.X_train, data.y_train)
		svm.fit(tfidf_to_counts(data.X_train), data.y_train)
		
		#setting class attributes
		self.rf = forest
		self.nbsvm = svm
	
	#averages and scores the separate probabilities
	def score(self, X, y, method='geometric', threshold=0.5):
		if method =='geometric':
			probs = self.probs(X, y)
			mean_probs = gmean(probs, axis=1).reshape(X.shape[0], 1)
			all_probs = np.concatenate((probs, mean_probs), axis=1)
			guesses = [int(x >= threshold) for x in mean_probs]
			
			#getting the accuracy of the three classifiers
			rf_acc = self.rf.score(X, y)
			svm_acc = self.nbsvm.score(X, y, verbose=False)
			acc = np.true_divide(np.sum(guesses == y), len(probs))
			
		return {'rf_acc':rf_acc, 'svm_acc':svm_acc, 'ens_acc':acc}
	
	#returns binary class predictions for the separate models
	def predict(self, X):
		rf_preds = self.rf.predict(X).reshape(X.shape[0], 1)
		svm_preds = self.nbsvm.predict(tfidf_to_counts(X)).reshape(X.shape[0], 1)
		return np.concatenate((rf_preds, svm_preds), axis=1)
	
	#returns probabilities for the test data
	def probs(self, X, y):
		rf_probs = self.rf.predict_proba(X)[:,1].reshape(X.shape[0], 1)
		svm_probs = platt_scale(tfidf_to_counts(X), y, self.nbsvm)['probs'].reshape(X.shape[0], 1)
		probs = np.concatenate((rf_probs, svm_probs), axis=1)
		return probs