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

#generalizing the RF-SVM class
class Ensemble:
	def __init__(self):
		self.mods = {}
		self.accs = {}
		self.data = []
		self.probs = []
		self.__name__ = 'Ensemble'
	
	#adds a model to the ensemble
	def add(self, model):
		self.mods[model.__name__] = model
		self.accs[model.__name__] = 0.0
		return
	
	#removes a model from the ensemble
	def remove(self, name):
		del self.mods[name]
		del self.accs[name]
		return
	
	#fitting the models to the training data
	def fit(self, X, y):
		for mod in self.mods:
			self.mods.get(mod).fit(X, y)
		return
	
	#scoring the models on the test data
	def score_sep(self, X, y, verbose=True):
		for mod in self.mods:
			self.accs[mod] = self.mods.get(mod).score(X, y)
		if verbose:
			print self.accs
		return
	
	#scoring the ensemble on the test data
	def score(self, X, y, method='geometric', threshold=0.5, erbose=True):
		probs = self.predict_proba(X, y)
		if method == 'geometric':
			mean_probs = gmean(probs, axis=1)bv bg
		guesses = [int(x >= threshold) for x in mean_probs]
		acc = np.true_divide(np.sum(guesses == y), len(y))
		return acc
	
	#gets the predicted probabilities of the test data
	def predict_proba(self, X, y):
		probs = pd.DataFrame(np.zeros([X.shape[0], len(self.mods)]))
		probs.columns = self.mods.keys()
		for i in range(len(self.mods)):
			if self.mods.keys()[i] != 'nbsvm':
				probs.iloc[:, i] = self.mods.values()[i].predict_proba(X)[:,1]
			else:
				probs.iloc[:, i] = self.mods['nbsvm'].predict_proba(X, y)['probs']
		return probs
			

		
		