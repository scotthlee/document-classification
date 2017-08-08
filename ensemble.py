'''A simple script for making ensemble classifiers with random forests and NBSVMs'''
import pandas as pd
import numpy as np
import re

from scipy.stats import gmean
from tools import *
from nbsvm import NBSVM
from rf import TextRF

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
		if 'sklearn' in model.__module__:
			modname = re.sub('sklearn', '', model.__module__)
			self.mods[modname] = model
			self.accs[modname] = 0.0
		else:
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
	def score(self, X, y, method='geometric', threshold=0.5):
		probs = self.predict_proba(X, y)
		if method == 'geometric':
			mean_probs = gmean(probs, axis=1)
		guesses = [int(x >= threshold) for x in mean_probs]
		acc = np.true_divide(np.sum(guesses == y), len(y))
		return acc
	
	#predicting results with the test data
	def predict(self, X, y, method='geometric', threshold=0.5):
		probs = self.predict_proba(X, y)
		if method == 'geometric':
			mean_probs = gmean(probs, axis=1)
		guesses = [int(x >= threshold) for x in mean_probs]
		return np.array(guesses)
	
	#gets the predicted probabilities of the test data
	def predict_proba(self, X, y, mean=False):
		probs = pd.DataFrame(np.zeros([X.shape[0], len(self.mods)]))
		probs.columns = self.mods.keys()
		for i in range(len(self.mods)):
			if self.mods.keys()[i] != 'nbsvm':
				probs.iloc[:, i] = self.mods.values()[i].predict_proba(X)[:,1]
			else:
				probs.iloc[:, i] = self.mods['nbsvm'].predict_proba(X, y)
		#probs[probs == 0] = 0.0000001
		if mean:
			return gmean(probs, axis=1)
		else:
			return probs
			
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	#positional arguments
	parser.add_argument('data', help='path for the input data')
	parser.add_argument('x_name', help='name of the column holding the text')
	parser.add_argument('y_name', help='name of the column holding the target values')

	#optional arguments for tuning
	parser.add_argument('-lm', '--limit_features', type=bool, default=True, help='limit the number of features?')
	parser.add_argument('-ft', '--features', type=int, default=35000, help='number of features, if limited')
	parser.add_argument('-ng', '--ngrams', type=int, default=2, help='max ngram size')
	parser.add_argument('-vm', '--vote_method', default='geometric', help='how to combine the class probabilities for scoring')
	parser.add_argument('-vc', '--vectorizer', default='tfidf', help='how to vectorize the corpus')
	parser.add_argument('-sm', '--split_method', default='train-test', help='split the data by var(iable), train-test, or cross-val')
	parser.add_argument('-sv', '--split_variable', help='which variable to use for splitting')
	parser.add_argument('-tv', '--test_value', help='which value of --split_variable to use for testing')
	parser.add_argument('-vb', '--verbose', default=True, help='should functions print updates as they go?')
	args = parser.parse_args()

	#loading and processing the data
	df = pd.read_csv(args.data)
	d = TextData()
	if args.limit_features:
		d.process(df, args.x_name, args.y_name, method=args.vectorizer, max_features=args.features, verbose=args.verbose)
	else:
		d.process(df, args.x_name, args.y_name, method=args.vectorizer, max_features=None, verbose=args.verbose)
	
	#getting the training and test sets
	d.split(args.split_method, args.split_variable, args.test_value)
	
	#adding the models
	ens = Ensemble()
	ens.add(TextNBSVM())
	ens.add(TextRF())
	ens.fit(d.X_train, d.y_train)
	ens.score_sep(d.X_test, d.y_test, verbose=False)
	acc = ens.score(d.X_test, d.y_test, method=args.vote_method)
	if args.verbose:
		print '\nResults:'
		print ens.accs
		print 'Ensemble accuracy is %0.4f' %acc
	
