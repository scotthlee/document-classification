
"""Uses TF-IDF, SVD, and cosine distance to classify documents"""
import sklearn
import scipy
import numpy as np
import pandas as pd
import argparse
import tools

from tools import *
from sklearn.decomposition import TruncatedSVD
from copy import deepcopy
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

#simple wrapper for the truncated SVD; allows for matrix reshaping
def decompose(doc_vecs, n_features=100, normalize=False, flip=False):
	svd = TruncatedSVD(n_features)	
	if normalize:	
		if flip:
			lsa = make_pipeline(svd, Normalizer(copy=False))
			doc_mat = lsa.fit_transform(doc_vecs.transpose())
			doc_mat = doc_mat.transpose()
		else:
			lsa = make_pipeline(svd, Normalizer(copy=False))		
			doc_mat = lsa.fit_transform(doc_vecs)
		return doc_mat
	else:
		if flip:
			doc_mat = svd.fit_transform(doc_vecs.transpose())
			doc_mat = doc_mat.transpose()
		else:
			doc_mat = svd.fit_transform(doc_vecs)
		return doc_mat

class TextLSA:
	def __init__(self, n_features=100, transform=True, classifier='lsvm', kernel='rbf', n_neighbors=5):
		self.n_features = n_features
		self.transform = transform
		self.clf_type = classifier
		if classifier == 'lsvm':
			self.clf = LinearSVC()
		elif classifier == 'svm':
			self.clf = SVC(kernel=kernel, probability=True)
		elif classifier == 'knn':
			self.clf = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='brute', metric='cosine')
	
	def fit(self, x, y, normalize=False, flip=False):
		X = deepcopy(x)
		if self.transform:
			X = decompose(X, self.n_features, normalize, flip)
		self.clf.fit(X, y)
	
	def predict(self, x, normalize=False, flip=False):
		X = deepcopy(x)
		if self.transform:
			X = decompose(X, self.n_features, normalize, flip)
		return self.clf.predict(X)
	
	def predict_proba(self, x, y, normalize=False, flip=False):
		X = deepcopy(x)
		if self.transform:
			X = decompose(X, self.n_features, normalize, flip)
		if self.clf_type == 'svm':
			return self.clf.predict_proba(X)
		elif self.clf_type in ['lsvm', 'nbsvm']:
			return platt_scale(X, y, self.clf)
		elif self.clf_type == 'knn':
			return self.clf.predict_proba(X)
	
	def score(self, x, y, normalize=False, flip=False):
		X = deepcopy(x)		
		if self.transform:
			X = decompose(X, self.n_features, normalize, flip)
		return self.clf.score(X, y)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('data', help='path for the data')
	parser.add_argument('x_name', help='name of the column holding the text')
	parser.add_argument('y_name', help='name of the column holding the target variable')
	parser.add_argument('-vm', '--vec_method', default='tfidf', help='vectorize by counts or tfidf?')
	parser.add_argument('-cl', '--classifier', default='knn', help='which classifier to use')
	parser.add_argument('-sm', '--split_method', default='train-test', help='how to split the data')
	parser.add_argument('-sv', '--split_variable', help='column var for splitting the data')
	parser.add_argument('-tv', '--test_value', help='value of --split_variable to use as test data')
	parser.add_argument('-kn', '--num_neighbors', type=int, default=5, help='number of nearest neighbors')
	parser.add_argument('-lm', '--limit_features', default=True, help='limit the number of features to consider?')
	parser.add_argument('-ft', '--num_features', type=int, default=10000, help='max number of features to consider, if limited')
	parser.add_argument('-ng', '--ngrams', type=int, default=2, help='size of ngrams to consider')
	parser.add_argument('-nc', '--n_components', type=int, default=100, help='number of dimensions for SVD')
	
	args = parser.parse_args()
	
	corpus = pd.read_csv(args.data)
	d = TextData()
	if args.limit_features:
		d.process(corpus, args.x_name, args.y_name, ngrams=args.ngrams, 
					max_features=args.num_features, method=args.vec_method)
	else:
		d.process(corpus, args.x_name, args.y_name, ngrams=args.ngrams, 
					max_features=None, method=args.vec_method)
	
	d.split(split_method=args.split_method, split_var=args.split_variable, test_val=args.test_value)
	X_train, X_test, y_train, y_test = d.X_train, d.X_test, d.y_train, d.y_test
	
	print "Performing the SVD..."
	#fetching the SVD document matrices
	train_svd = decompose(X_train, n_features=args.n_components)
	test_svd = decompose(X_test, n_features=args.n_components)
	
	#fitting a classifier to the training data; options are KNN or a linear SVM
	if args.classifier == 'knn':
		clf = KNeighborsClassifier(n_neighbors=args.num_neighbors, algorithm='brute', metric='cosine')
		clf.fit(train_svd, y_train)
	
	elif args.classifier == 'svm':
		clf = LinearSVC()
		clf.fit(train_svd, y_train)
	
	guesses = clf.predict(test_svd)
	acc = clf.score(test_svd, y_test)
	
	print "\nResults:"
	print "Test accuracy is %0.4f" %acc
	
