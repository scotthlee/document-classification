
"""Uses TF-IDF, SVD, and cosine distance to classify documents"""
import sklearn
import scipy
import numpy as np
import pandas as pd
import argparse
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import *

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
		
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('data', help='path for the data')
	parser.add_argument('y_name', help='name of the column holding the target variable')
	parser.add_argument('x_name', help='name of the column holding the text')
	parser.add_argument('-cl', '--classifier', default='knn', help='which classifier to use')
	parser.add_argument('-sm', '--split_method', default='train-test', help='how to split the data')
	parser.add_argument('-kn', '--num_neighbors', type=int, default=5, help='number of nearest neighbors')
	parser.add_argument('-lm', '--limit_features', default='yes', help='limit the number of features to consider?')
	parser.add_argument('-ft', '--num_features', type=int, default=10000, help='max number of features to consider, if limited')
	parser.add_argument('-ng', '--ngrams', type=int, default=2, help='size of ngrams to consider')
	parser.add_argument('-nc', '--n_components', type=int, default=100, help='number of dimensions for SVD')
	
	args = parser.parse_args()
	
	if args.limit_features == 'yes':
		vectorizer = TfidfVectorizer(max_features=args.num_features, ngram_range=(1, args.ngrams), decode_error='replace')
	else:
		vectorizer = TfidfVectorizer(ngram_range=(1, args.ngrams), decode_error='replace')

	full_set = pd.read_csv(args.data)
	full_corpus = full_set[args.x_name]
	full_y = full_set[args.y_name]

	#getting the tfidf matrices for the evaluations
	print "\nVectorizing the corpus..."
	full_fit = vectorizer.fit_transform(full_corpus)
	full_names = vectorizer.get_feature_names()
	full_counts = full_fit.toarray()
	
	if args.split_method == 'train-test':
		X_train, X_test, y_train, y_test = train_test_split(full_counts, full_y)

	else:
		train_indices = full_set[~full_set[args.split_var].isin([args.test_var])].index.tolist()
		test_indices = full_set[full_set[args.split_var].isin([args.test_var])].index.tolist()
		X_train = full_counts[train_indices, :]
		X_test = full_counts[test_indices, :]
		y_train = full_y[train_indices]
		y_test = full_y[test_indices]
	
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
	
