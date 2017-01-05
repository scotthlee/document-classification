import pandas as pd
import numpy as np
import lda
import tools
import argparse

from tools import *
from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition import LatentDirichletAllocation as LDA

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	#positional arguments
	parser.add_argument('data', default='~/data/addm/corpus_with_lemmas.csv', help='path for the input data')
	parser.add_argument('x_name', default='dx', help='name of the column holding the text')
	parser.add_argument('y_name', default='aucaseyn', help='name of the column holding the target values')

	#optional arguments for tuning
	parser.add_argument('-lm', '--limit_features', type=bool, default=True, help='limit the number of features for the SVM? (yes or no)')
	parser.add_argument('-ft', '--features', type=int, default=35000, help='number of features for the SVM, if limited')
	parser.add_argument('-ng', '--ngrams', type=int, default=2, help='max ngram size')
	parser.add_argument('-tp', '--topics', type=int, default=10, help='number of topics for LDA')
	parser.add_argument('-sm', '--split_method', default='train-test', help='split the data by var(iable), train-test, or cross-val')
	parser.add_argument('-sv', '--split_variable', default='year', help='which variable to use for splitting')
	parser.add_argument('-tv', '--test_value', default=2008, help='which value of --split_variable to use for testing')
	parser.add_argument('-vb', '--verbose', default=True, help='should functions print updates as they go?')
	args = parser.parse_args()

	#loading and processing the data
	df = pd.read_csv(args.data)
	d = TextData()
	if args.limit_features:
		d.process(df, args.x_name, args.y_name, max_features=args.features, verbose=args.verbose)
	else:
		d.process(df, args.x_name, args.y_name, max_features=None, verbose=args.verbose)
	
	#fitting the LDA model
	if args.verbose:
		print "Fitting the LDA model..."
	lda = LDA(learning_method='online')
	topics = lda.fit_transform(d.X)
	
	#feeding the LDA matrix back to the TextData() object
	d.set_xy(topics, d.y)
	
	#getting the training and test sets	
	if args.verbose:
		print "Splitting the data into training and test sets..."
	d.split(args.split_method, args.split_variable, args.test_value)
	
	#modeling setting up the classifier
	if args.verbose:
		print "Training the SVM..."
	clf = LinearSVC()
	clf.fit(d.X_train, d.y_train)
	acc = clf.score(d.X_test, d.y_test)
	print "\nResults:\nAccuracy with the SVM on LDA is %0.4f" %acc
	
	
