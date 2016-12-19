"Document classification with a random forest" 
import pandas as pd
import numpy as np
import math
import sklearn
import argparse
from generic_functions import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold


#prints the top n features of a random forest by feature importance
def top_features(features, names, n=10, reverse=False): 
    start = 0    
    stop = n-1
    print "Top %s features are:" %n
    if reverse:
        stop = len(features)
        start = stop - n-1
    for index in features[start:stop]:
        print(names[index])

#performs a stepwise search for the optimal number of features in a random forest;
#not implemented by default, but may be used for further model customization
def tune_forests(full_model, x_train, y_train, x_test, y_test, min_features=1, max_features=200, n_estimators=1000):
    out = pd.DataFrame(data=np.empty([max_features - min_features + 1, 2]), columns=['n_features', 'acc'])
    out['n_features'] =  range(min_features, max_features + 1)    
    for i in range(min_features, max_features + 1):
        all_ftrs = full_model.feature_importances_
        sort_ftrs = np.argsort(all_ftrs)[-i:]
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf = clf.fit(x_train[:, sort_ftrs], y_train)
        acc = clf.score(x_test[:, sort_ftrs], y_test)
        print "Number of features: %s" %i
        print "Model accuracy: %s \n" %acc
        out['acc'][i - min_features] = acc
    return out

if __name__ == '__main__':

	#setting up the command-line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('data', help='path for the input data')
	parser.add_argument('y_name', help='name of the column holding the (binary) target variable')
	parser.add_argument('x_name', help='name of the column holding the text')
	parser.add_argument('-cr', '--cross_val', default='no', help='perform cross-validation on the training data?')
	parser.add_argument('-k', '--n_folds', type=int, default=10, help='number of folds for cross-validation')
	parser.add_argument('-ft', '--features', type=int, default=10000, help='max number of features to consider')
	parser.add_argument('-ng', '--ngrams', type=int, default=3, help='max size of ngrams to calculate')
	parser.add_argument('-vc', '--vectorizer', default='tfidf', help='which vectorizer to use')
	parser.add_argument('-tr', '--trees', type=int, default=1000, help='how many trees to use in the random forest')
	parser.add_argument('-tp', '--top', type=int, default=100, help='how many top features to use when trimming the trees')
	parser.add_argument('-sm', '--split_method', default='train-test', help='how to split the data')
	parser.add_argument('-sv', '--split_var', help='variable used for splitting the data')
	parser.add_argument('-te', '--test_val', help='values of split_var to be used for testing')

	args = parser.parse_args()

	#tokenizing the text and fitting the RF to the training data
	#vectorizer = CountVectorizer(max_features=10000, ngram_range=(1, 2), decode_error='replace', binary=True)
	if args.vectorizer == 'tfidf':
		vectorizer = TfidfVectorizer(max_features=args.features, 
				ngram_range=(1, args.ngrams), decode_error='replace')
	elif args.vectorizer == 'counts':
		vectorizer = CountVectorizer(max_features=args.features, ngram_range=(1, args.ngrams), 
				decode_error='replace', binary=False)
	elif args.vectorizer == 'binary counts':
		vectorizer = CountVectorizer(max_features=args.features, ngram_range=(1, args.ngrams),
				decode_error='replace', binary=True)

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
		full_set[args.split_var] = [str(val) for val in full_set[args.split_var]]	
		train_indices = full_set[~full_set[args.split_var].isin([args.test_val])].index.tolist()
		test_indices = full_set[full_set[args.split_var].isin([args.test_val])].index.tolist()
		print train_indices
		X_train = full_counts[train_indices, :]
		X_test = full_counts[test_indices, :]
		y_train = full_y[train_indices]
		y_test = full_y[test_indices]

	print "Training the random forest..."
	rf = RandomForestClassifier(n_estimators=args.trees)
	rf_train = rf.fit(X_train, y_train)

	#scoring the trained model
	rf_test = rf.score(X_test, y_test)

	#trimming the tree to the top 90 features
	all_features = rf_train.feature_importances_
	sorted_features = np.argsort(all_features)[-args.top:]
	sorted_features_1000 = np.argsort(all_features)[-1000:]

	full_trimmed = full_counts[:, sorted_features]
	train_trimmed = X_train[:, sorted_features]
	test_trimmed = X_test[:, sorted_features]

	rf_trimmed = RandomForestClassifier(n_estimators=args.trees)
	rf_trimmed_train = rf_trimmed.fit(train_trimmed, y_train)
	rf_trimmed_test = rf_trimmed.score(test_trimmed, y_test)

	if args.cross_val == 'yes':
		print "Performing %i-fold cross-validation..." %args.n_folds
		rf_cross = cross_val_score(rf, X_train, y_train, cv=args.n_folds)
		rf_trimmed_cross = cross_val_score(rf_trimmed, X_train, y_train, cv=args.n_folds)
		print "\nResults:"
		print "With the full model, cross-val accuracy is %0.4f, and test accuracy is %0.4f" %(rf_cross.mean(), rf_test)
		print "With the trimmed model, cross-val accuracy is %0.4f, and test accuracy is %0.4f\n" %(rf_trimmed_cross.mean(), rf_trimmed_test)

	else:
		print "\nResults:"
		print "Raw model accuracy is %0.4f" %rf_test
		print "Trimmed model accuracy is %0.4f\n" %rf_trimmed_test

	top_features(sorted_features, full_names, reverse=True)


