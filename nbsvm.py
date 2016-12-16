
"""NB-SVMs a la Wang and Manning (2012)"""
import argparse
import pandas as pd
import numpy as np
import sklearn
from functions import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, KFold

parser = argparse.ArgumentParser()

#required arguments
parser.add_argument('data', help='path for the input data')
parser.add_argument('y_name', help='name of the column holding the (binary) target variable')
parser.add_argument('x_name', help='name of the column holding the text')

#optional arguments for tuning the classifier's performance
parser.add_argument('-lm', '--limit_features', default='yes', help='limit the number of features for the SVM? (yes or no)')
parser.add_argument('-ft', '--features', type=int, default=10000, help='number of features for the SVM, if limited')
parser.add_argument('-ng', '--ngrams', type=int, default=2, help='max ngram size')
parser.add_argument('-k', '--n_folds', default=10, help='number of folds for cross-validation')
parser.add_argument('-sm', '--split_method', default='train-test', help='how to split the data: train-test or other')
parser.add_argument('-sv', '--split_var', help='variable used for splitting the data')
parser.add_argument('-te', '--test_var', help='values of split_var to be used for testing; all others will be used for training')

args = parser.parse_args()

"""Global values for the SVM"""
num_features = args.features
interpolation_param = 0.25

#column names for the diagnostics data frame; just saving some space
diag_names = ['ctf', 'tp', 'fp', 'tn', 'fn', 'se', 'sp', 'ppv', 'npv', 'acc', 'f']

"""Processing the text and training the SVMs"""
#setting up the vectorizer
if args.limit_features == 'yes':
	print "\nMaking the word-count vectors, using %i features..." %int(num_features)
	vectorizer = CountVectorizer(max_features=num_features, ngram_range=(1, args.ngrams), decode_error='replace', binary=True)
else:
	print "\nMaking the word-count vectors, using all features in the corpus..."
	vectorizer = CountVectorizer(ngram_range=(1, args.ngrams), decode_error='replace', binary=True)

#importing the data
full_set  = pd.read_csv(args.data)
full_corpus = full_set[args.x_name]
full_y = np.array(full_set[args.y_name])

#getting the tfidf matrices for the evaluations
full_fit = vectorizer.fit_transform(full_corpus)
full_names = vectorizer.get_feature_names()
full_counts = full_fit.toarray()

#optional write to csv for the word-count matrix
#pd.DataFrame(full_counts).to_csv('~/data/addm/full_counts.csv')

#invoking the custom splitter to divide the data
print "Splitting the data into training and test sets..."

if args.split_method == 'train-test':
	X_train, X_test, y_train, y_test = train_test_split(full_counts, full_y)

else:
	train_indices = full_set[~full_set[args.split_var].isin([args.test_var])].index.tolist()
	test_indices = full_set[full_set[args.split_var].isin([args.test_var])].index.tolist()
	X_train = full_counts[train_indices, :]
	X_test = full_counts[test_indices, :]
	y_train = full_y[train_indices]
	y_test = full_y[test_indices]
	
X_train_pos = X_train[np.where(y_train == 1)]
X_train_neg = X_train[np.where(y_train == 0)]

r_hat = log_count_ratio(X_train_pos, X_train_neg)
X_train_nb = np.multiply(r_hat, X_train)

#setting up the test data; only evals from 2008
X_test_nb = np.multiply(r_hat, X_test)

#setting the npos and nneg variables
n_pos = X_train_pos.shape[0]
n_neg = X_train_neg.shape[0]
nb_bias = np.log(np.true_divide(n_pos, n_neg))

#training the standard SVM without NB features
print "Training the SVM..."
lsvc = LinearSVC()
lsvc.fit(X_train, y_train)
lsvc_acc = lsvc.score(X_test, y_test)

#training the SVM with NB features but no interpolation
print "Training the NB-SVM..."
nbsvm = LinearSVC()
nbsvm.fit(X_train_nb, y_train)
trained_weights = nbsvm.coef_
trained_bias = nbsvm.intercept_
nbsvm_acc = nbsvm.score(X_test_nb, y_test)

#finding the optimal interpolation paramater
int_accs = tune_beta(X_test_nb, y_test, trained_weights, nb_bias, np.arange(0, 1, .025))

print "\nResults:"
print "MNB accuracy is %0.4f" %accuracy(X_test, y_test, r_hat, nb_bias)
print "SVM accuracy is %0.4f" %lsvc_acc
print "NB-SVM accuracy is %0.4f" %nbsvm_acc
print "Interpolated model accuracy is %0.4f" %int_accs[np.argsort(int_accs[:,1])[-1], 1]
print "Best interpolation parameter is %s\n" %int_accs[np.argsort(int_accs[:,1])[-1], 0]

#pulling out the incorrect examples
best_beta = int_accs[np.argsort(int_accs[:,1])[-1], 0]
best_guesses = prediction(X_test_nb, interpolate(trained_weights, best_beta), nb_bias).reshape(y_test.shape)


