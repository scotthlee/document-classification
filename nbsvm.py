
"""Scott's script for training SVMs a la Wang and Manning (2012) on the 2006 and 2008 ADDM evaluations"""
import argparse
import pandas as pd
import numpy as np
import sklearn
from functions import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, KFold

#calculates the log-count ratio r
def log_count_ratio(pos_text, neg_text, alpha=1):
    p = np.add(alpha, np.sum(pos_text, axis=0))
    q = np.add(alpha, np.sum(neg_text, axis=0))
    p_norm, q_norm = np.sum(p), np.sum(q)
    p_ratio = np.true_divide(p, p_norm)
    q_ratio = np.true_divide(q, q_norm)
    r = np.log(np.true_divide(p_ratio, q_ratio))
    return r

def nb_svm(x, y, w, b, C=1):
    wt = w.transpose()
    y = y.reshape(y.shape[0], 1)
    l2_loss = np.square(np.maximum(0, 1 - y * (np.matmul(x, wt) + b)))
    return np.matmul(w, wt) + C * np.sum(l2_loss)

#returns interpolated weights for constructing the nb-svm
def interpolate(w, beta):
    return ((1 - beta) * (np.sum(w) / w.shape[1])) + (beta * w)

#finds the interpolation paramater beta that yields the highest accuracy
def tune_beta(x, y, w, b, betas):
    n = len(betas)
    results = np.zeros([n, 2])
    results[:,0] = betas
    for i in range(0, n):
        int_weights = interpolate(w, betas[i])
        results[i,1] = accuracy(x, y, int_weights, b)
    return results

def tune_C(x_tr, y_tr, x_te, y_te, c_params, beta=0.25, interpolate=False):
    out = pd.DataFrame(np.zeros([len(c_params), 11]), columns=diag_names)
    i = 0
    for c_param in c_params:
        print c_param
        clf = LinearSVC(C=c_param).fit(x_tr, y_tr)
        w, b = clf.coef_, clf.intercept_
        if interpolate:
            w = interpolate(clf.coef_, beta)
            b = nb_bias
        out.iloc[i,:] = np.array(diagnostics(x_te, y_te, w, b))
        i += 1
    out.iloc[:,0] = c_params
    return out

#main class for the NB-SVM
class NBSVM:
	def __init__(self):
		#setting basic parameters for model evaluation
		self.accuracy = {}
		self.predictions = {}
		
		#setting attributes for the data
		self.data = pd.DataFrame()
		self.X, self.y = [], []
		self.X_train, self.X_test = [], []
		self.y_train, self.y_test = [], []
		self.X_train_pos, self.X_train_neg = [], []
		self.X_train_nb, self.X_test_nb = [], []

		#setting attributes for the NBSVM
		self.r = 0.0
		self.nb_bias = 0.0

	#simple wrapper for saving the data frame to the RF object
	def set_data(self, df):
		self.data = df
		return
	
	#another wrapper for the vectorization functions; optional, and will take a while
	def process(self, df, x_name, y_name, max_features=10000, limit='yes'):
		print "Vectorizing the corpus..."
		if limit == 'yes':
			vectorizer = CountVectorizer(max_features, decode_error='replace', binary=True)
		elif limit == 'no':
			vectorizer = CountVectorizer(decode_error='replace', binary=True)

		full_fit = vectorizer.fit_transform(df[x_name])
		full_counts = full_fit.toarray()
		self.X = full_counts
		self.y = np.array(df[y_name])
		self.split(self.X, self.y)
		return
	
	#splits the data into training and test sets; either called from self.process()
	#or on its own when your text is already vectorized and divided into x and y
	def split(self, x, y, split_method='train-test', split_var=None, test_val=None):
		self.X = x
		self.y = y
		
		if split_method == 'train-test':
			self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y)
		elif split_method == 'var':
			self.X_train, self.X_test, self.y_train, self.y_test = split_by_var(x, y, data)
		
		#splitting by target class so we can calculate the log-count ratio
		X_train_pos = self.X_train[np.where(self.y_train == 1)]
		X_train_neg = self.X_train[np.where(self.y_train == 0)]
		
		r = log_count_ratio(X_train_pos, X_train_neg)
		self.X_train_nb = np.multiply(r, self.X_train)
		self.X_test_nb = np.multiply(r, self.X_test)

		#setting the npos and nneg variables
		n_pos = X_train_pos.shape[0]
		n_neg = X_train_neg.shape[0]
		
		#passing the attribtues up to the instance
		self.r = r
		self.X_train_pos, self.X_train_neg = X_train_pos, X_train_neg
		self.nb_bias = np.log(np.true_divide(n_pos, n_neg))

	def run(self, verbose=True):
		
		#accuracy of the regular MNB
		mnb_acc = accuracy(self.X_test, self.y_test, self.r, self.nb_bias)
		mnb_pred = prediction(self.X_test, self.r, self.nb_bias)

		#training the standard SVM without NB features
		print "Training the SVM..."
		lsvc = LinearSVC()
		lsvc.fit(self.X_train, self.y_train)
		lsvc_acc = lsvc.score(self.X_test, self.y_test)
		lsvc_pred = lsvc.predict(self.X_test)

		#training the SVM with NB features but no interpolation
		print "Training the NB-SVM..."

		nbsvm = LinearSVC()
		nbsvm.fit(self.X_train_nb, self.y_train)
		trained_weights = nbsvm.coef_
		trained_bias = nbsvm.intercept_
		nbsvm_acc = nbsvm.score(self.X_test_nb, self.y_test)
		nbsvm_pred = nbsvm.predict(self.X_test_nb)
		
		#finding the optimal interpolation paramater
		int_accs = tune_beta(self.X_test_nb, self.y_test, trained_weights, self.nb_bias, np.arange(0, 1, .025))
		inter_acc = int_accs[np.argsort(int_accs[:,1])[-1], 1]
		best_beta = int_accs[np.argsort(int_accs[:,1])[-1], 0]
		inter_pred = prediction(self.X_test_nb, interpolate(trained_weights, best_beta), self.nb_bias).reshape(self.y_test.shape)
		
		#putting all the accuracy stats in one place
		self.accuracy = {'mnb':mnb_acc, 'svm':lsvc_acc, 'nbsvm':nbsvm_acc, 'inter':inter_acc}
		self.predictions = {'mnb':mnb_pred, 'svm':lsvc_pred, 'nbsvm':nbsvm_pred, 'inter':inter_pred}
		
		if verbose:
			print "\nResults:"
			print "MNB accuracy is %0.4f" %mnb_acc
			print "SVM accuracy is %0.4f" %lsvc_acc
			print "NB-SVM accuracy is %0.4f" %nbsvm_acc
			print "Interpolated model accuracy is %0.4f" %inter_acc
			print "Best interpolation parameter is %s\n" %best_beta

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	
	#positional arguments
	parser.add_argument('data', help='path for the input data')
	parser.add_argument('x_name', help='name of the column holding the text')
	parser.add_argument('y_name', help='name of the column holding the target values')

	#optional arguments for tuning
	parser.add_argument('-lm', '--limit_features', default='yes', help='limit the number of features for the SVM? (yes or no)')
	parser.add_argument('-ft', '--features', type=int, default=35000, help='number of features for the SVM, if limited')
	parser.add_argument('-ng', '--ngrams', type=int, default=2, help='max ngram size')
	parser.add_argument('-sm', '--split_method', default='train-test', help='split the data by year, train-test, or cross-val')
	parser.add_argument('-sv', '--split_variable', default='year', help='which variable to use for splitting')
	parser.add_argument('-tv', '--test_value', default=2008, help='which value of --split_variable to use for testing')
	args = parser.parse_args()

	#loading the data and training the RF
	df = pd.read_csv(args.data)
	mod = NBSVM()
	
	if args.limit_features == 'yes':
		mod.process(df, args.x_name, args.y_name, max_features=args.features)
	else:
		mod.process(df, args.x_name, args.y_name, limit='no')
	
	mod.run()
	

