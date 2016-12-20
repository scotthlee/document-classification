"Scott's script for training a random forest on the 2006 and 2008 ADDM evaluations" 
import pandas as pd
import numpy as np
import sklearn
import argparse
from functions import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold

#performs a stepwise search for the optimal number of features in a trimmed random forest
def tune(rf, min_features=1, max_features=200):
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

#main class for the text-based random forest; has methods for loading and processing data,
#as well as model-specific attributes like accuracy and feature names
class TextRF:
	def __init__(self):
		self.accuracy = 0.0
		self.shape = {'n_trees': 0, 'n_features': 0}
		self.tuning_curve = pd.DataFrame()
		self.feature_names = []
		self.trimmed_features = []
		self.top_features = []
		
		#setting attributes for the data
		self.data = pd.DataFrame()
		self.X, self.y = [], []
		self.X_train, self.X_test = [], []
		self.y_train, self.y_test = [], []
	
	#simple wrapper for saving the data frame to the RF object
	def set_data(self, df):
		self.data = df
		return

	def print_top_features(self):
		for i in xrange(len(self.top_features)):
			print self.top_features[i]
		return

	#another wrapper for the vectorization functions; optional, and will take a while
	def process(self, df, x_name, y_name, method='tfidf', max_features=10000):
		print "Vectorizing the corpus..."
		if method == 'tfidf':
			vectorizer = TfidfVectorizer(max_features)
			full_fit = vectorizer.fit_transform(df[x_name])
			full_counts = full_fit.toarray()
			self.X = full_counts
			self.y = np.array(df[y_name])
			self.feature_names = vectorizer.get_feature_names()
		self.split(self.X, self.y, self.feature_names)
		return
	
	#splits the data into training and test sets; either called from self.process()
	#or on its own when your text is already vectorized and divided into x and y
	def split(self, x, y, names, split_method='train-test', split_var=None, test_val=None):
		self.X = x
		self.y = y
		self.feature_names = names
		if split_method == 'train-test':
			self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y)
		elif split_method == 'var':
			self.X_train, self.X_test, self.y_train, self.y_test = split_by_var(x, y, data)

	#main function for training and testing the random forest
	def run(self, trees=1000, top=100, verbose=True):
		
		#setting the shape parameter
		self.shape = [trees, top]
		
		#training the RF on the docs
		print "Training the random forest..."
		rf = RandomForestClassifier(n_estimators=trees)
		rf_train = rf.fit(self.X_train, self.y_train)
		
		#scoring the trained model
		rf_test = rf.score(self.X_test, self.y_test)
		
		#trimming the tree to the top 90 features
		all_features = rf_train.feature_importances_
		sorted_features = np.argsort(all_features)[-top:]
		sorted_features_1000 = np.argsort(all_features)[-1000:]
		
		full_trimmed = self.X[:, sorted_features]
		train_trimmed = self.X_train[:, sorted_features]
		test_trimmed = self.X_test[:, sorted_features]
		
		rf_trimmed = RandomForestClassifier(n_estimators=trees)
		rf_trimmed_train = rf_trimmed.fit(train_trimmed, self.y_train)
		rf_trimmed_test = rf_trimmed.score(test_trimmed, self.y_test)
		
		if verbose:
			print "\nResults:"
			print "Raw model accuracy is %0.4f" %rf_test
			print "Trimmed model accuracy is %0.4f\n" %rf_trimmed_test
		
		self.accuracy = rf_trimmed_test
		#self.tuning_curve = tune(rf, X_train, y_train, X_test) 

#running an example of the model
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	#positional arguments
	parser.add_argument('data', help='data path for the corpus (in CSV format)')
	parser.add_argument('x_name', help='name of the column holding the text')
	parser.add_argument('y_name', help='name of the column holding the target variable')
	
	#optional arguments
	parser.add_argument('-lm', '--limit_features', default='yes', help='limit the number of features for the RF? (yes or no)')
	parser.add_argument('-ft', '--features', type=int, default=10000, help='number of features for the SVM, if limited')
	parser.add_argument('-tr', '--n_trees', type=int, default=1000, help='number of trees to use in the RF')
	parser.add_argument('-ng', '--ngrams', type=int, default=2, help='max ngram size')
	parser.add_argument('-sm', '--split_method', default='train-test', help='split the data by year, train-test, or cross-val')
	parser.add_argument('-sv', '--split_variable', default='year', help='variable to used for splitting the data')
	parser.add_argument('-yr', '--test_val', help='which value of split_variable to use for the test data')
	
	args = parser.parse_args()
	
	#loading the data and training the RF
	df = pd.read_csv(args.data)
	mod = TextRF()
	mod.process(df, args.x_name, args.y_name)
	mod.run(trees=args.n_trees)


