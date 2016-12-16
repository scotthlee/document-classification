# document_classification
This is a collection of scripts for doing sentiment analysis (i.e. document classification). 

# general guidelines for use
All of these functions take CSV files as their input and convert them to Pandas DataFrames before model training. As of now, the files must be in a document-level format, i.e. with one row per document and a minimum of two columns, one holding the document text, and the other holding the outcome or target variable (e.g. positive-negative, 1-0, yes-no, etc.). The functions use sklearn's built in count vectorizers to vectorize the text data, which you can manipulate via the command-line arguments in the modeling scripts.    

# included scripts
  1. functions.py includes functions for training, testing, and tuning supervised document classification models
  2. nbsvm.py implements Wang and Manning's (2012) interpolated multinomial naive Bayes/support vector machine to.
  3. rf.py performs the same task as the NB-SVM, only using a random forest as the classifier instead of an SVM.

# system requirements
In addition to a working installation of Python 2.7.x, you'll need the Pandas, ScikitLearn, and Scipy/Numpy modules installed to run these scripts. 

