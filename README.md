# document_classification
This is a collection of Python modules for doing sentiment analysis or, considered more broadly, document classification. Generally, each script will vectorize your text (i.e. by some form of bag-of-words) and then train a supervised model to predict which class each of the test documents belongs to. The modules can be executed piece-by-piece from within Python or run as scripts from the command line by specifying a few positional arguments.

## what's included
  1. tools.py includes a holder class for text data objects and a few functions for working with linear models.
  2. nbsvm.py implements Wang and Manning's (2012) interpolated multinomial naive Bayes/support vector machine.
  3. rf.py performs the same task as the NB-SVM, only using a random forest as the classifier instead of an SVM.
  4. lsa.py performs LSA via SVD and either a KNN-classifier or a linear SVM. 
  5. lda.py decomposes the documents into topics via LDA and then classifies them with an SVM.
  6. ensemble.py will combine the votes of a random forest and a NB-SVM to classify the documents.

## using the scripts
All of these scripts take CSV files as their input and convert them to Pandas DataFrames before model training; this is largely to facilitate data transfer between R and Python. As of now, the CSV files must be in a document-level format, i.e. with one row per document and a minimum of two columns, one holding the document text, and the other holding the outcome or target variable. The functions use sklearn's built in count vectorizers to vectorize the text data, which you can manipulate via the command-line arguments in the modeling scripts or through the class-specific methods for each model. 
  
## sample code
This line will train an NB-SVM on the imaginary docs.csv using the values in the 'sentiment' column as the target and the text in the 'text' column as the input. 
```
$ python nbsvm.py '~/data/docs.csv' 'text' 'sentiment' -lm 'no' -ng 3 -sm 'train-test'
```
The optional arguments specify that vectorizer should not limit the number features it considers; that the maximum n-gram size should be 3; and that the data should be split using sklearn's train_test() function. See the readme for [nbsvm.py](docs/nbsvm_README.md) for more details.

## system requirements
To use these modules, you'll need the Pandas, ScikitLearn, and Scipy/Numpy modules in addition to a working installation of Python (the code was written using 2.7.x). The code is also not optimized for speed, so model training may take a while, especially if the dimensionality is particularly high, e.g. with a large number of estimators in a random forest or with a big corpus, or if you perform cross-validation. 

## references
Blei D, Ng A, Jordan M. Latent dirichlet allocation. *Journal of Machine Learning Research*, 3 (Jan), 993-1022. 2003. [pdf](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) 

Platt J. Probabilistic outputs for support vector machines and comparison to regularized likelihood methods. In *Advances in Large Margin Classifiers*. MIT Press. 1999. [pdf](http://citeseer.ist.psu.edu/viewdoc/download;jsessionid=ECEB33C240AB0DAD4A08834BBB95919D?doi=10.1.1.41.1639&rep=rep1&type=pdf)

Landauer TK. Latent semantic analysis. John Wiley & Sons, Ltd; 2006.

Wang S, Manning C. Baselines and bigrams: Simple, good sentiment and topic classificiation. In *Proceedings of ACL 2012*. 2012. [pdf](http://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf)

