# Multinomial Naive Bayes + SVM

This script runs a few models, starting with a pure MNB and leading up to the interpolated MNB-SVM Wang and Manning (2012) report as their best-performing across all tasks. Generally, MNB will work better on short snippets of text, and the SVM will work better on paragraphs and full-length documents. This script will automatically tune the interpolation parameter, though, and will return the interpolated model with the highest accuracy on the test data.

## Classes and Functions

There are two models included here: the standard MNB classifier, and the (M)NB-SVM. Each gets its own class, which can be instantiated by ```TextMNB()``` and ```TextNBSVM()```, respectively. Like the other classes in this module, these guys follow the ScikitLearn style of training and testing, so they can be fit and scored using the standard ```.fit()``` and ```.score()``` methods. If your corpus is already converted to a binary document-term matrix, you can fit the models directly; otherwise, process the text first using the ```TextData()``` class and ```.process()``` and ```.split()``` methods included in [generic.py](generic_README.md).

## Command-line Arguments

Like the other scripts in this module, there are three positional arguments required to run the script:
 
  1. ```data```: the file path for the CSV holding both the raw text and the target variable
  2. ```x_name```: the name of the column holding the text
  3. ```y_name```: the name of the column holding the target variable
 
There are also a number of optional arguments that allow for model customization and tuning:

  1. ```-lm, --limit_features```: If 'yes', this will limit the size of the feature vectors for each document to whatever is  specified by the -ft, --max_features argument. Most authors will use the full feature set (i.e. corpus vocabulary) when reporting benchmark results, but reducing it will make training faster and will in some cases improve your accuracy. The default is 'yes'.
  2. ```-ft, --max_features```: As above; the number of maximum features for the vectorizer to calculate (default is 10,000)
  3. ```-ng, --ngrams```: The maximum size of ngrams to consider. Note that the minimum size is always 1, and the default is 2.
  4. ```-sm, --split_method```: The method for splitting the data into training, validation, and test sets. The default is 'train-test', which calls sklearn's train_test_split() function, but 'cross-val' and 'other' maybe also be used. The second option will perform the same train-test split but report the mean cross-validation accuracy during training; and the third will split the data according to the levels of the user-specified column variable, like 'year' or 'sex'.
  5. ```-sv, --split_var```: The column variable to be used for splitting the data when -sm, --split_method is 'other'.
  6. ```-te, --test_val```: The level of -sv, --split_var to use for the test data, with all other levels being used for training and, when selected, cross-validation.
