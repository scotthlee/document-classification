# Ensemble methods

This script allows you to implement model averaging to perform document classification, combining information from ```TextRF```s, ```TextNBSVM```s, and other ScikitLearn-inspired classifiers to (hopefully) get you higher accuracy on your test data. It trains the models separately--not jointly--and then combines their predicted probabilities for the test data using one of a few simple methods (the default is the geometric mean). 

##Classes
Model averaging is done by way of the ```Ensemble``` class, which will hold the classifiers in the ensemble, fit them separately to the (same) training data, and then score them both separately and as a group on the test data. You can add models to an ```Ensemble()``` instance with ```.add()```, and you can remove them with ```.remove()```. When you're ready to proceed, you can train and test the models with single calls to the ```.fit()``` and ```.score()``` functions, respectively. Other options are ```.score_sep()```, which will return the accuracies of the models individually; and ```.predict_proba()```, which will return a Pandas ```DataFrame``` with the predicted probabilities for the test data for each model. 

Other methods for combining model votes will be added soon.

##Example code
```$ python ensemble.py 'data.csv' 'corpus' 'sentiment'``` will train a RF-NBSVM ensemble on the 'corpus' and combine their votes to predict the scores in 'sentiment'. 

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

