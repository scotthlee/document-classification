
"""General functions for training, testing, and tuning document classification models on the autism data"""
import math
import numpy as np
import sklearn
import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier

"""Functions for working with linear models"""
#gets the diagnostic accuracy of a linear classifier; kinda pointless for an SVM, but still interesting
def diagnostics(x, y, w, b, exp=False, cutoff=.5):
    out = pd.DataFrame(np.zeros([1, 11]), columns=diag_names)
    true_targets = y
    if exp:
        curr = prediction(x, w, b, exp=True)
        curr[curr >= cutoff] = 1
        curr[curr < cutoff] = 0
    else:
        curr = prediction(x, w, b)
    curr = curr.reshape(true_targets.shape)
    tab = pd.crosstab(curr, true_targets)
    tp, fp, tn, fn = tab.iloc[1, 1], tab.iloc[1, 0], tab.iloc[0, 0], tab.iloc[0, 1]
    se, sp, ppv, npv = np.true_divide([tp, tn, tp, tn], [tp+fn, tn+fp, tp+fp, tn+fn])
    acc = np.true_divide(tp+tn, tp+tn+fp+fn)
    f = 2 * np.true_divide(se * ppv, se + ppv)
    out.iloc[0,:] = [cutoff, tp, fp, tn, fn, se, sp, ppv, npv, acc, f]
    return out

#returns the predicted outputs based on inputs, training weights, and training bias
#exp=True will exponentiate the predicted values, transforming to [0, 1]
def prediction(x, w, b, neg=0, exp=False):
    guesses = np.matmul(x, w.transpose()) + b
    if exp:
        prediction = np.true_divide(1, 1 + np.exp(-1 * (guesses)))
    else:
        prediction = np.sign(guesses)
        if neg==0:
            prediction[prediction == -1] = 0    
    return prediction

#returns the accuracy of a classifier based on inputs, outputs, training weights, and training bias
def accuracy(x, y, w, b):
    guess = prediction(x, w, b)
    return np.true_divide(np.sum(guess.reshape(y.shape) == y), x.shape[0])

def roc(x, y, w, b, exp=False, cutoff=.5, by=.01):
    exp_guesses = prediction(x, w, b, exp=True)
    th = np.arange(min(exp_guesses) + by, 1, by)
    n = len(th)
    out = pd.DataFrame(np.zeros([n, 11]), columns=diag_names)    
    i = 0
    for cutoff in th:
        out.iloc[i,:] = np.array(diagnostics(x, y, w, b, exp, cutoff))
        i += 1
    return out





