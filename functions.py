
"""Functions for training, testing, and tuning document classification models on the autism data"""
import math
import numpy as np
import sklearn
import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier

"""Random forest functions"""
def top_features(features, names, n=10, reverse=False): 
    start = 0    
    stop = n-1
    print "Top %s features are:" %n
    if reverse:
        stop = len(features)
        start = stop - n-1
    for index in features[start:stop]:
        print(names[index])

#performs a stepwise search for the optimal number of features in a random forest
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

"""NB-SVM functions"""
#calculates the log-count ratio r
def log_count_ratio(pos_text, neg_text, alpha=1):
    p = np.add(alpha, np.sum(pos_text, axis=0))
    q = np.add(alpha, np.sum(neg_text, axis=0))
    p_norm, q_norm = np.sum(p), np.sum(q)
    p_ratio = np.true_divide(p, p_norm)
    q_ratio = np.true_divide(q, q_norm)
    r = np.log(np.true_divide(p_ratio, q_ratio))
    return r

#used to implement your own NB-SVM, if you don't want to use the sklearn built-in
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

#returns SVM accuracy as a function of the C parameter
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


"""General functions"""        
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



