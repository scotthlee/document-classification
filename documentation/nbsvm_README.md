# Multinomial Naive Bayes + SVM

This script runs a few models, starting with a pure MNB and leading up to the interpolated MNB-SVM Wang and Manning (2012) report as their best-performing across all tasks. Generally, MNB will work better on short snippets of text, and the SVM will work better on paragraphs and full-length documents. This script will automatically tune the interpolation parameter, though, and will return the interpolated model with the highest accuracy on the test data. 

## Arguments

Like the other scripts in this module, there are three positional arguments required to run the script:
 
  1. -d, --data: the file path for the CSV holding both the raw text and the target variable
  2. -y, --y_name: the name of the column holding the target varizble
  3. -x --x_name: the name of the column holding the the text
 
There are also a number of optional arguments that allow for model customization and tuning:

  1. 
