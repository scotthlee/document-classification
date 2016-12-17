# Random forests

This script will train a random forest on the corpus and then return its accuracy on the test data. 

## Arguments

The script has 3 positional arguments you'll need to enter for it to run:

  1. ```data```: The path for the CSV file holding the document text and the target variable.
  2. ```y_name```: The name of the column holding the target variable, like 'outcome' or 'sentiment'.
  3. ```x_name```: The name of the column holding the text, like 'docs' or 'evaluations'.

The script also has a number of optional arguments you can use to fine-tune the model's performance:

  1. 
