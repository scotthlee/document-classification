#Generic classes and functions for performing document classification

This script is basicaly the module's bargain bin--there's not really a theme to what's in it, other than that it doesn't really belong in any of the other scripts.

##Classes
The only class here is the ```TextData``` class, which is a (sort-of) flexible holder for...text data. It comes with the ```.process()``` method for processing a Pandas ```DataFrame``` holding your corpus and target classes, as well as a ```.split()``` method for dividing the vectorized data into training and test sets. 

##Functions
The other functions in the script, like ```accuracy()``` and ```linear_prediction``` mostly get called by other scripts during model training and evaluation. They can be useful for building your own classifiers, though, especially if they're somehow linear.

##Sample code
```>>> data = TextData()``` and ```>>> data.process(corpus, x_name, y_name)``` will vectorize the raw corpus and store the results as a Numpy array in the ```TextData``` object, which can be retrieved with calls to the ```.X``` and ```.y``` attributes. Once you divide the data into training and test sets using with ```.split()```, the sets can be retrieved with calls to the ```.X_group``` and ```.y_group``` attributes, respecitvely ("group" should be replaced with either "train" or "test").


