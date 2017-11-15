"""
Learning the shape of an object using uncertainty based sampling

In this example, we will demonstrate the use of ActiveLearner with
the scikit-learn implementation of the Random Forest Classifier algorithm
"""


from scipy.misc import imread
from sklearn.ensemble import RandomForestClassifier
from ModAL.active_learning.models import ActiveLearner
from ModAL.active_learning.utilities import uncertainty

# import the image
data = imread('square.png')
# reshape the data
data = data.reshape(-1,)

# create an ActiveLearner instance
learner = ActiveLearner(predictor=RandomForestClassifier(), utility_function=uncertainty)