"""
Learning the shape of an object using uncertainty based sampling

In this example, we will demonstrate the use of ActiveLearner with
the scikit-learn implementation of the Random Forest Classifier algorithm
"""


from scipy.misc import imread
from ModAL.active_learning.models import ActiveLearner
from sklearn.ensemble import RandomForestClassifier

# import the image
data = imread('square.png')
# reshape the data
data = data.reshape(-1,)