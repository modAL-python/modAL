"""
Learning the shape of an object using uncertainty based sampling

In this example, we will demonstrate the use of ActiveLearner with
the scikit-learn implementation of the Random Forest Classifier algorithm
"""

import numpy as np
from scipy.misc import imread
from sklearn.ensemble import RandomForestClassifier
from ModAL.active_learning.models import ActiveLearner
from ModAL.active_learning.utilities import uncertainty

# import the image
data = imread('square.png')
# create the data from the image
X = np.transpose(
	[np.tile(np.asarray(range(data.shape[0])), data.shape[1]),
	 np.repeat(np.asarray(range(data.shape[1])), data.shape[0])]
)

# create an ActiveLearner instance
learner = ActiveLearner(
	predictor=RandomForestClassifier(), utility_function=uncertainty
)