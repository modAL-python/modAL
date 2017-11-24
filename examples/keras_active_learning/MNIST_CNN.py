"""
This example demonstrates how to use the active learning interface with Keras.
The example uses the scikit-learn wrappers of Keras. For more info, see https://keras.io/scikit-learn-api/
"""

import keras
import numpy as np
from modAL.models import ActiveLearner
from modAL.utilities import classifier_uncertainty