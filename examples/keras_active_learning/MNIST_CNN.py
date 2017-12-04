"""
This example demonstrates how to use the active learning interface with Keras.
The example uses the scikit-learn wrappers of Keras. For more info, see https://keras.io/scikit-learn-api/
"""

import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from modAL.models import ActiveLearner
from modAL.utilities import classifier_uncertainty


def create_keras_model():
    """
    This function compiles and returns a Keras model.
    Should be passed for the KerasClassifier in the
    Keras scikit-learn API.
    :return: Keras model
    """
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784, )))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    return model


"""
Data munching
1. Reading data from Keras
2. Assembling initial training data for ActiveLearner
3. Generating the pool
"""

# read training data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784).astype('float32')/255
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# select the first example from each category
initial_idx = list()
for label in range(10):
    for elem_idx, elem in enumerate(x_train):
        if y_train[elem_idx][label] == 1.0:
            initial_idx.append(elem_idx)
            break

# assemble initial data
x_initial = x_train[initial_idx]
y_initial = y_train[initial_idx]

# generate the pool
# remove the initial data from the training dataset
x_train = np.delete(x_train, initial_idx, axis=0)
y_train = np.delete(y_train, initial_idx, axis=0)
# sample random elements from x_train
pool_size = 1000
pool_idx = np.random.choice(range(len(x_train)), pool_size)
x_pool = x_train[pool_idx]
y_pool = y_train[pool_idx]

"""
Training the ActiveLearner
"""

# create the classifier
classifier = KerasClassifier(create_keras_model)

# initialize ActiveLearner
learner = ActiveLearner(
    predictor=classifier, utility_function=classifier_uncertainty,
    training_data=x_initial, training_labels=y_initial
)
learner.score(x_test, y_test, verbose=0)
