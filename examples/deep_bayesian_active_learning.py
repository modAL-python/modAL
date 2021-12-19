import keras
import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.models import Sequential
from keras.regularizers import l2
from keras.wrappers.scikit_learn import KerasClassifier
from modAL.models import ActiveLearner


def create_keras_model():
    model = Sequential()
    model.add(Conv2D(32, (4, 4), activation='relu'))
    model.add(Conv2D(32, (4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model


# create the classifier
classifier = KerasClassifier(create_keras_model)

# read training data
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# assemble initial data
initial_idx = np.array([],dtype=np.int)
for i in range(10):
    idx = np.random.choice(np.where(y_train==i)[0], size=2, replace=False)
    initial_idx = np.concatenate((initial_idx, idx))

# Preprocessing
X_train = X_train.reshape(60000, 28, 28, 1).astype('float32') / 255.
X_test = X_test.reshape(10000, 28, 28, 1).astype('float32') / 255.
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

X_initial = X_train[initial_idx]
y_initial = y_train[initial_idx]

# remove the initial data from the pool of unlabelled examples
X_pool = np.delete(X_train, initial_idx, axis=0)
y_pool = np.delete(y_train, initial_idx, axis=0)

"""
Query Strategy
"""

def max_entropy(learner, X, n_instances=1, T=100):
    random_subset = np.random.choice(X.shape[0], 2000, replace=False)
    MC_output = K.function([learner.estimator.model.layers[0].input, K.learning_phase()],
                           [learner.estimator.model.layers[-1].output])
    learning_phase = True
    MC_samples = [MC_output([X[random_subset], learning_phase])[0] for _ in range(T)]
    MC_samples = np.array(MC_samples)  # [#samples x batch size x #classes]
    expected_p = np.mean(MC_samples, axis=0)
    acquisition = - np.sum(expected_p * np.log(expected_p + 1e-10), axis=-1)  # [batch size]
    idx = (-acquisition).argsort()[:n_instances]
    return random_subset[idx]

def uniform(learner, X, n_instances=1):
    return np.random.choice(range(len(X)), size=n_instances, replace=False)

"""
Training the ActiveLearner
"""

# initialize ActiveLearner
learner = ActiveLearner(
    estimator=classifier,
    X_training=X_initial,
    y_training=y_initial,
    query_strategy=max_entropy,
    verbose=0
)

# the active learning loop
n_queries = 100
perf_hist = [learner.score(X_test, y_test, verbose=0)]
for index in range(n_queries):
    query_idx, query_instance = learner.query(X_pool, n_instances=10)
    learner.teach(X_pool[query_idx], y_pool[query_idx], epochs=50, batch_size=128, verbose=0)
    # remove queried instance from pool
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)
    model_accuracy = learner.score(X_test, y_test, verbose=0)
    print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))
    perf_hist.append(model_accuracy)
