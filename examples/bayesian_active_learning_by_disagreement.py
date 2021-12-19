"""
This is a original implementation of the algorithm Bayesian Active Learning by Disagreements.
(Pl. refer - https://arxiv.org/abs/1112.5745). It calculates the disagreement between an ensemble
of classifiers and a single classifier using monte carlo estimates.
"""
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from keras.wrappers.scikit_learn import KerasClassifier
from modAL.models import ActiveLearner

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train/255.0
X_test_norm = X_test/255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

def LeNet():
    model = Sequential()

    model.add(Conv2D(filters = 6, kernel_size = (5,5), padding = 'same', 
                   activation = 'relu', input_shape = (28,28,1)))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters = 16, kernel_size = (5,5), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add(Dense(120, activation = 'relu'))
    model.add(Dropout(0.55)) 
    model.add(Dense(10, activation = 'softmax'))
    opt = Adam(learning_rate = 0.001)
    model.compile(loss = categorical_crossentropy, 
                optimizer = opt, 
                metrics = ['accuracy']) 

    return model

def max_disagreement(model, X, n=32, n_mcd=10):

    partial_model = Model(model.estimator.model.inputs, model.estimator.model.layers[-1].output)
    prob = np.stack([partial_model(X.reshape(-1, 28, 28, 1), training=True) for _ in range(n_mcd)])
    pb = np.mean(prob, axis=0)
    entropy1 = (-pb*np.log(pb)).sum(axis=1)
    entropy2 = (-prob*np.log(prob)).sum(axis=2).mean(axis=0)
    un = entropy2-entropy1
    return np.argpartition(un, n)[:n]

model = KerasClassifier(LeNet)

U_x = np.copy(X_train)
U_y = np.copy(y_train)

INITIAL_SET_SIZE = 32
ind = np.random.choice(range(len(U_x)), size=INITIAL_SET_SIZE)

X_initial = U_x[ind]
y_initial = U_y[ind]

U_x = np.delete(U_x, ind, axis=0)
U_y = np.delete(U_y, ind, axis=0)

active_learner = ActiveLearner(
    estimator=model,
    X_training=X_initial,
    y_training=y_initial,
    query_strategy=max_disagreement,
    verbose=0
)

N_QUERIES = 20

scores = [active_learner.score(X_test, y_test, verbose=0)]

for index in range(N_QUERIES):

    query_idx, query_instance = active_learner.query(U_x)

    L_x = U_x[query_idx]
    L_y = U_y[query_idx]

    active_learner.teach(L_x, L_y, epochs=50, batch_size=128, verbose=0)

    U_x = np.delete(U_x, query_idx, axis=0)
    U_y = np.delete(U_y, query_idx, axis=0)
    
    acc = active_learner.score(X_test, y_test)

    print(F'Query {index+1}: Test Accuracy: {acc}')
    
    scores.append(acc)