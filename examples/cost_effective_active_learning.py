"""
This is a modified implementation of the algorithm Cost Effective Active Learning
(Pl. refer - https://arxiv.org/abs/1701.03551). This version not only picks up the 
top K uncertain samples but also picks up the top N highly confident samples that
may represent information and diversity. It is better than the original implementation
as it does not involve tuning the confidence threshold parameter for every dataset.
"""

from keras.datasets import mnist
import numpy as np
from modAL.models import ActiveLearner
from sklearn.ensemble import RandomForestClassifier
from scipy.special import entr


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255
X_test = X_test / 255
y_train = y_train.astype(np.uint8)
y_test = y_test.astype(np.uint8)

X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

model = RandomForestClassifier(n_estimators=100)

INITIAL_SET_SIZE = 32

U_x = np.copy(X_train)
U_y = np.copy(y_train)

ind = np.random.choice(range(len(U_x)), size=INITIAL_SET_SIZE)

X_initial = U_x[ind]
y_initial = U_y[ind]

U_x = np.delete(U_x, ind, axis=0)
U_y = np.delete(U_y, ind, axis=0)


def max_entropy(active_learner, X, K=16, N=16):

    class_prob = active_learner.predict_proba(X)
    entropy = entr(class_prob).sum(axis=1)
    uncertain_idx = np.argpartition(entropy, -K)[-K:]
    confidence_idx = np.argpartition(entropy, N)[:N]

    return np.concatenate((uncertain_idx, confidence_idx), axis=0)


active_learner = ActiveLearner(
    estimator=model,
    X_training=X_initial,
    y_training=y_initial,
    query_strategy=max_entropy
)

N_QUERIES = 20

K_MAX_ENTROPY = 16
N_MIN_ENTROPY = 16

scores = [active_learner.score(X_test, y_test)]

for index in range(N_QUERIES):

    query_idx, query_instance = active_learner.query(U_x, K_MAX_ENTROPY, N_MIN_ENTROPY)

    active_learner.teach(U_x[query_idx], U_y[query_idx])

    U_x = np.delete(U_x, query_idx, axis=0)
    U_y = np.delete(U_y, query_idx, axis=0)
    
    acc = active_learner.score(X_test, y_test)

    print(F'Query {index+1}: Test Accuracy: {acc}')
    
    scores.append(acc)