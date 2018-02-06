from modAL.models import ActiveLearner
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.datasets import load_iris

X_train, y_train = load_iris().data, load_iris().target

learner = ActiveLearner(estimator=RandomForestClassifier())

print(learner.get_params())