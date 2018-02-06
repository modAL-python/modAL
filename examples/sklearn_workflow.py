from modAL.models import ActiveLearner
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import cross_val_score

X_train, y_train = load_iris().data, load_iris().target

learner = ActiveLearner(estimator=RandomForestClassifier())
scores = cross_val_score(learner, X_train, y_train, cv=10)

check_estimator(learner)