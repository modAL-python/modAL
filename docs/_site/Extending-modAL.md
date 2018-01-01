# Extending modAL
modAL was designed for researchers, allowing quick and efficient prototyping. For this purpose, modAL makes it easy for you to use your customly designed parts, for instance query strategies or new classifier algorithms.

## Page contents
- [Writing your own query strategy](#query-strategy)  
- [Using your custom estimators](#custom-estimators)

## Writing your own query strategy<a name="query-strategy"></a>
In modAL, a query strategy for active learning is implemented as a function, taking an estimator with a bunch of data and turning it into a data point from the data you supplied to it. Exactly like in the following.
```python
def some_query_strategy(classifier, X, a_keyword_argument=42):
    proba = classifier.predict_proba(X)
    # ...
    # ... do some magic and find the most informative instance ...
    # ...
    return query_idx, X[query_idx]
```
Putting this to work is as simple as the following.
```python
from modAL.models import ActiveLearner
from sklearn.ensemble import RandomForestClassifier

# initializing the learner
learner = ActiveLearner(
    predictor=RandomForestClassifier(),
	query_strategy=some_query_strategy
)

# querying for labels
query_idx, query_instance = learner.query(X)
```
For a more elaborate example see for instance this [active regression](Active-regression).

## Using your custom estimators<a name="custom-estimators"></a>
As long as your classifier follows the scikit-learn API, you can use it in your modAL workflow. (Really, all it needs is a ```.fit(X, y)``` and ```.predict(X)``` method.) For instance, the ensemble model implemented in Committee can be given to an ActiveLearner.
```python
# initializing the learners
n_learners = 3
learner_list = []
for _ in range(n_learners):
    learner = ActiveLearner(
        predictor=RandomForestClassifier(),
        X_initial=X_initial, y_initial=y_pool[initial_idx],
        bootstrap_init=True
    )
    learner_list.append(learner)

# assembling the Committee
committee = Committee(learner_list)

# ensemble active learner from the Committee
ensemble_learner = ActiveLearner(
    predictor=committee
)
```
Now you are ready for active learning with an ensemble of classifiers! If you would like to keep bagging the data pass ```bootstrap=True``` to the ```.teach()``` method!
