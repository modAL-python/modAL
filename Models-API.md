# The modAL.models module
Currently, modAL supports two popular active learning models: *uncertainty based sampling* and *query by committee*. The classical uncertainty based models with one classifier are implemented in the ActiveLearner class, while with the Committee class you can use several scikit-learn estimators in an ensemble model.

## Page contents

- [ActiveLearner](#ActiveLearner)
  - [ActiveLearner.fit(X, y, bootstrap=False)](#ActiveLearner.fit)
  - [ActiveLearner.predict(X)](#ActiveLearner.predict)
  - [ActiveLearner.predict_proba(X)](#ActiveLearner.predict_proba)
  - [ActiveLearner.query(X)](#ActiveLearner.query)
  - [ActiveLearner.score(X, y)](#ActiveLearner.score)
  - [ActiveLearner.teach(X, y, bootstrap=False)](#ActiveLearner.teach)
- [Committee](#Committee)
  - [Committee.fit(X, y)](#Committee.fit)
  - [Committee.predict(X)](#Committee.predict)
  - [Committee.predict_proba(X)](#Committee.predict_proba)
  - [Committee.query(X)](#Committee.query)
  - [Committee.rebag()](#Committee.rebag)
  - [Committee.teach(X, y, bootstrap=False)](#Committee.teach)
  - [Committee.vote(X)](#Committee.vote)
  - [Committee.vote_proba(X)](#Committee.vote_proba)
- [CommitteeRegressor](#CommitteeRegressor)
  - [CommitteeRegressor.fit(X, y)](#CommitteeRegressor.fit)
  - [CommitteeRegressor.predict(X)](#CommitteeRegressor.predict)
  - [CommitteeRegressor.query(X)](#CommitteeRegressor.query)
  - [CommitteeRegressor.rebag()](#CommitteeRegressor.rebag)
  - [CommitteeRegressor.teach(X, y, bootstrap=False)](#CommitteeRegressor.teach)
  - [CommitteeRegressor.vote(X)](#CommitteeRegressor.vote)

# ActiveLearner<a name="ActiveLearner"></a>
This class is an abstract model of a general active learning algorithm.

**Parameters**  
*estimator*: scikit-learn estimator  
    The estimator to be used in the active learning loop.

*query_strategy*: function  
    Function providing the query strategy for the active learning
    loop, for instance modAL.uncertainty.uncertainty_sampling.

*X_training*: None or numpy.ndarray of shape (n_samples, n_features)  
    Initial training samples, if available.

*y_training*: None or numpy.ndarray of shape (n_samples, )  
    Initial training labels corresponding to initial training samples

*bootstrap_init*: boolean  
    If initial training data is available, bootstrapping can be done
    during the first training. Useful when building Committee models
    with bagging.

*fit_kwargs*: keyword arguments for the fit method

**Attributes**  
*estimator*: scikit-learn estimator  
    The estimator to be used in the active learning loop.

*query_strategy*: function  
    Function providing the query strategy for the active learning
    loop, for instance modAL.query.max_uncertainty.

*_X_training*: None numpy.ndarray of shape (n_samples, n_features)  
    If the model hasn't been fitted yet: None  
    If the model has been fitted already: numpy.ndarray containing the
    samples which the model has been trained on

*_y_training*: None or numpy.ndarray of shape (n_samples, )  
    If the model hasn't been fitted yet: None  
    If the model has been fitted already: numpy.ndarray containing the
    labels corresponding to _X_initial

**Examples**  
```python
>>> import numpy as np
>>> from sklearn.datasets import load_iris
>>> from sklearn.ensemble import RandomForestClassifier
>>> from modAL.models import ActiveLearner
>>>
>>> iris = load_iris()
>>> # give initial training examples
>>> X_training = iris['data'][[0, 50, 100]]
>>> y_training = iris['target'][[0, 50, 100]]
>>>
>>> # initialize active learner
>>> learner = ActiveLearner(
...     estimator=RandomForestClassifier(),
...     X_training=X_training, y_training=y_training
... )
>>>
>>> # querying for labels
>>> query_idx, query_sample = learner.query(iris['data'])
>>>
>>> # ...obtaining new labels from the Oracle...
>>>
>>> # teaching newly labelled examples
>>> learner.teach(
...     X=iris['data'][query_idx].reshape(1, -1),
...     y=iris['target'][query_idx].reshape(1, )
... )
```

## ActiveLearner.fit(X, y, bootstrap=False)<a name="ActiveLearner.fit"></a>
Interface for the fit method of the estimator. Fits the estimator
to the supplied data, then stores it internally for the active
learning loop.

**Parameters**  
*X*: numpy.ndarray of shape (n_samples, n_features)  
    The samples to be fitted.

*y*: numpy.ndarray of shape (n_samples, )  
    The corresponding labels.

*bootstrap*: boolean  
    If true, trains the estimator on a set bootstrapped from X. Useful for building
    Committee models with bagging.

*fit_kwargs*: keyword arguments  
    Keyword arguments to be passed to the fit method of the estimator.

**DANGER ZONE**  
When using scikit-learn estimators, calling this method will make the
ActiveLearner forget all training data it has seen!

## ActiveLearner.predict(X)<a name="ActiveLearner.predict"></a>
Estimator predictions for X. Interface with the predict
method of the estimator.

**Parameters**  
*X*: numpy.ndarray of shape (n_samples, n_features)  
    The samples to be predicted.

*predict_kwargs*: keyword arguments  
    Keyword arguments to be passed to the predict method of the classifier.

**Returns**  
*pred*: numpy.ndarray of shape (n_samples, )  
    Estimator predictions for X.

## ActiveLearner.predict_proba(X)<a name="ActiveLearner.predict_proba"></a>
Class probabilities if the estimator is a classifier.
Interface with the predict_proba method of the classifier.

**Parameters**  
*X*: numpy.ndarray of shape (n_samples, n_features)  
    The samples for which the class probabilities are to be predicted.

*predict_proba_kwargs*: keyword arguments  
    Keyword arguments to be passed to the predict_proba method of the
    classifier.

**Returns**  
*proba*: numpy.ndarray of shape (n_samples, n_classes)  
    Class probabilities for X.

## ActiveLearner.query(X)<a name="ActiveLearner.query"></a>
Finds the n_instances most informative point in the data provided by calling
the query_strategy function. Returns the queried instances and its indices.

**Parameters**  
*X*: numpy.ndarray of shape (n_samples, n_features)  
    The pool of samples from which the query strategy should choose
    instances to request labels.

*query_kwargs*: keyword arguments  
    Keyword arguments for the query strategy function.

**Returns**  
*query_idx*: numpy.ndarray of shape (n_instances, )  
    The indices of the instances from X_pool chosen to be labelled.

*X[query_idx]*: numpy.ndarray of shape (n_instances, n_features)  
    The instances from X_pool chosen to be labelled.

## ActiveLearner.score(X, y)<a name="ActiveLearner.score"></a>
Interface for the score method of the estimator.

**Parameters**  
*X*: numpy.ndarray of shape (n_samples, n_features)  
    The samples for which prediction accuracy is to be calculated

*y*: numpy.ndarray of shape (n_samples, )  
    Ground truth labels for X

*score_kwargs*: keyword arguments  
    Keyword arguments to be passed to the .score() method of the
    classifier

**Returns**  
*mean_accuracy*: numpy.float containing the mean accuracy of the estimator

## ActiveLearner.teach(X, y, bootstrap=False)<a name="ActiveLearner.teach"></a>
Adds X and y to the known training data and retrains the estimator
with the augmented dataset.

**Parameters**  
*X*: numpy.ndarray of shape (n_samples, n_features)  
    The new samples for which the labels are supplied by the expert.

*y*: numpy.ndarray of shape (n_samples, )  
    Labels corresponding to the new instances in X.

*bootstrap*: boolean  
    If True, training is done on a bootstrapped dataset. Useful for building
    Committee models with bagging.

*fit_kwargs*: keyword arguments  
    Keyword arguments to be passed to the fit method
    of the estimator.

# Committee<a name="Committee"></a>
This class is an abstract model of a committee-based active learning algorithm.

**Parameters**  
*learner_list*: list  
    A list of ActiveLearners forming the Committee.

*query_strategy*: function  
    Query strategy function. Committee supports disagreement-based query strategies
    from modAL.disagreement, but uncertainty-based strategies from modAL.uncertainty
    are also supported.

**Attributes**  
*classes_*: numpy.ndarray of shape (n_classes, )  
    Class labels known by the Committee.

*n_classes_*: int  
    Number of classes known by the Committee

**Examples**
```python
>>> import numpy as np
>>> from sklearn.datasets import load_iris
>>> from sklearn.neighbors import KNeighborsClassifier
>>> from sklearn.ensemble import RandomForestClassifier
>>> from modAL.models import ActiveLearner, Committee
>>>
>>> iris = load_iris()
>>>
>>> # initialize ActiveLearners
>>> learner_1 = ActiveLearner(
...     estimator=RandomForestClassifier(),
...     X_training=iris['data'][[0, 50, 100]], y_training=iris['target'][[0, 50, 100]]
... )
>>> learner_2 = ActiveLearner(
...     estimator=KNeighborsClassifier(n_neighbors=3),
...     X_training=iris['data'][[1, 51, 101]], y_training=iris['target'][[1, 51, 101]]
... )
>>>
>>> # initialize the Committee
>>> committee = Committee(
...     learner_list=[learner_1, learner_2]
... )
>>>
>>> # querying for labels
>>> query_idx, query_sample = committee.query(iris['data'])
>>>
>>> # ...obtaining new labels from the Oracle...
>>>
>>> # teaching newly labelled examples
>>> committee.teach(
...     X=iris['data'][query_idx].reshape(1, -1),
...     y=iris['target'][query_idx].reshape(1, )
... )
```

## Committee.fit(X, y)<a name="Committee.fit"></a>
Fits every learner to a subset sampled with replacement from X.
Calling this method makes the learner forget the data it has seen up until this point and
replaces it with X! If you would like to perform bootstrapping on each learner using the
data it has seen, use the method .rebag()!

**Parameters**  
*X*: numpy.ndarray of shape (n_samples, n_features)  
    The samples to be fitted.

*y*: numpy.ndarray of shape (n_samples, )  
    The corresponding labels.

*fit_kwargs*: keyword arguments  
    Keyword arguments to be passed to the fit method of the estimator.

**DANGER ZONE**  
Calling this method makes the learner forget the data it has seen up until this point and
replaces it with X!

## Committee.predict(X)<a name="Committee.predict"></a>
Predicts the class of the samples by picking the consensus prediction.

**Parameters**  
*X*: numpy.ndarray of shape (n_samples, n_features)  
    The samples to be predicted.

*predict_proba_kwargs*: keyword arguments  
    Keyword arguments to be passed to the predict_proba method of the
    Committee.

**Returns**  
*prediction*: numpy.ndarray of shape (n_samples, )  
    The predicted class labels for X.

## Committee.predict_proba(X)<a name="Committee.predict_proba"></a>
Consensus probabilities of the Committee.

**Parameters**  
*X*: numpy.ndarray of shape (n_samples, n_features)  
    The samples for which the class probabilities are
    to be predicted.

*predict_proba_kwargs*: keyword arguments  
    Keyword arguments to be passed to the predict_proba method of the
    Committee.

**Returns**  
*proba*: numpy.ndarray of shape (n_samples, n_classes)  
    Class probabilities for X.

## Committee.query(X)<a name="Committee.query"></a>
Finds the n_instances most informative point in the data provided by calling
the query_strategy function. Returns the queried instances and its indices.

**Parameters**  
*X*: numpy.ndarray of shape (n_samples, n_features)  
    The pool of samples from which the query strategy should choose
    instances to request labels.

*query_kwargs*: keyword arguments  
    Keyword arguments for the query strategy function.

**Returns**  
*query_idx*: numpy.ndarray of shape (n_instances, )  
    The indices of the instances from X_pool chosen to be labelled.

*X[query_idx]*: numpy.ndarray of shape (n_instances, n_features)  
    The instances from X_pool chosen to be labelled.

## Committee.rebag()<a name="Committee.rebag"></a>
Refits every learner with a dataset bootstrapped from its training instances. Contrary to
```.bag()```, it bootstraps the training data for each learner based on its own examples.

**Parameters**  
*fit_kwargs*: keyword arguments  
    Keyword arguments to be passed to the fit method of the estimator.

## Committee.teach(X, y, bootstrap=False)<a name="Committee.teach"></a>
Adds X and y to the known training data for each learner
and retrains the Committee with the augmented dataset.

**Parameters**  
*X*: numpy.ndarray of shape (n_samples, n_features)  
    The new samples for which the labels are supplied
    by the expert.

*y*: numpy.ndarray of shape (n_samples, )  
    Labels corresponding to the new instances in X.

*bootstrap*: boolean  
    If True, trains each learner on a bootstrapped set. Useful
    when building the ensemble by bagging.

*fit_kwargs*: keyword arguments  
    Keyword arguments to be passed to the fit method
    of the estimator.

## Committee.vote(X)<a name="Committee.vote"></a>
Predicts the labels for the supplied data for each learner in
the Committee.

**Parameters**  
*X*: numpy.ndarray of shape (n_samples, n_features)  
    The samples to cast votes.

*predict_kwargs*: keyword arguments  
    Keyword arguments to be passed for the learners .predict() method.

**Returns**  
*vote*: numpy.ndarray of shape (n_samples, n_learners)  
    The predicted class for each learner in the Committee and each sample in X.

## Committee.vote_proba(X)<a name="Committee.vote_proba"></a>
Predicts the probabilities of the classes for each sample and each learner.

**Parameters**  
*X*: numpy.ndarray of shape (n_samples, n_features)  
    The samples for which class probabilities are to be calculated.

*predict_proba_kwargs*: keyword arguments  
    Keyword arguments for the .predict_proba() method of the learners.

**Returns**  
*vote_proba*: numpy.ndarray of shape (n_samples, n_learners, n_classes)  
    Probabilities of each class for each learner and each instance.

# CommitteeRegressor<a name="CommitteeRegressor"></a>

This class is an abstract model of a committee-based active learning regression.

**Parameters**  
*learner_list*: list  
    A list of ActiveLearners forming the CommitteeRegressor.

*query_strategy*: function  
    Query strategy function.

**Examples**
```python
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from sklearn.gaussian_process import GaussianProcessRegressor
>>> from sklearn.gaussian_process.kernels import WhiteKernel, RBF
>>> from modAL.models import ActiveLearner, CommitteeRegressor
>>>
>>> # generating the data
>>> X = np.concatenate((np.random.rand(100)-1, np.random.rand(100)))
>>> y = np.abs(X) + np.random.normal(scale=0.2, size=X.shape)
>>>
>>> # initializing the regressors
>>> n_initial = 10
>>> kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
>>>
>>> initial_idx = list()
>>> initial_idx.append(np.random.choice(range(100), size=n_initial, replace=False))
>>> initial_idx.append(np.random.choice(range(100, 200), size=n_initial, replace=False))
>>> learner_list = [ActiveLearner(
...                         estimator=GaussianProcessRegressor(kernel),
...                         X_training=X[idx].reshape(-1, 1), y_training=y[idx].reshape(-1, 1)
...                 )
...                 for idx in initial_idx]
>>>
>>> # query strategy for regression
>>> def ensemble_regression_std(regressor, X):
...     _, std = regressor.predict(X, return_std=True)
...     query_idx = np.argmax(std)
...     return query_idx, X[query_idx]
>>>
>>> # initializing the CommitteeRegressor
>>> committee = CommitteeRegressor(
...     learner_list=learner_list,
...     query_strategy=ensemble_regression_std
... )
>>>
>>> # active regression
>>> n_queries = 10
>>> for idx in range(n_queries):
...     query_idx, query_instance = committee.query(X.reshape(-1, 1))
...     committee.teach(X[query_idx].reshape(-1, 1), y[query_idx].reshape(-1, 1))
```

## CommitteeRegressor.fit(X, y)<a name="CommitteeRegressor.fit"></a>
Fits every learner to a subset sampled with replacement from X.
Calling this method makes the learner forget the data it has seen up until this point and
replaces it with X! If you would like to perform bootstrapping on each learner using the
data it has seen, use the method .rebag()!

**Parameters**  
*X*: numpy.ndarray of shape (n_samples, n_features)  
    The samples to be fitted.

*y*: numpy.ndarray of shape (n_samples, )  
    The corresponding labels.

*fit_kwargs*: keyword arguments  
    Keyword arguments to be passed to the fit method of the estimator.

**DANGER ZONE**  
Calling this method makes the learner forget the data it has seen up until this point and
replaces it with X!

## CommitteeRegressor.predict(X)<a name="CommitteeRegressor.predict"></a>
Predicts the values of the samples by averaging the prediction of each regressor.

**Parameters**  
*X*: numpy.ndarray of shape (n_samples, n_features)  
    The samples to be predicted.

*predict_kwargs*: keyword arguments  
    Keyword arguments to be passed to the .vote() method of the CommitteeRegressor.

**Returns**  
*prediction*: numpy.ndarray of shape (n_samples, )  
    The predicted class labels for X.

## CommitteeRegressor.query(X)<a name="CommitteeRegressor.query"></a>
Finds the n_instances most informative point in the data provided by calling
the query_strategy function. Returns the queried instances and its indices.

**Parameters**  
*X*: numpy.ndarray of shape (n_samples, n_features)  
    The pool of samples from which the query strategy should choose
    instances to request labels.

*query_kwargs*: keyword arguments  
    Keyword arguments for the query strategy function.

**Returns**  
*query_idx*: numpy.ndarray of shape (n_instances, )  
    The indices of the instances from X_pool chosen to be labelled.

*X[query_idx]*: numpy.ndarray of shape (n_instances, n_features)  
    The instances from X_pool chosen to be labelled.

## CommitteeRegressor.rebag()<a name="CommitteeRegressor.rebag"></a>
Refits every learner with a dataset bootstrapped from its training instances. Contrary to
```.bag()```, it bootstraps the training data for each learner based on its own examples.

**Parameters**  
*fit_kwargs*: keyword arguments  
    Keyword arguments to be passed to the fit method of the estimator.

## CommitteeRegressor.teach(X, y, bootstrap=False)<a name="CommitteeRegressor.teach"></a>
Adds X and y to the known training data for each learner
and retrains the CommitteeRegressor with the augmented dataset.

**Parameters**  
*X*: numpy.ndarray of shape (n_samples, n_features)  
    The new samples for which the labels are supplied
    by the expert.

*y*: numpy.ndarray of shape (n_samples, )  
    Labels corresponding to the new instances in X.

*bootstrap*: boolean  
    If True, trains each learner on a bootstrapped set. Useful
    when building the ensemble by bagging.

*fit_kwargs*: keyword arguments  
    Keyword arguments to be passed to the fit method
    of the estimator.

## CommitteeRegressor.vote(X)<a name="CommitteeRegressor.vote"></a>
Predicts the values for the supplied data for each regressor in the CommitteeRegressor.

**Parameters**  
*X*: numpy.ndarray of shape (n_samples, n_features)  
    The samples to cast votes.

*predict_kwargs*: keyword arguments  
    Keyword arguments to be passed for the learners .predict() method.

**Returns**  
*vote*: numpy.ndarray of shape (n_samples, n_regressors)  
    The predicted value for each regressor in the CommitteeRegressor and each sample in X.
