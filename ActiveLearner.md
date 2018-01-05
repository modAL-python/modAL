# The ActiveLearner class
In modAL, the base active learning model is the ActiveLearner class. In this short tutorial, we are going to see how to use it and what are its basic functionalities.

## Page contents
- [Initialization](#initialization)  
- [Training](#training)  
- [Bootstrapping](#bootstrapping)  
- [Querying for labels](#querying)  
- [Query strategies](#query-strategies)  
- [Prediction and scoring](#prediction)  

## Initialization<a name="initialization"></a>
To create an ActiveLearner object, you need to provide two things: a *scikit-learn estimator object* and a *query strategy function* (The latter one is optional, the default strategy is maximum uncertainty sampling.). Regarding the query strategies, you can find built-ins in ```modAL.uncertainty```, but you can also implement your own. For instance, you can just simply do the following.
```python
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.ensemble import RandomForestClassifier

learner = ActiveLearner(
    predictor=RandomForestClassifier(),
    query_strategy=uncertainty_sampling
)
```
If you have initial training data available, you can train the predictor by passing it via the arguments ```X_initial``` and ```y_initial```. For instance, if the samples are contained in ```X_training``` and the labels are in ```y_training```, you can do the following.
```python
learner = ActiveLearner(
    predictor=RandomForestClassifier(),
    query_strategy=uncertainty_sampling
    X_initial=X_training, y_initial=y_training
)
```
After initialization, your ActiveLearner is ready to ask and learn! The learner keeps track of the training data it has seen during its lifetime.

## Training<a name="training"></a>
To teach newly acquired labels for the ActiveLearner, you should use the ```.teach(X, y)``` method. This augments the available training data with the new samples ```X``` and new labels ```y```, then refits the predictor to this augmented training dataset. Just like this:
```python
learner.teach(X, y)
```
If you would like to start from scratch, you can use the ```.fit(X, y)``` method to make the learner forget everything it has seen and fit the model to the newly provided data.

## Bootstrapping<a name="bootstrapping"></a>
Training is also available with bootstrapping by passing ```bootstrap=True``` for ```learner.teach()``` or ```learner.fit()```. In this case, a random set is sampled with replacement from the training data available (or the data provided in the case of ```.fit()```), which is used to train the estimator. Bootstrapping is mostly useful when building ensemble models with bagging, for instance in a *query by committee* setting.

## Querying for labels<a name="querying"></a>
Active learners are called *active* because if you provide them unlabelled samples, they can select you the best instances to label. In modAL, you can achieve this by calling the ```.query(X)``` method:
```python
query_idx, query_sample = learner.query(X)

# ...obtaining new labels from the Oracle...

learner.teach(query_sample, query_label)
```
Under the hood, the ```.query(X)``` method simply calls the *query strategy function* specified by you upon initialization of the ActiveLearner.

The available built-in query strategies are *max uncertainty sampling*, *max margin sampling* and *entropy sampling*. For more details, see the [Uncertainty sampling](Uncertainty-sampling) wiki page.

## Query strategies<a name="query-strategies"></a>
In modAL, currently there are three built-in query strategies: *max uncertainty*, *max margin* and *max entropy*, they are located in the ```modAL.uncertainty``` module. You can find an [informal tutorial here](Uncertainty-sampling) about them. For technical details, see the [API reference for uncertainty sampling](Uncertainty-sampling-API).

## Prediction and scoring<a name="prediction"></a>
To use the ActiveLearner for prediction and to calculate the mean accuracy score, you can just do what you would do with a *scikit-learn* classifier: call the ```.predict(X)``` and ```.score(X, y)``` methods. If you would like to use more sophisticated metrics for your prediction, feel free to use a function from ```sklearn.metrics```, they are compatible with modAL.