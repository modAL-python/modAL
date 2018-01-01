# The modAL.uncertainty module
The module contains two type of functions: uncertainty measures and sampling strategies. Uncertainty measures take a classifier and an array of samples as input and they return an array of corresponding uncertainties. Sampling strategies take the same input but they return the samples to be labelled by the Oracle.

## Page contents
- [Query strategies](#query-strategies)
  - [Uncertainty sampling](#uncertainty-sampling)
  - [Margin sampling](#margin-sampling)
  - [Entropy sampling](#entropy-sampling)
- [Uncertainty measures](#uncertainty-measures)
  - [Classifier uncertainty](#classifier-uncertainty)
  - [Classifier margin](#classifier-margin)
  - [Classifier entropy](#classifier-entropy)

# Query strategies<a name="query-strategies"></a>

## Uncertainty sampling<a name="uncertainty-sampling"></a>

```uncertainty_sampling(classifier, X, n_instances=1, **uncertainty_measure_kwargs)```

Uncertainty sampling query strategy. Selects the least sure instances for labelling.

**Parameters**  
*classifier*: sklearn classifier object, for instance sklearn.ensemble.RandomForestClassifier  
    The classifier for which the labels are to be queried.

*X*: numpy.ndarray of shape (n_samples, n_features)  
    The pool of samples to query from.

*n_instances*: int  
    Number of samples to be queried.

*uncertainty_measure_kwargs*: keyword arguments  
    Keyword arguments to be passed for the uncertainty measure function.

**Returns**  
*query_idx*: numpy.ndarray of shape (n_instances, )  
    The indices of the instances from X_pool chosen to be labelled.

*X_pool[query_idx]*: numpy.ndarray of shape (n_instances, n_features)  
    The instances from X_pool chosen to be labelled.

## Margin sampling<a name="margin-sampling"></a>

```margin_sampling(classifier, X, n_instances=1, **uncertainty_measure_kwargs)```

Margin sampling query strategy. Selects the instances where the difference between
the first most likely and second most likely classes are the smallest.

**Parameters**  
*classifier*: sklearn classifier object, for instance sklearn.ensemble.RandomForestClassifier  
    The classifier for which the labels are to be queried.

*X*: numpy.ndarray of shape (n_samples, n_features)  
    The pool of samples to query from.

*n_instances*: int  
    Number of samples to be queried.

*uncertainty_measure_kwargs*: keyword arguments  
    Keyword arguments to be passed for the uncertainty measure function.

**Returns**  
*query_idx*: numpy.ndarray of shape (n_instances, )  
    The indices of the instances from X_pool chosen to be labelled.

*X_pool[query_idx]*: numpy.ndarray of shape (n_instances, n_features)  
    The instances from X_pool chosen to be labelled.

## Entropy sampling<a name="entropy-sampling"></a>

```entropy_sampling(classifier, X, n_instances=1, **uncertainty_measure_kwargs)```

Entropy sampling query strategy. Selects the instances where the class probabilities
have the largest entropy.

**Parameters**  
*classifier*: sklearn classifier object, for instance sklearn.ensemble.RandomForestClassifier  
    The classifier for which the labels are to be queried.

*X*: numpy.ndarray of shape (n_samples, n_features)  
    The pool of samples to query from.

*n_instances*: int  
    Number of samples to be queried.

*uncertainty_measure_kwargs*: keyword arguments  
    Keyword arguments to be passed for the uncertainty measure function.

**Returns**  
*query_idx*: numpy.ndarray of shape (n_instances, )  
    The indices of the instances from X_pool chosen to be labelled.

*X_pool[query_idx]*: numpy.ndarray of shape (n_instances, n_features)  
    The instances from X_pool chosen to be labelled.

***

# Uncertainty measures<a name="uncertainty-measures"></a>

## Classifier uncertainty<a name="classifier-uncertainty"></a>

```classifier_uncertainty(classifier, X, **predict_proba_kwargs)```

Classification uncertainty of the classifier for the provided samples.

**Parameters**  
*classifier*: sklearn classifier object, for instance sklearn.ensemble.RandomForestClassifier  
    The classifier for which the uncertainty is to be measured.

*X*: numpy.ndarray of shape (n_samples, n_features)  
    The samples for which the uncertainty of classification is to be measured.

*predict_proba_kwargs*: keyword arguments  
    Keyword arguments to be passed for the predict_proba method of the classifier.

**Returns**  
*uncertainty*: numpy.ndarray of shape (n_samples, )  
    Classifier uncertainty, which is 1 - P(prediction is correct).

**References**  
\[1\] Settles, Burr: Active Learning, (Morgan & Claypool Publishers), equation no. (2.1)

## Classifier margin<a name="classifier-margin"></a>

```classifier_margin(classifier, X, **predict_proba_kwargs)```

Classification margin uncertainty of the classifier for the provided samples.
This uncertainty measure takes the first and second most likely predictions
and takes the difference of their probabilities, which is the margin.

**Parameters**  
*classifier*: sklearn classifier object, for instance sklearn.ensemble.RandomForestClassifier  
    The classifier for which the uncertainty is to be measured

*X*: numpy.ndarray of shape (n_samples, n_features)  
    The samples for which the uncertainty of classification is to be measured

*predict_proba_kwargs*: keyword arguments  
    Keyword arguments to be passed for the predict_proba method of the classifier

**Returns**  
*margin*: numpy.ndarray of shape (n_samples, )  
    Margin uncertainty, which is the difference of the probabilities of first
    and second most likely predictions.

**References**  
\[1\] Settles, Burr: Active Learning, (Morgan & Claypool Publishers), equation no. (2.2)

## Classifier entropy<a name="classifier-entropy"></a>

```classifier_entropy(classifier, X, **predict_proba_kwargs)```  

Entropy of predictions of the for the provided samples.

**Parameters**  
*classifier*: sklearn classifier object, for instance sklearn.ensemble.RandomForestClassifier  
    The classifier for which the prediction entropy is to be measured.

*X*: numpy.ndarray of shape (n_samples, n_features)  
    The samples for which the prediction entropy is to be measured.

*predict_proba_kwargs*: keyword arguments  
    Keyword arguments to be passed for the predict_proba method of the classifier.

**Returns**  
*entr*: numpy.ndarray of shape (n_samples, )  
    Entropy of the class probabilities.

**References**  
\[1\] Settles, Burr: Active Learning, (Morgan & Claypool Publishers), equation no. (2.3)
