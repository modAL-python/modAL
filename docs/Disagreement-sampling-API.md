# The modAL.disagreement module
Similarly to modAL.uncertainty, this module contains two type of functions: disagreement measures and *query by committee* strategies. They both take a Committee object and an array of samples as input, but while disagreement measures return a numpy.ndarray containing the corresponding disagreements, the strategies return the samples to be labelled by the Oracle.

## Page contents

[Query by Committee strategies](#query-by-commmittee)  
* [Vote entropy sampling](#vote-entropy-sampling)  
* [Consensus entropy sampling](#consensus-entropy-sampling)  
* [Maximum disagreement sampling](#maximum-disagreement-sampling)  

[Disagreement measures](#disagreement-measures)  
* [Vote entropy](#vote-entropy)  
* [Consensus entropy](#consensus-entropy)  
* [Maximum disagreement](#maximum-disagreement)  

# Query by Committee strategies<a name="query-by-committee"></a>

## Vote entropy sampling<a name="vote-entropy-sampling"></a>

```vote_entropy_sampling(committee, X, n_instances=1, **disagreement_measure_kwargs)```

Vote entropy sampling strategy.

**Parameters**  
*committee*: Committee object  
    The committee for which the labels are to be queried.

*X*: numpy.ndarray of shape (n_samples, n_features)  
    The pool of samples to query from.

*n_instances*: int  
    Number of samples to be queried.

*disagreement_measure_kwargs*:  
    Keyword arguments to be passed for the disagreement measure function.

**Returns**  
*query_idx*: numpy.ndarray of shape (n_instances, )  
    The indices of the instances from X_pool chosen to be labelled.

*X_pool[query_idx]*: numpy.ndarray of shape (n_instances, n_features)  
    The instances from X_pool chosen to be labelled.

## Consensus entropy sampling<a name="consensus-entropy-sampling"></a>

```consensus_entropy_sampling(committee, X, n_instances=1, **disagreement_measure_kwargs)```

Consensus entropy sampling strategy.

**Parameters**  
*committee*: Committee object  
    The committee for which the labels are to be queried.

*X*: numpy.ndarray of shape (n_samples, n_features)  
    The pool of samples to query from.

*n_instances*: int  
    Number of samples to be queried.

*disagreement_measure_kwargs*: keyword arguments  
    Keyword arguments to be passed for the disagreement measure function.

**Returns**  
*query_idx*: numpy.ndarray of shape (n_instances, )  
    The indices of the instances from X_pool chosen to be labelled.

*X_pool[query_idx]*: numpy.ndarray of shape (n_instances, n_features)  
    The instances from X_pool chosen to be labelled.

## Maximum disagreement sampling<a name="maximum-disagreement-sampling"></a>

```max_disagreement_sampling(committee, X, n_instances=1, **disagreement_measure_kwargs)```

Maximum disagreement sampling strategy.

**Parameters**  
*committee*: Committee object  
    The committee for which the labels are to be queried.

*X*: numpy.ndarray of shape (n_samples, n_features)  
    The pool of samples to query from.

*n_instances*: int  
    Number of samples to be queried.

*disagreement_measure_kwargs*: keyword arguments  
    Keyword arguments to be passed for the disagreement measure function.

**Returns**  
*query_idx*: numpy.ndarray of shape (n_instances, )  
    The indices of the instances from X_pool chosen to be labelled.

*X_pool[query_idx]*: numpy.ndarray of shape (n_instances, n_features)  
    The instances from X_pool chosen to be labelled.

# Disagreement measures<a name="disagreement-measures"></a>

## Vote entropy<a name="vote-entropy"></a>

```vote_entropy(committee, X, **predict_proba_kwargs)```

Calculates the vote entropy for the Committee. First it computes the
predictions of X for each learner in the Committee, then calculates
the probability distribution of the votes. The entropy of this distribution
is the vote entropy of the Committee, which is returned.

**Parameters**  
*committee*: modAL.models.Committee object  
    The Committee instance for which the vote entropy is to be calculated.

*X*: numpy.ndarray of shape (n_samples, n_features)  
    The data for which the vote entropy is to be calculated.

*predict_proba_kwargs*: keyword arguments  
    Keyword arguments for the predict_proba method of the Committee.

**Returns**  
*entr*: numpy.ndarray of shape (n_samples, )  
    Vote entropy of the Committee for the samples in X.

**References**  
Settles, Burr: Active Learning, (Morgan & Claypool Publishers), equation no. (3.1)

## Consensus entropy<a name="consensus-entropy"></a>

```consensus_entropy(committee, X, **predict_proba_kwargs)```

Calculates the consensus entropy for the Committee. First it computes the class
probabilties of X for each learner in the Committee, then calculates the consensus
probability distribution by averaging the individual class probabilities for each
learner. The entropy of the consensus probability distribution is the vote entropy
of the Committee, which is returned.

**Parameters**  
*committee*: modAL.models.Committee object  
    The Committee instance for which the vote uncertainty entropy is to be calculated.

*X*: numpy.ndarray of shape (n_samples, n_features)  
    The data for which the vote uncertainty entropy is to be calculated.

*predict_proba_kwargs*: keyword arguments  
    Keyword arguments for the predict_proba method of the Committee.

**Returns**  
*entr*: numpy.ndarray of shape (n_samples, )  
    Vote uncertainty entropy of the Committee for the samples in X.

**References**  
Settles, Burr: Active Learning, (Morgan & Claypool Publishers), equation no. (3.2)

## Maximum disagreement<a name="maximum-disagreement"></a>

```KL_max_disagreement(committee, X, **predict_proba_kwargs)```

Calculates the max disagreement for the Committee. First it computes the class probabilties
of X for each learner in the Committee, then calculates the consensus probability
distribution by averaging the individual class probabilities for each learner. Then each
learner's class probabilities are compared to the consensus distribution in the sense of
Kullback-Leibler divergence. The max disagreement for a given sample is the argmax of the
KL divergences of the learners from the consensus probability.

**Parameters**  
*committee*: modAL.models.Committee object  
    The Committee instance for which the max disagreement is to be calculated.

*X*: numpy.ndarray of shape (n_samples, n_features)  
    The data for which the max disagreement is to be calculated.

*predict_proba_kwargs*: keyword arguments  
    Keyword arguments for the predict_proba method of the Committee.

**Returns**  
*entr*: numpy.ndarray of shape (n_samples, )  
    Max disagreement of the Committee for the samples in X.

**References**  
Settles, Burr: Active Learning, (Morgan & Claypool Publishers), equation no. (3.3)

