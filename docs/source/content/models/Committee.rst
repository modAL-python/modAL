Committee
=========

One of the popular active learning strategies is the *Query by Committee*, where we keep several hypotheses (i.e. trained classifiers)
about the data, and we select our queries by measuring the disagreement of the hypotheses. In modAL, this model is implemented in the Committee class.

Initialization
--------------

To create a Committee object, you need to provide two things: a list of *ActiveLearner* objects and a *query strategy function*. (A list of *scikit-learn estimators* won't suffice, because each learner needs to keep track of the training examples it has seen.) For instance, you can do the following.

.. code:: python

    from modAL.models import Committee
    from modAL.disagreement import vote_entropy_sampling

    # a list of ActiveLearners:
    learners = [learner_1, learner_2]

    committee = Committee(
        learner_list=learners,
        query_strategy=vote_entropy_sampling
    )

You can find the disagreement-based query strategies in ``modAL.disagreement``, although a Committee works with uncertainty sampling strategies also. (Using a Committee with uncertainty sampling can be useful, for instance if you would like to build an active regression model and you need an ensemble of regressors to estimate uncertainty.)

Iterating through the learners of the Committee
-----------------------------------------------

Since Committee is iterable, feel free to use a ``for`` loop just as you would in any other case:

.. code:: python

    for learner in committee:
        # ...do something with the learner...

``len(committee)`` also works, it returns the number of learners in the
Committee.

Training and querying for labels
--------------------------------

Training and querying for labels work exactly the same as for the ActiveLearner. To teach new examples for the Committee, you should use the ``.teach(X, y)`` method, where ``X`` contains the new training examples and ``y`` contains the corresponding labels or values. This teaches the new training example for all learners in the Committee, hopefully improving performance.

To select the best instances to label, use the ``.query(X)`` method, just like for the ActiveLearner. It simply calls the query strategy function you specified, which measures the utility for each sample and selects the ones having the highest utility.

Bagging
-------

When building ensemble models such as in the Query by Committee setting, bagging can be useful and can improve performance. In Committee, this can be done with the methods ``.bag(X, y)`` and ``.rebag()``. The difference between them is that ``.bag(X, y)`` makes each learner forget the data it has seen until this point and replaces it with ``X`` and ``y``, while ``.rebag()`` refits each learner in the Committee by bootstrapping its training instances but leaving them as they were.

Bagging is also available during teaching new examples by passing ``bootstrap=True`` for the ``.teach()`` method. Just like this:

.. code:: python

    committee.teach(X_new, y_new, bootstrap=True)

First, this stores the new training examples and labels in each learner, then fits them using a bootstrapped subset of the known examples for the learner.

Query strategies
----------------

Currently, there are three built-in query by committee strategies in modAL: *max vote entropy*, *max uncertainty entropy* and *max
disagreement*. They are located in the ``modAL.disagreement`` module. You can find an informal tutorial at the page :ref:`Disagreement-sampling`.

Voting and predicting
---------------------

Although the API for predicting is the same for Committee than the ActiveLearner class, they are working slightly differently under the
hood.

| To obtain the predictions and class probabilities *for each learner*
  in the Committee, you should use the ``.vote(X)`` and
  ``.vote_proba(X)`` methods, where ``X`` contains the samples to be
  predicted.
| ``.vote(X)`` returns a numpy array of shape [n\_samples, n\_learners]
  containing the class predictions according to each learner, while
  ``.vote_proba(X)`` returns a numpy array of shape [n\_samples,
  n\_learners, n\_classes] containing the class probabilities for each
  learner. You don't need to worry about different learners seeing a
  different set of class labels, Committee is smart about that.

To get the predictions and class probabilities of the Committee itself, you shall use the ``.predict(X)`` and ``.predict_proba()`` methods, they are used like their ``scikit-learn`` relatives. ``.predict_proba()`` returns the class probability averaged across each learner (the so-called *consensus probabilities*), while ``.predict()`` selects the most likely label based on the consensus probabilities.
