Query by committee
==================

*Query by committee* is another popular active learning strategy, which alleviates many disadvantages of uncertainty sampling. For instance, uncertainty sampling tends to be biased towards the actual learner and it may miss important examples which are not in the sight of the estimator. This is fixed by keeping several hypotheses at the same time, selecting queries where disagreement occurs between them. In this example, we shall see how this works in the simplest case, using the iris dataset.

The executable script for this example is `available here! <https://github.com/cosmic-cortex/modAL/blob/master/examples/query_by_committee.py>`__

The dataset
-----------

We are going to use the iris dataset for this example. For more information on the iris dataset, see `its wikipedia page <https://en.wikipedia.org/wiki/Iris_flower_data_set>`__. For its scikit-learn interface, see `the scikit-learn documentation <http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html>`__.

.. code:: python

    import numpy as np
    from copy import deepcopy
    from sklearn.datasets import load_iris

    # loading the iris dataset
    iris = load_iris()

    # generate the pool
    X_pool = deepcopy(iris['data'])
    y_pool = deepcopy(iris['target'])

If you perform a PCA on the iris dataset, here is how it looks like: 

.. figure:: img/qbc-iris_pca.png
   :align: center

Initializing the Committee
--------------------------

In this example, we are going to use the ``Committee`` class from ``modAL.models``. Its interface is almost exactly identical to the
``ActiveLearner``. Upon initialization, ``Committee`` requires a list of active learners.

.. code:: python

    from modAL.models import ActiveLearner, Committee
    from sklearn.ensemble import RandomForestClassifier

    # initializing Committee members
    n_members = 2
    learner_list = list()

    for member_idx in range(n_members):
        # initial training data
        n_initial = 5
        train_idx = np.random.choice(range(X_pool.shape[0]), size=n_initial, replace=False)
        X_training = X_pool[train_idx]
        y_training = y_pool[train_idx]

        # creating a reduced copy of the data with the known instances removed
        X_pool = np.delete(X_pool, train_idx, axis=0)
        y_pool = np.delete(y_pool, train_idx)

        # initializing learner
        learner = ActiveLearner(
            estimator=RandomForestClassifier(),
            X_training=X_training, y_training=y_training
        )
        learner_list.append(learner)

    # assembling the committee
    committee = Committee(learner_list=learner_list)

As you can see, the various hypotheses (which are taking the form of ActiveLearners) can be quite different.

.. figure:: img/qbc-initial_learners.png
   :align: center

Prediction is done by averaging the class probabilities for each learner and chosing the most likely class.

.. figure:: img/qbc-initial_committee.png
   :align: center

Active learning
---------------

The active learning loop is the same as for the ``ActiveLearner``.

.. code:: python

    # query by committee
    n_queries = 10
    for idx in range(n_queries):
        query_idx, query_instance = committee.query(X_pool)
        committee.teach(
            X=X_pool[query_idx].reshape(1, -1),
            y=y_pool[query_idx].reshape(1, )
        )
        # remove queried instance from pool
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx)

After a few queries, the hypotheses straighten out their disagreements and they reach consensus. Prediction accuracy is greatly improved in this case.

.. figure:: img/qbc-final_learners.png
   :align: center

.. figure:: img/qbc-final_committee.png
   :align: center
