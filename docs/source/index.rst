modAL: A modular active learning framework for Python3
======================================================

|   *modal: adjective, relating to structure as opposed to substance*
|   (Merriam-Webster Dictionary)

modAL is an active learning framework for Python3, designed with *modularity, flexibility* and *extensibility* in mind. Built on top of scikit-learn, it allows you to rapidly create active learning workflows with nearly complete freedom. What is more, you can easily replace parts with your custom built solutions, allowing you to design novel algorithms with ease.

Active learning from bird's-eye view
------------------------------------

With the recent explosion of available data, you have can have millions of unlabelled examples with a high cost to obtain labels. For instance, when trying to predict the sentiment of tweets, obtaining a training set
can require immense manual labour. But worry not, active learning comes to the rescue! In general, AL is a framework allowing you to increase classification performance by intelligently querying you to label the most informative instances. To give an example, suppose that you have the following data and classifier with shaded regions signifying the classification probability.

.. image:: content/overview/img/motivating-example.png
   :align: center

Suppose that you can query the label of an unlabelled instance, but it costs you a lot. Which one would you choose? By querying an instance in the uncertain region, surely you obtain more information than querying by random. Active learning gives you a set of tools to handle problems like this. In general, an active learning workflow looks like the following.

.. image:: content/overview/img/active-learning.png
   :align: center

The key components of any workflow are the **model** you choose, the **uncertainty** measure you use and the **query** strategy you apply to request labels. With modAL, instead of choosing from a small set of built-in components, you have the freedom to seamlessly integrate scikit-learn or Keras models into your algorithm and easily tailor your custom query strategies and uncertainty measures.

modAL in action
---------------

Let's see what modAL can do for you!

From zero to one in a few lines of code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Active learning with a scikit-learn classifier, for instance RandomForestClassifier, can be as simple as the following.

.. code:: python

    from modAL.models import ActiveLearner
    from sklearn.ensemble import RandomForestClassifier

    # initializing the learner
    learner = ActiveLearner(
        estimator=RandomForestClassifier(),
        X_training=X_training, y_training=y_training
    )

    # query for labels
    query_idx, query_inst = learner.query(X_pool)

    # ...obtaining new labels from the Oracle...

    # supply label for queried instance
    learner.teach(X_pool[query_idx], y_new)

Replacing parts quickly
^^^^^^^^^^^^^^^^^^^^^^^

If you would like to use different uncertainty measures and query
strategies than the default uncertainty sampling, you can either replace
them with several built-in strategies or you can design your own by
following a few very simple design principles. For instance, replacing
the default uncertainty measure to classification entropy looks the
following.

.. code:: python

    from modAL.models import ActiveLearner
    from modAL.uncertainty import entropy_sampling
    from sklearn.ensemble import RandomForestClassifier

    learner = ActiveLearner(
        estimator=RandomForestClassifier(),
        query_strategy=entropy_sampling,
        X_training=X_training, y_training=y_training
    )

Replacing parts with your own solutions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

modAL was designed to make it easy for you to implement your own query
strategy. For example, implementing and using a simple random sampling
strategy is as easy as the following.

.. code:: python

    import numpy as np

    def random_sampling(classifier, X_pool):
        n_samples = len(X_pool)
        query_idx = np.random.choice(range(n_samples))
        return query_idx, X_pool[query_idx]

    learner = ActiveLearner(
        estimator=RandomForestClassifier(),
        query_strategy=random_sampling,
        X_training=X_training, y_training=y_training
    )

For more details on how to implement your custom strategies, visit the
page :ref:`Extending-modAL`!


An example with active regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To see modAL in *real* action, let's consider an active regression
problem with Gaussian processes! In this example, we shall try to learn
the *noisy sine* function:

.. code:: python

    import numpy as np

    X = np.random.choice(np.linspace(0, 20, 10000), size=200, replace=False).reshape(-1, 1)
    y = np.sin(X) + np.random.normal(scale=0.3, size=X.shape)

For active learning, we shall define a custom query strategy tailored to
Gaussian processes. In a nutshell, a *query stategy* in modAL is a
function taking (at least) two arguments (an estimator object and a pool
of examples), outputting the index of the queried instance and the
instance itself. In our case, the arguments are ``regressor`` and ``X``.

.. code:: python

    def GP_regression_std(regressor, X):
        _, std = regressor.predict(X, return_std=True)
        query_idx = np.argmax(std)
        return query_idx, X[query_idx]

After setting up the query strategy and the data, the active learner can
be initialized.

.. code:: python

    from modAL.models import ActiveLearner
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import WhiteKernel, RBF

    n_initial = 5
    initial_idx = np.random.choice(range(len(X)), size=n_initial, replace=False)
    X_training, y_training = X[initial_idx], y[initial_idx]

    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
             + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))

    regressor = ActiveLearner(
        estimator=GaussianProcessRegressor(kernel=kernel),
        query_strategy=GP_regression_std,
        X_training=X_training.reshape(-1, 1), y_training=y_training.reshape(-1, 1)
    )

The initial regressor is not very accurate.

.. image:: content/overview/img/gp-initial.png
   :align: center

The blue band enveloping the regressor represents the standard deviation
of the Gaussian process at the given point. Now we are ready to do
active learning!

.. code:: python

    # active learning
    n_queries = 10
    for idx in range(n_queries):
        query_idx, query_instance = regressor.query(X)
        regressor.teach(X[query_idx].reshape(1, -1), y[query_idx].reshape(1, -1))

After a few queries, we can see that the prediction is much improved.

.. image:: content/overview/img/gp-final.png
   :align: center

Citing
------

If you use modAL in your projects, you can cite it as

::

    @article{modAL2018,
        title={mod{AL}: {A} modular active learning framework for {P}ython},
        author={Tivadar Danka and Peter Horvath},
        url={https://github.com/cosmic-cortex/modAL},
        note={available on arXiv at \url{https://arxiv.org/abs/1805.00979}}
    }

About the developer
-------------------

modAL is developed by me, `Tivadar
Danka <https://www.tivadardanka.com>`__ (aka
`cosmic-cortex <https://github.com/cosmic-cortex>`__ in GitHub). I have a PhD in pure mathematics, but I fell in love with biology and machine learning right after I finished my PhD. I have changed fields and now I work in the `Bioimage Analysis and Machine Learning Group of Peter Horvath <http://group.szbk.u-szeged.hu/sysbiol/horvath-peter-lab-index.html>`__, where I am working to develop active learning strategies for intelligent sample analysis in biology. During my work I realized that in Python, creating and prototyping active learning workflows can be made really easy and fast with scikit-learn, so I ended up developing a general framework for this. The result is modAL :) If you have any questions, requests or suggestions, you can contact me at 85a5187a@opayq.com! I hope you'll find modAL useful!

.. toctree::
   :maxdepth: 1
   :caption: Overview
   
   modAL in a nutshell <content/overview/modAL-in-a-nutshell>
   Installation <content/overview/Installation>
   Extending modAL <content/overview/Extending-modAL>

.. toctree::
   :maxdepth: 1
   :caption: Models
   
   ActiveLearner <content/models/ActiveLearner>
   BayesianOptimizer <content/models/BayesianOptimizer>
   Committee <content/models/Committee>
   CommitteeRegressor <content/models/CommitteeRegressor>

.. toctree::
   :maxdepth: 1
   :caption: Query strategies
   
   Acquisition functions <content/query_strategies/Acquisition-functions>
   Uncertainty sampling <content/query_strategies/Uncertainty-sampling>
   Disagreement sampling <content/query_strategies/Disagreement-sampling>
   content/query_strategies/Ranked-batch-queries

.. toctree::
   :maxdepth: 1
   :caption: Examples
   
   content/examples/pool-based_sampling
   content/examples/ranked_batch_mode
   content/examples/Stream-based-sampling
   Active regression <content/examples/active_regression>
   Ensemble regression <content/examples/ensemble_regression>
   content/examples/bayesian_optimization
   content/examples/Query-by-committee
   content/examples/Bootstrapping-and-bagging
   content/examples/extending_modal
   Keras integration <content/examples/Keras-integration>
   
.. toctree::
   :glob:
   :maxdepth: 1
   :caption: API reference

   content/apireference/*
