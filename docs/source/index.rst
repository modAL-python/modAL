modAL: A modular active learning framework for Python3
======================================================

Welcome to the documentation for modAL!

modAL is an active learning framework for Python3, designed with *modularity, flexibility* and *extensibility* in mind. Built on top of scikit-learn, it allows you to rapidly create active learning workflows with nearly complete freedom. What is more, you can easily replace parts with your custom built solutions, allowing you to design novel algorithms with ease.

.. toctree::
   :maxdepth: 1
   :caption: Overview
   
   content/overview/modAL-in-a-nutshell
   content/overview/Installation
   content/overview/Extending-modAL

.. toctree::
   :maxdepth: 1
   :caption: Models
   
   content/models/ActiveLearner
   content/models/BayesianOptimizer
   content/models/Committee
   content/models/CommitteeRegressor

.. toctree::
   :maxdepth: 1
   :caption: Query strategies
   
   content/query_strategies/Acquisition-functions
   content/query_strategies/uncertainty_sampling
   content/query_strategies/Disagreement-sampling
   content/query_strategies/ranked_batch_mode
   content/query_strategies/information_density

.. toctree::
   :maxdepth: 1
   :caption: Examples
   
   content/examples/pool-based_sampling
   content/examples/ranked_batch_mode
   content/examples/stream-based_sampling
   content/examples/active_regression
   content/examples/ensemble_regression
   content/examples/bayesian_optimization
   content/examples/query_by_committee
   content/examples/bootstrapping_and_bagging
   content/examples/Keras_integration
   
.. toctree::
   :glob:
   :maxdepth: 1
   :caption: API reference

   content/apireference/models.rst
   content/apireference/uncertainty.rst
   content/apireference/disagreement.rst
   content/apireference/acquisition.rst
   content/apireference/batch.rst
   content/apireference/density.rst
   content/apireference/utils.rst
