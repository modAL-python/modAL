modAL: A modular active learning framework for Python3
======================================================

.. image:: https://travis-ci.org/modAL-python/modAL.svg?branch=master
   :target: https://travis-ci.org/modAL-python/modAL
.. image:: https://codecov.io/gh/modAL-python/modAL/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/modAL-python/modAL
.. image:: https://readthedocs.org/projects/modal-python/badge/?version=latest
   :target: https://modal-python.readthedocs.io/en/latest/?badge=latest

Welcome to the documentation for modAL!

modAL is an active learning framework for Python3, designed with *modularity, flexibility* and *extensibility* in mind. Built on top of scikit-learn, it allows you to rapidly create active learning workflows with nearly complete freedom. What is more, you can easily replace parts with your custom built solutions, allowing you to design novel algorithms with ease.

Currently supported active learning strategies are

- **uncertainty-based sampling:** *least confident* (`Lewis and Catlett <https://www.sciencedirect.com/science/article/pii/B978155860335650026X?via%3Dihub>`_), *max margin* and *max entropy* 
- **committee-based algorithms:** *vote entropy*, *consensus entropy* and *max disagreement* (`Cohn et al. <http://www.cs.northwestern.edu/~pardo/courses/mmml/papers/active_learning/improving_generalization_with_active_learning_ML94.pdf>`_)
- **multilabel strategies:** *SVM binary minimum* (`Brinker <https://link.springer.com/chapter/10.1007%2F3-540-31314-1_24>`_), *max loss*, *mean max loss*, (`Li et al. <http://dx.doi.org/10.1109/ICIP.2004.1421535>`_) *MinConfidence*, *MeanConfidence*, *MinScore*, *MeanScore* (`Esuli and Sebastiani <http://dx.doi.org/10.1007/978-3-642-00958-7_12>`_)
- **Bayesian optimization:** *probability of improvement*, *expected improvement* and *upper confidence bound* (`Snoek et al. <https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf>`_)
- **batch active learning:** *ranked batch-mode sampling* (`Cardoso et al. <https://www.sciencedirect.com/science/article/pii/S0020025516313949>`_)
- **information density framework** (`McCallum and Nigam <http://www.kamalnigam.com/papers/emactive-icml98.pdf>`_)
- **stream-based sampling** (`Atlas et al. <https://papers.nips.cc/paper/261-training-connectionist-networks-with-queries-and-selective-sampling.pdf>`_)
- **active regression** with *max standard deviance* sampling for Gaussian processes or ensemble regressors


.. toctree::
   :maxdepth: 1
   :caption: Overview
   
   content/overview/modAL-in-a-nutshell
   content/overview/Installation
   content/overview/Extending-modAL
   content/overview/Contributing

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
   
   content/examples/interactive_labeling
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
   content/apireference/multilabel.rst
   content/apireference/acquisition.rst
   content/apireference/batch.rst
   content/apireference/density.rst
   content/apireference/utils.rst
