.. _Acquisition-functions:

Acquisition functions
=====================

In Bayesian optimization, a so-called *acquisition funciton* is used instead of the uncertainty based utility measures of active learning. In modAL, Bayesian optimization algorithms are implemented in the ``modAL.models.BayesianOptimizer`` class. Currently, there are three available acquisition funcions: probability of improvement, expected improvement and upper confidence bound.

Probability of improvement
--------------------------

The probability of improvement is defined by

.. math::
   
   PI(x) = \psi\Big( \frac{\mu(x) - f(x^+) - \xi}{\sigma(x)} \Big), 

where :math:`\mu(x)` and :math:`\sigma(x)` are the mean and variance of the regressor at :math:`x`, :math:`f`  is the function to be optimized with estimated maximum at :math:`x^+`, :math:`\xi` is a parameter controlling the degree of exploration and :math:`\psi(z)` denotes the cumulative distribution function of a standard Gaussian distribution.

.. image:: img/bo-PI.png
   :align: center

If you would like to use it with a BayesianOptimizer``, you should pass ``modAL.acquisition.max_PI`` as the query strategy upon initialization.

Expected improvement
--------------------

The expected improvement is defined by

.. math::  
   
   \begin{align*}
   EI(x) = & (\mu(x) - f(x^+) - \xi) \psi\Big( \frac{\mu(x) - f(x^+) - \xi}{\sigma(x)} \Big) \\
   & + \sigma(x) \phi\Big( \frac{\mu(x) - f(x^+) - \xi}{\sigma(x)} \Big),
   \end{align*} 

where :math:`\mu(x)` and :math:`\sigma(x)` are the mean and variance of the regressor at :math:`x`, :math:`f` is the function to be optimized with estimated maximum at :math:`x^+`, :math:`\xi` is a parameter controlling the degree of exploration and :math:`\psi(z)`, :math:`\phi(z)`  denotes the cumulative distribution function and density function of a standard Gaussian distribution. 

.. image:: img/bo-EI.png
   :align: center

If you would like to use it with a ``BayesianOptimizer``, you should pass ``modAL.acquisition.max_EI`` as the query strategy upon initialization.

Upper confidence bound
----------------------

The upper confidence bound is defined by

.. math::
   
   UCB(x) = \mu(x) + \beta \sigma(x), 

where :math:`\mu(x)` and :math:`\sigma(x)` are the mean and variance of the regressor and :math:`\beta` is a parameter controlling the degree of exploration.

.. image:: img/bo-UCB.png
   :align: center

If you would like to use it with a ``BayesianOptimizer``, you should pass ``modAL.acquisition.max_UCB`` as the query strategy upon initialization.
