Bringing your Keras models to modAL workflows
=============================================

Thanks for the scikit-learn API of Keras, you can seamlessly integrate Keras models into your modAL workflow. In this tutorial, we shall quickly introduce how to use the scikit-learn API of Keras and we are going to see how to do active learning with it. More details on the Keras scikit-learn API `can be found here <https://keras.io/scikit-learn-api/>`__.

The executable script for this example can be `found here! <https://github.com/cosmic-cortex/modAL/blob/master/examples/keras_integration.py>`__

Keras' scikit-learn API
-----------------------

By default, a Keras model's interface differs from what is used for scikit-learn estimators, it is possible to adapt your model.

.. code:: python

    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.wrappers.scikit_learn import KerasClassifier


    # build function for the Keras' scikit-learn API
    def create_keras_model():
        """
        This function compiles and returns a Keras model.
        Should be passed to KerasClassifier in the Keras scikit-learn API.
        """
        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(784, )))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation='sigmoid'))
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adadelta',
            metrics=['accuracy']
        )

        return model


    # create the classifier
    classifier = KerasClassifier(create_keras_model)

For our purposes, the ``classifier`` object acts just like any scikit-learn estimator.

Active learning with Keras
--------------------------

In this example, we are going to use the famous MNIST dataset, which is available as a built-in for Keras.

.. code:: python

    from keras.datasets import mnist

    # read training data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784).astype('float32') / 255
    X_test = X_test.reshape(10000, 784).astype('float32') / 255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # assemble initial data
    n_initial = 1000
    initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)
    X_training = X_train[initial_idx]
    y_training = y_train[initial_idx]

    # generate the pool
    # remove the initial data from the training dataset
    X_pool = np.delete(X_train, initial_idx, axis=0)
    y_pool = np.delete(y_train, initial_idx, axis=0)

Active learning with data and classifier ready is as easy as always. Because training is *very* expensive in large neural networks, this time we are going to query the best 200 instances each time we measure the uncertainty of the pool.

.. code:: python

    from modAL.models import ActiveLearner

    # initialize ActiveLearner
    learner = ActiveLearner(
        estimator=classifier,
        X_training=X_training, y_training=y_training,
        verbose=0
    )

To make sure that you train only on newly queried labels, pass ``only_new=True`` to the ``.teach()`` method of the learner.

.. code:: python

    # the active learning loop
    n_queries = 10
    for idx in range(n_queries):
        query_idx, query_instance = learner.query(X_pool, n_instances=200, verbose=0)
        learner.teach(
            X=X_pool[query_idx], y=y_pool[query_idx],
            only_new=True
            verbose=0
        )
        # remove queried instance from pool
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx, axis=0)
