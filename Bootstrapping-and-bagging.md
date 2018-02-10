# Bootstrapping and bagging
Bootstrapping and bagging can be very useful when using ensemble models such as the Committee. In essence, bootstrapping is random sampling with replacement from the available training data. Bagging (= bootstrap aggregation) is performing it many times and training an estimator for each bootstrapped dataset. It is available in modAL for both the base ActiveLearner model and the Committee model as well. In this short tutorial, we are going to see how to perform bootstrapping and bagging in your active learning workflow.

The executable script for this example can be [found here!](https://github.com/cosmic-cortex/modAL/blob/master/examples/bagging.py)

## The dataset
In this short example, we will try to learn the shape of three black disks on a white background.
```python
import numpy as np
from itertools import product

# creating the dataset
im_width = 500
im_height = 500
data = np.zeros((im_height, im_width))
# each disk is coded as a triple (x, y, r), where x and y are the centers and r is the radius
disks = [(150, 150, 80), (200, 380, 50), (360, 200, 100)]
for i, j in product(range(im_width), range(im_height)):
    for x, y, r in disks:
        if (x-i)**2 + (y-j)**2 < r**2:
            data[i, j] = 1

# create the pool from the image
X_pool = np.transpose(
    [np.tile(np.asarray(range(data.shape[0])), data.shape[1]),
     np.repeat(np.asarray(range(data.shape[1])), data.shape[0])]
)
# map the intensity values against the grid
y_pool = np.asarray([data[P[0], P[1]] for P in X_pool])
```
Here is how it looks:

![b-data](img/b-data.png)

First we shall train three ActiveLearners on a bootstrapped dataset. Then we are going to bundle them together in a Committee and see how bagging is done with modAL.
## Bootstrapping
```python
from modAL.models import ActiveLearner, Committee
from sklearn.neighbors import KNeighborsClassifier

# initial training data: 100 random pixels
initial_idx = np.random.choice(range(len(X_pool)), size=100)

# initializing the learners
n_learners = 3
learner_list = []
for _ in range(n_learners):
    learner = ActiveLearner(
        estimator=KNeighborsClassifier(n_neighbors=10),
        X_training=X_pool[initial_idx], y_training=y_pool[initial_idx],
        bootstrap_init=True
    )
    learner_list.append(learner)
```
As you can see, the main difference in this from the regular use of ActiveLearner is passing ```bootstrap_init=True``` upon initialization. This makes it to train the model on a bootstrapped dataset, although it stores the *complete* training dataset among its known examples. In this exact case, here is how the classifiers perform:

![b-committee_learners](img/b-committee_learners.png)

## Bagging
We can put our learners together in a Committee and see how they perform.
```python
# assembling the Committee
committee = Committee(learner_list)
```

![b-committee_prediction](img/b-committee_prediction.png)

If you would like to take each learner in the Committee and retrain them using bagging, you can use the ```.rebag()``` method:
```python
# rebagging the data
committee.rebag()
```
In this case, the classifiers perform in the following way after rebagging.

![b-rebag](img/b-rebag.png)
