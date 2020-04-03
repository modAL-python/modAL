import typing
import unittest
import numpy as np
from modAL.datamanager import DataManager


def first_true(ar :np.ndarray):
    return ar.nonzero()[0][0]


class TestAddLabels(unittest.TestCase):
    def test_test_that_when_the_first_add_is_at_0_it_updates_correctly(self):
        features = np.array([[x + y for x in range(10)] for y in range(10)])
        self.assertEqual(features.shape, (10, 10))
        manager = DataManager(features=features)
        manager.add_labels([(0, 1)])
        self.assertEqual(first_true(manager.labeled_mask), 0)
        # the index of the first unlabeled example is one past the first labeled
        self.assertEqual(first_true(manager.unlabeled_mask), 1)
    def test_addto_first_continuously(self):
        features = np.array([[x+y for x in range(10)] for y in range(10)])
        self.assertEqual(features.shape,(10,10))
        manager = DataManager(features=features)
        manager.add_labels([(0,1)])
        self.assertEqual(first_true(manager.labeled_mask),0)
        self.assertEqual(first_true(manager.unlabeled_mask), 1)

        manager.add_labels([(0,1)])
        self.assertEqual(first_true(manager.labeled_mask),0)
        self.assertEqual(first_true(manager.unlabeled_mask), 2)

        manager.add_labels([(0, 1)])
        self.assertEqual(first_true(manager.labeled_mask), 0)
        self.assertEqual(first_true(manager.unlabeled_mask), 3)

    def test_adding_in_the_middle(self):
        features = np.array([[x+y for x in range(10)] for y in range(10)])
        self.assertEqual(features.shape,(10,10))
        manager = DataManager(features=features)
        manager.add_labels([(2,1)])
        self.assertEqual(first_true(manager.labeled_mask),2)
        self.assertEqual(first_true(manager.unlabeled_mask), 0)
    def test_adding_two_in_the_middle(self):
        features = np.array([[x + y for x in range(10)] for y in range(10)])
        self.assertEqual(features.shape, (10, 10))
        manager = DataManager(features=features)
        manager.add_labels([(2, 1)])
        self.assertEqual(first_true(manager.labeled_mask), 2)
        self.assertEqual(first_true(manager.unlabeled_mask), 0)

        manager.add_labels([(1, 1)])
        self.assertEqual(first_true(manager.labeled_mask), 1)
        # We still didn't label the one at 0
        self.assertEqual(first_true(manager.unlabeled_mask), 0)

    def test_adding_two_in_the_middle_and_then_at_0(self):
        features = np.array([[x + y for x in range(10)] for y in range(10)])
        self.assertEqual(features.shape, (10, 10))
        manager = DataManager(features=features)
        manager.add_labels([(2, 1)])
        self.assertEqual(first_true(manager.labeled_mask), 2)
        self.assertEqual(first_true(manager.unlabeled_mask), 0)

        manager.add_labels([(1, 1)])
        self.assertEqual(first_true(manager.labeled_mask), 1)
        # We still didn't label the one at 0
        self.assertEqual(first_true(manager.unlabeled_mask), 0)

        manager.add_labels([(0, 1)])
        self.assertEqual(first_true(manager.labeled_mask), 0)
        # we labeled 0,1,2 the next one should be 3
        self.assertEqual(first_true(manager.unlabeled_mask), 3)




if __name__ == '__main__':
    unittest.main()
