"""
This example shows how to use the new data manager class.
For clarity, all the setup has been moved into functions and
the core is in the __main__ section which is commented

Also look at prepare_manager to see how a DataManager is instantiated

"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.ensemble import RandomForestClassifier
from modAL.datamanager import DataManager
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from functools import partial


from modAL.models import ActiveLearner
from modAL.batch import uncertainty_batch_sampling

RANDOM_STATE_SEED = 123
np.random.seed(RANDOM_STATE_SEED)
BATCH_SIZE = 5
N_QUERIES = 50


def prepare_data():
    SKIP_SIZE = 50  # Skip to make the example go fast.
    docs, original_labels = fetch_20newsgroups(return_X_y=True)
    docs_train = docs[::SKIP_SIZE]
    original_labels_train = original_labels[::SKIP_SIZE]
    docs_test = docs[1::SKIP_SIZE]  # Offset by one means no overlap
    original_labels_test = original_labels[
        1::SKIP_SIZE
    ]  # Offset by one means no overlap
    return docs_train, original_labels_train, docs_test, original_labels_test


def prepare_features(docs_train, docs_test):
    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 3), max_df=0.9, max_features=5000
    )

    vectors_train = vectorizer.fit_transform(docs_train).toarray()
    vectors_test = vectorizer.transform(docs_test).toarray()
    return vectors_train, vectors_test


def prepare_manager(vectors_train, docs_train):
    manager = DataManager(vectors_train, sources=docs_train)
    return manager


def prepare_learner():

    estimator = RandomForestClassifier()
    preset_batch = partial(uncertainty_batch_sampling, n_instances=BATCH_SIZE)
    learner = ActiveLearner(estimator=estimator, query_strategy=preset_batch)
    return learner


def make_pretty_summary_plot(performance_history):
    with plt.style.context("seaborn-white"):
        fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)

        ax.plot(performance_history)
        ax.scatter(range(len(performance_history)), performance_history, s=13)

        ax.xaxis.set_major_locator(
            mpl.ticker.MaxNLocator(nbins=N_QUERIES + 3, integer=True)
        )
        ax.xaxis.grid(True)

        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
        ax.set_ylim(bottom=0, top=1)
        ax.yaxis.grid(True, linestyle="--", alpha=1 / 2)

        ax.set_title("Incremental classification accuracy")
        ax.set_xlabel("Query iteration")
        ax.set_ylabel("Classification Accuracy")

        plt.show()


if __name__ == "__main__":
    docs_train, original_labels_train, docs_test, original_labels_test = prepare_data()
    vectors_train, vectors_test = prepare_features(docs_train, docs_test)
    manager = prepare_manager(vectors_train, docs_train)
    learner = prepare_learner()
    performance_history = []
    # performance_history.append(learner.score(docs_test, original_labels_test))

    for i in range(N_QUERIES):
        # Check if there are more examples that are not labeled. If not, break
        if manager.unlabeld.size == 0:
            break

        for index in range(1):
            # query the learner as usual, in this case we are using a batch learning strategy
            # so indices_to_label is an array
            indices_to_label, query_instance = learner.query(manager.unlabeld)
            labels = []  # Hold a list of the new labels
            for ix in indices_to_label:
                """
                Here is the tricky part that the manager solves. The indicies are indexed with respect to unlabeled data
                but we want to work with them with respect to the original data. The manager makes this almost transparent
                """
                # Map the index that is with respect to unlabeled data back to an index with respect to the whole dataset
                original_ix = manager.get_original_index_from_unlabeled_index(ix)
                # print(manager.sources[original_ix]) #Show the original data so we can decide what to label
                # Now we can lookup the label in the original set of labels without any bookkeeping
                y = original_labels_train[original_ix]
                # We create a Label instance, a tuple of index and label
                # The index should be with respect to the unlabeled data, the add_labels function will automatically
                # calculate the offsets
                label = (ix, y)
                # append the labels to a list
                labels.append(label)
            # Insert them all at once.
            manager.add_labels(labels)
            # Note that if you need to add labels with indicies that repsect the original dataset you can do
            # manager.add_labels(labels,offset_to_unlabeled=False)
        # Now teach as usual
        learner.teach(manager.labeled, manager.labels)
        performance_history.append(learner.score(vectors_test, original_labels_test))
    # Finnaly make a nice plot
    make_pretty_summary_plot(performance_history)
