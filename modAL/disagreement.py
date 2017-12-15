"""
Disagreement measures for the Committee model.
"""

import numpy as np
from collections import Counter
from scipy.stats import entropy


def vote_entropy(committee, data, **predict_proba_kwargs):
    n_learners = len(committee)
    votes = committee.vote(data, **predict_proba_kwargs)
    vote_proba = np.zeros(shape=(data.shape[0], len(committee.classes_)))
    entr = np.zeros(shape=(data.shape[0], ))

    for vote_idx, vote in enumerate(votes):
        vote_counter = Counter(vote)

        for class_idx, class_label in enumerate(committee.classes_):
            vote_proba[vote_idx, class_idx] = vote_counter[class_label]/n_learners

        entr[vote_idx] = entropy(vote_proba[vote_idx])

    return entr
