"""
Disagreement measures for the Committee model.
"""

import numpy as np
from collections import Counter


def vote_entropy(committee, data, **predict_proba_kwargs):
    check_array(data)

    vote = committee.predict(data, **predict_proba_kwargs)
    vote_proba = np.zeros(shape=())
