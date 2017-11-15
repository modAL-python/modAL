import numpy as np


def uncertainty(predictor, data):
	uncertainties = predictor.score(data)
	max_uncertainty_idx = np.argmax(uncertainties)
	return max_uncertainty_idx, data[max_uncertainty_idx]
