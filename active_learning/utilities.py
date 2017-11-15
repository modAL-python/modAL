import numpy as np


def uncertainty(model, data):
	uncertainties = model.score(data)
	max_uncertainty_idx = np.argmax(uncertainties)
	return max_uncertainty_idx, data[max_uncertainty_idx]
