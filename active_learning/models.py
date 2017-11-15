class ActiveLearner:
	def __init__(self, predictor, utility_function):
		assert callable(utility_function), 'utility_function must be callable'

		self.predictor = predictor
		self.utility_function = utility_function
		self.training_data = None
		self.training_labels = None

	def calculate_utility(self, data):
		return self.utility_function(self.predictor, data)

	def query(self):
		pass

	def fit(self, observation, label):
		pass

	def add_training_data(self, new_data, new_label):
		# TODO: get rid of the if clause
		# TODO: test if this works with multiple shapes and types of data
		if type(training_data) != type(None):
			try:
				self.training_data = np.vstack((self.training_data, new_data))
				self.training_labels = np.vstack((self.training_labels, new_label))
			except ValueError:
				raise ValueError('the dimensions of the new training data and label must'
								 'agree with the training data and labels provided so far')

		else:
			self.training_data = new_data
			self.training_labels = new_label


class Committee:
	def __init__(self):
		pass

	def vote(self):
		pass
