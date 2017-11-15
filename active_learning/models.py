class ActiveLearner:
	def __init__(
			self,
			predictor, utility_function, 					# building blocks of the learner
			initial_data=None, initial_labels=None			# initial data if available
	):
		assert callable(utility_function), 'utility_function must be callable'

		self.predictor = predictor
		self.utility_function = utility_function
		self.training_data = initial_data
		self.training_labels = initial_labels

		if (type(initial_data) != type(None)) and (type(initial_labels) != type(None)):
			self.fit_to_known()

	def calculate_utility(self, data):
		return self.utility_function(self.predictor, data)

	def query(self):
		pass

	def fit_to_known(self):
		"""
		This method fits self.predictor to the training data and labels
		provided to it so far.
		"""
		self.predictor.fit(self.training_data, self.training_labels)

	def add_and_retrain(self, new_data, new_label):
		"""
		This function adds the given data to the training examples
		and retrains the predictor with the augmented dataset
		:param new_data: new training data
		:param new_label: new training labels for the data
		"""
		self.add_training_data(new_data, new_label)
		self.fit_to_known()

	def add_training_data(self, new_data, new_label):
		# TODO: get rid of the if clause
		# TODO: test if this works with multiple shapes and types of data

		assert len(new_data) == len(new_label), 'the number of new data points and number of labels must match'

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
