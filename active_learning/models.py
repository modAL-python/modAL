class ActiveLearner:
	def __init__(self, predictor, utility_function):
		assert callable(utility_function), 'utility_function must be callable'

		self.predictor = predictor
		self.utility_function = utility_function
		self.training_data = None
		self.training_labels = None

	def calculate_utility(self, data):
		return utility_function(self.predictor, data)

	def query(self):
		pass

	def fit(self, observation, label):
		pass

	def give_labels(self):
		pass


class Committee:
	def __init__(self):
		pass

	def vote(self):
		pass
