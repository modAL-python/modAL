class ActiveLearner:
	def __init__(self, predictor, utility_function):
		self.predictor = predictor
		self.utility_function = utility_function
		self.training_data = None
		self.training_labels = None

	def calculate_utility(self, data):
		return

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
