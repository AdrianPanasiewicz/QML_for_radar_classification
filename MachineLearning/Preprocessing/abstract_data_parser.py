from abc import ABC, abstractmethod

class DataParser(ABC):
	def __init__(self):
		pass

	@abstractmethod
	def parse_data_object(self, data):
		pass

	@abstractmethod
	def extract_training_data_and_label(self, data):
		pass
