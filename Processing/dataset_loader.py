from abc import ABC, abstractmethod
from Data.

class DatasetClass(ABC):
	def __init__(self):
		pass

	@abstractmethod
	def load_data(self):
		pass

	@abstractmethod
	def prepare_data(self):
		pass

	def export_data(self):
		pass


class SynthethicDataset(DatasetClass):
	def __init__(self):
		super().__init__()
		pass

	def load_data(self):
		pass

	def prepare_data(self):
		pass

	def export_data(self):
		pass
