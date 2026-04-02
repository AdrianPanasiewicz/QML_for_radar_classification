from abc import ABC, abstractmethod
from Data.environment_classes import Drone, Radar, Context
from Data.synthetic_dataset_generator import DatasetMetadata

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
