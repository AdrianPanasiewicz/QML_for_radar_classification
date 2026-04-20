from abc import ABC, abstractmethod
from Data.Generators.synthetic_dataset_generator import DatasetMetadata
import pickle


class FileLoader(ABC):
	def __init__(self, dataset_metadata: DatasetMetadata):
		self.dataset_metadata = dataset_metadata
		pass

	@abstractmethod
	def peek_sample(self, index):
		pass

	@abstractmethod
	def load_all_data(self):
		pass


class SyntheticDataFileLoader(FileLoader):
	def __init__(self, dataset_metadata: DatasetMetadata):
		super().__init__(dataset_metadata)
		pass

	def peek_sample(self, index=1):
		md = self.dataset_metadata
		full_path = md.save_path / f"{md.filename}.{md.file_format}"
		if md.file_format == 'pkl':
			with open(full_path, "rb") as f:
				for i in range(index + 1):
					obj = pickle.load(f)
			return obj
		else:
			raise ValueError("Only pkl files are supported")

	def load_all_data(self):
		md = self.dataset_metadata
		full_path = md.save_path / f"{md.filename}.{md.file_format}"
		data = []
		with open(full_path, 'rb') as f:
			try:
				while True:
					data.append(pickle.load(f))
			except EOFError:
				pass
		return data