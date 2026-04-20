from torch.utils.data import Dataset
from torch import stack
from MachineLearning.Processing.time_domain_parser import TimeDomainDataParser
from MachineLearning.Processing.file_loader import SyntheticDataFileLoader
from Data.Generators.synthetic_dataset_generator import DatasetMetadata

class SyntheticTimeDomainRadarDataset(Dataset):
	def __init__(self, dataset_file_path, transform=None, mean=None, std=None, target_transform=None):
		super().__init__()

		md = DatasetMetadata.create_from_path(dataset_file_path)
		loader = SyntheticDataFileLoader(dataset_metadata=md)
		self.td_data_parser = TimeDomainDataParser()
		self.transform = transform
		self.target_transform = target_transform

		raw = loader.load_all_data()
		self.metadata = raw[0]
		raw_data = raw[1:]

		unnormalized_data = self._parse_data(raw_data)
		if mean is None or std is None:
			self.mean, self.std = self._compute_stats(unnormalized_data)
		else:
			self.mean, self.std = mean, std
		self._data = unnormalized_data

		self.dataset_name = self.metadata['dataset_name']

	def __len__(self):
		return len(self._data)

	def __getitem__(self, idx):
		data, label = self._data[idx]
		data = (data - self.mean) / self.std
		if self.transform:
			data = self.transform(data)
		if self.target_transform:
			label = self.target_transform(label)
		return data, label

	def _parse_data(self, data):
		parsed_data = []
		for obj in data:
			parsed_signal, label, misc_data = self.td_data_parser.parse_data_object(obj)
			parsed_data.append([parsed_signal, label])
		return parsed_data

	def _compute_stats(self, data):
		all_signals = stack([item[0] for item in data], dim=0)
		mean = all_signals.mean()
		std = all_signals.std()
		return mean, std
