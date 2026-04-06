from torch.utils.data import Dataset
from MachineLearning.Preprocessing.time_domain_parser import TimeDomainDataParser
from MachineLearning.Preprocessing.file_loader import SyntheticDataFileLoader
from Data.Generators.synthetic_dataset_generator import DatasetMetadata

class SyntheticTimeDomainRadarDataset(Dataset):
	def __init__(self, dataset_file_path, transform=None, target_transform=None):
		super().__init__()

		md = DatasetMetadata.create_from_path(dataset_file_path)
		loader = SyntheticDataFileLoader(dataset_metadata=md)

		raw = loader.load_all_data()
		self.metadata = raw[0]
		self._data = raw[1:]

		self.dataset_name = self.metadata['dataset_name']
		self._length = self.metadata['len']
		self.td_data_parser = TimeDomainDataParser()

		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return self._length

	def __getitem__(self, idx):
		# obj = self.synt_dataset.peek_sample(index=idx)
		obj = self._data[idx]
		parsed_signal, label, misc_data = self.td_data_parser.parse_data_object(obj)
		return parsed_signal, label