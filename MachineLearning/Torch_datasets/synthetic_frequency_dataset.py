from torch.utils.data import Dataset
from torch import tensor, stack
from torch.nn.functional import normalize
from Data.Generators.synthetic_dataset_generator import DatasetMetadata
from MachineLearning.Processing.file_loader import SyntheticDataFileLoader
from MachineLearning.Processing.frequency_domain_parser import FrequencyDomainDataParser

class SyntheticFrequencyDomainRadarDataset(Dataset):
	def __init__(self, dataset_file_path, transform=None, target_transform=None):
		super().__init__()

		md = DatasetMetadata.create_from_path(dataset_file_path)
		loader = SyntheticDataFileLoader(dataset_metadata=md)

		raw = loader.load_all_data()
		self.metadata = raw[0]
		self._data = raw[1:]

		self._normalize_data()

		self.dataset_name = self.metadata['dataset_name']
		self._length = self.metadata['len']
		self.td_data_parser = FrequencyDomainDataParser()

		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return self._length

	def __getitem__(self, idx):
		# obj = self.synt_dataset.peek_sample(index=idx)
		obj = self._data[idx]
		parsed_signal, label, misc_data = self.td_data_parser.parse_data_object(obj)
		return parsed_signal, label

	def _normalize_data(self):
		temp_tensor_list = []
		for i in range(len(self._data)):
			temp_tensor_list.append(tensor(self._data[i]['signal']))

		temp_tensor = stack(temp_tensor_list, dim=0)
		temp_tensor = normalize(temp_tensor, dim=(2,3))

		for i in range(len(self._data)):
			self._data[i]['signal'] = temp_tensor[i]