from abc import ABC, abstractmethod
from MachineLearning.Torch_datasets.synthetic_time_dataset import SyntheticTimeDomainRadarDataset

class AbstractTrainer(ABC):
    def __init__(self, training_path, validating_path, testing_path, criterion):
        self.trainset = SyntheticTimeDomainRadarDataset(training_path)
        self.valset = SyntheticTimeDomainRadarDataset(validating_path, mean=self.trainset.mean, std=self.trainset.std)
        self.testset = SyntheticTimeDomainRadarDataset(testing_path, mean=self.trainset.mean, std=self.trainset.std)
        self.criterion = criterion

    @abstractmethod
    def train_model(self, config, model):
        pass

    @abstractmethod
    def test_model(self, model):
        pass