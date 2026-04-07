from abc import ABC, abstractmethod

class DataParser(ABC):
	def __init__(self):
		self.class_map = {
			"noise":				0,
			"DJI_Mavic_Mini":		1,
			"Parrot_Disco":			2,
		 }
		pass

	@abstractmethod
	def parse_data_object(self, data):
		pass

	@abstractmethod
	def extract_training_data_and_label(self, data):
		pass
