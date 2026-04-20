from abc import ABC, abstractmethod

class DataParser(ABC):
	def __init__(self):
		self.class_map = {
			"DJI_Mavic_Mini":		0,
			"Parrot_Disco":			1,
		 }
		pass

	@abstractmethod
	def parse_data_object(self, data):
		pass

	@abstractmethod
	def extract_training_data_and_label(self, data):
		pass
