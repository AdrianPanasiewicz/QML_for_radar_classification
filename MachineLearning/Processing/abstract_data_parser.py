from abc import ABC, abstractmethod

class DataParser(ABC):
	def __init__(self, language="english"):
		self.class_map = {
			"DJI_Mavic_Mini":		0,
			"Parrot_Disco":			1,
		 }

		self.translations = {
			"english": {
				"spectrogram": {
					"figure_title": "Drone: {drone_name}, SNR={snr} dB",
					"xlabel": "Time t (dt={delta_t:g} s) [s]",
					"ylabel": "Frequency f ({f_pts} bins, df={delta_f:g} Hz) [Hz]",
					"real_title": "Real part",
					"imag_title": "Imaginary part",
					"colorbar": "Magnitude |S(t,f)|"
				}
			},
			"polish": {
				"spectrogram": {
					"figure_title": "Dron: {drone_name}, SNR={snr} dB",
					"xlabel": "Czas t (dt={delta_t:g} s) [s]",
					"ylabel": "Częstotliwość f ({f_pts} prążków, df={delta_f:g} Hz) [Hz]",
					"real_title": "Składowa rzeczywista",
					"imag_title": "Składowa urojona",
					"colorbar": "Moduł |S(t,f)|"
				}
			}
		}
		self.language = language.lower()
		self.labels = self.translations.get(self.language, self.translations["english"])

	def set_language(self, language):
		self.language = language.lower()
		self.labels = self.translations.get(self.language, self.translations["english"])

	def get_labels(self, plot_type):
		return self.labels[plot_type]

	@abstractmethod
	def parse_data_object(self, data):
		pass

	@abstractmethod
	def extract_training_data_and_label(self, data):
		pass
