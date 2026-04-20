from MachineLearning.Processing.abstract_data_parser import DataParser
import re
import numpy as np
import torch
from matplotlib import pyplot as plt


class FrequencyDomainDataParser(DataParser):
	def __init__(self):
		super().__init__()

	def parse_data_object(self, dataset_obj, bin_size=1, return_mag = False):
		signal, label, misc_data = self.extract_training_data_and_label(dataset_obj)

		# signal = self.bin_data(signal, bin_size)
		if return_mag:
			signal = self.compute_magnitude(signal)
		signal = self.to_tensor(signal)

		encoded_label = self.encode_label(label)

		return signal, encoded_label, misc_data

	def extract_training_data_and_label(self, data):
		request_name = data['request'].request_name
		label = re.search(r'label=(\S+)', request_name).group(1)
		signal = data['signal']
		misc_data = data['request']
		return signal, label, misc_data

	def bin_data(self, data, bin_size=1):
		bin_array = []
		for i in range(data.shape[0]//bin_size):
			bin_array.append(np.average(data[bin_size*i:bin_size*(i+1)])*np.sqrt(bin_size))
		return np.array(bin_array)

	def compute_magnitude(self, data):
		mag_data = np.linalg.norm(data, axis=0)
		return mag_data

	def to_tensor(self, data):
		if isinstance(data, np.ndarray):
			return torch.from_numpy(data).float()
		else:
			return data.float()

	def encode_label(self, label):
		return self.class_map[label]

	def plot_drone_spectrogram(self, time_signal, misc_data):

		f_pts = time_signal.shape[1]
		delta_t = 16 * misc_data.context.dt
		delta_f = (1 / misc_data.context.dt) / 32

		fig, axs = plt.subplots(2, 1, figsize=(12, 4), sharex=True, sharey=True)
		fig.suptitle(f"Drone: {misc_data.drone.name}")

		fig.supxlabel(f"Time t (dt={delta_t:g} s) [s]")
		fig.supylabel(f"Freq. f ({f_pts} bins, df={delta_f:g} Hz) [Hz]")

		im1 = axs[0].imshow(time_signal[0], origin='lower', aspect='auto', cmap='viridis')
		axs[0].set_title("Real")
		fig.colorbar(im1, ax=axs[0], label="Magnitude |S(t,f)|")

		im2 = axs[1].imshow(time_signal[1], origin='lower', aspect='auto', cmap='viridis')
		axs[1].set_title("Imag.")
		fig.colorbar(im2, ax=axs[1], label="Magnitude |S(t,f)|")

		fig.tight_layout()
		plt.show()