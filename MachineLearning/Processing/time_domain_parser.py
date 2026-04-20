from MachineLearning.Processing.abstract_data_parser import DataParser
from matplotlib import pyplot as plt
import re
import numpy as np
from scipy.fft import fft
import torch
import scipy

class TimeDomainDataParser(DataParser):
	def __init__(self):
		super().__init__()

	def parse_data_object(self, dataset_obj, bin_size=100):
		signal, label, misc_data = self.extract_training_data_and_label(dataset_obj)

		binned_signal = self.bin_data(signal, bin_size)
		freq_signal = self.discrete_fourier_transform(binned_signal)
		parsed_signal = self.compute_modulus(freq_signal)
		parsed_tensor = self.to_tensor(parsed_signal)

		encoded_label = self.encode_label(label)

		return parsed_tensor, encoded_label, misc_data

	def extract_training_data_and_label(self, data):
		request_name = data['request'].request_name
		label = re.search(r'label=(\S+)', request_name).group(1)
		signal = data['signal']
		misc_data = data['request']
		return signal, label, misc_data

	def bin_data(self, data, bin_size=100):
		bin_array = []
		for i in range(data.shape[0]//bin_size):
			bin_array.append(np.average(data[bin_size*i:bin_size*(i+1)])*np.sqrt(bin_size))
		return np.array(bin_array)

	def discrete_fourier_transform(self, time_domain_data):
		frequency_domain_data = fft(time_domain_data)
		return frequency_domain_data

	def compute_modulus(self, data):
		modulus_data = np.absolute(data)
		return modulus_data

	def to_tensor(self, data):
		if isinstance(data, np.ndarray):
			return torch.from_numpy(data).float()
		else:
			return data.float()

	def encode_label(self, label):
		return self.class_map[label]

	def apply_stft(self, time_signal, misc_data, nperseg=32, noverlap=16):
		f, t, Zreal = scipy.signal.stft(
		time_signal.real, 1 / misc_data.context.dt, window='hamming', nperseg=nperseg, noverlap=noverlap, return_onesided=True)
		Xreal = 20*np.log10(np.abs(Zreal))

		f, t, Zimag = scipy.signal.stft(
		time_signal.imag, 1 / misc_data.context.dt, window='hamming', nperseg=nperseg, noverlap=noverlap, return_onesided=True)
		Ximag = 20*np.log10(np.abs(Zimag))

		return np.stack((Xreal[1:,:], Ximag[1:,:]))

	def plot_drone_spectrogram(self, time_signal, misc_data, nperseg=32, noverlap=16):

		stft_signal = self.apply_stft(time_signal, misc_data, nperseg, noverlap)

		f_pts = stft_signal.shape[1]
		delta_t = noverlap * misc_data.context.dt
		delta_f = (1 / misc_data.context.dt) / nperseg

		fig, axs = plt.subplots(2, 1, figsize=(12, 4), sharex=True, sharey=True)
		fig.suptitle(f"Drone: {misc_data.drone.name}")

		fig.supxlabel(f"Time t (dt={delta_t:g} s) [s]")
		fig.supylabel(f"Freq. f ({f_pts} bins, df={delta_f:g} Hz) [Hz]")

		im1 = axs[0].imshow(stft_signal[0], origin='lower', aspect='auto', cmap='viridis')
		axs[0].set_title("Real")
		fig.colorbar(im1, ax=axs[0], label="Magnitude |S(t,f)|")

		im2 = axs[1].imshow(stft_signal[1], origin='lower', aspect='auto', cmap='viridis')
		axs[1].set_title("Imag.")
		fig.colorbar(im2, ax=axs[1], label="Magnitude |S(t,f)|")

		fig.tight_layout()
		plt.show()