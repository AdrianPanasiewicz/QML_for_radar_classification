from Data.Primitives.environment_classes import Drone, Radar, Context
from Data.Primitives.noise_models import AdditiveWhiteGaussianNoise
from Data.Generators.synthetic_signal_generator import SyntheticSignalGenerator
from Data.Primitives.noise_models import BaseNoiseModel
from pathlib import Path
from tqdm import tqdm
import pickle
import warnings
from dataclasses import dataclass

@dataclass
class DatasetMetadata:
    file_format:    str
    filename:       str
    save_path:      Path

    @classmethod
    def create_from_path(cls, path: Path):
        return cls(
            file_format=path.suffix.lstrip("."),
            filename=path.stem,
            save_path=path.parent
        )

@dataclass
class DataRequest:
    request_name:   str
    drone:          Drone
    radar:          Radar
    context:        Context
    noise_model:    BaseNoiseModel
    sample_size:    int


class SyntheticDatasetGenerator:
    def __init__(self, dataset_metadata: DatasetMetadata):
        self.dataset_metadata = dataset_metadata
        self.data_requests = []
        self._noise_config = None

        self._create_dataset_file()

    def _create_dataset_file(self):
        md = self.dataset_metadata
        full_path = md.save_path / f"{md.filename}.{md.file_format}"
        full_path.parent.mkdir(parents=True, exist_ok=True)

        if full_path.exists():
            warnings.warn(f"File '{full_path}' already exists. Resolving name conflict.")
            i = 1
            while full_path.exists():
                new_filename = f"{md.filename}_({i})"
                full_path = md.save_path / f"{new_filename}.{md.file_format}"
                i += 1
            self.dataset_metadata.filename = new_filename

        if md.file_format in ("pkl",):       # For now only pkl format has the native support
            full_path.touch()
        else:
            raise ValueError(f"Unsupported file format: {md.file_format}")


    def save_data(self, data, metadata):
        md = self.dataset_metadata
        full_path = md.save_path / f"{md.filename}.{md.file_format}"
        if md.file_format == "pkl":
            with open(full_path, "ab") as f:
                pickle.dump({"request": metadata, "signal": data}, f)
        else:
            raise ValueError(f"Unsupported file format: {md.file_format}")


    def append_data_requests(self, data_requests: list[DataRequest]):
        for item in data_requests:
            if not isinstance(item, DataRequest):
                raise TypeError("Submitted data_requests contain non DataRequest items")

        self.data_requests+=data_requests

    def set_noise_samples(self, n: int, t_len: float, dt: float):
        self._noise_config = (n, t_len, dt)


    def _generate_noise_data(self, file_handle, n: int, t_len: float, dt: float, stft_form=False):
            noise_req_shape = int(t_len / dt)
            noise_metadata = DataRequest(
                request_name = "label=noise",
                drone        = None,
                radar        = None,
                context      = Context(
                    R=None, V_rad=None, θ=None, Φ_p=None,
                    A_r=None, snr=None, t_start=0, t_stop=t_len, dt=dt
                ),
                noise_model  = AdditiveWhiteGaussianNoise(),
                sample_size  = n
            )
            for _ in tqdm(range(n), desc="Noise samples"):
                noise_data = AdditiveWhiteGaussianNoise.generate_noise(sigma=1, shape=noise_req_shape)
                if stft_form:
                    noise_data = SyntheticSignalGenerator.apply_stft(noise_data, noise_metadata.context)

                pickle.dump({"request": noise_metadata, "signal": noise_data}, file_handle)

    def generate_signal_data(self, stft_form=False):
        md = self.dataset_metadata
        full_path = md.save_path / f"{md.filename}.{md.file_format}"

        signal_count = sum(req.sample_size for req in self.data_requests)
        noise_count = self._noise_config[0] if self._noise_config is not None else 0
        dataset_len = signal_count + noise_count

        with open(full_path, "ab") as f:
            pickle.dump({"dataset_name": md.filename, "len": dataset_len}, f)

            with tqdm(total=signal_count, desc="Signal samples") as pbar:
                for req in self.data_requests:
                    sig_gen = SyntheticSignalGenerator(req.drone, req.radar, req.noise_model)
                    for _ in range(req.sample_size):
                        _, signal = sig_gen.generate_signal(req.context, stft_form)
                        pickle.dump({"request": req, "signal": signal}, f)
                        pbar.update(1)

            if self._noise_config is not None:
                n, t_start, dt = self._noise_config
                self._generate_noise_data(f, n, t_start, dt, stft_form)

        self.data_requests = []
        self._noise_config = None