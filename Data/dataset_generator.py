from environment_classes import Drone, Radar, Context
from synthetic_signal_generator import SyntheticSignalGenerator
from noise_models import BaseNoiseModel
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
    def create_from_save_path(cls, path: Path):
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

    def generate_signal_data(self):    # Can be optimized with multiprocessing and buffering saving, but it is good enough
        md = self.dataset_metadata
        full_path = md.save_path / f"{md.filename}.{md.file_format}"

        with open(full_path, "ab") as f:
            for req in tqdm(self.data_requests):
                sig_gen = SyntheticSignalGenerator(req.drone, req.radar, req.noise_model)
                for _ in range(req.sample_size):
                    _, signal = sig_gen.generate_signal(req.context)
                    pickle.dump({"request": req, "signal": signal}, f)

        self.data_requests = []