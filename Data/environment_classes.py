import numpy as np
from dataclasses import dataclass
from typing import Union, Callable

Scalar = Union[float, int]
Field  = Union[Scalar, Callable[[np.ndarray], np.ndarray]]

@dataclass
class Drone:
    name: str
    N: int
    L_1: float
    L_2: float
    f_rot: float

@dataclass
class Radar:
    λ: float
    f_c: float

@dataclass
class Context:
    R:       Field
    V_rad:   Field
    θ:       Field
    Φ_p:     Field
    A_r:     Field
    snr:     Field
    t_start: float
    t_stop:  float
    dt:      float

    def resolve(self, field, t: np.ndarray) -> np.ndarray:
        return field(t) if callable(field) else np.full_like(t, field, dtype=float)