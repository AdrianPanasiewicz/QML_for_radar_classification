# Drone Classification from Radar Data

Synthetic radar return signals are generated using the **Martin-Mulgrew model**, which captures
the micro-Doppler signatures produced by rotating drone blades. These signals are used to
benchmark classical and quantum classification approaches.

## Methods

**Signal Generation** — The Martin-Mulgrew model parametrises each drone by blade count `N`,
blade lengths `L_1`/`L_2`, and rotor frequency `f_rot`. Signals are generated synthetically
for five drone classes: DJI Matrice 300 RTK, DJI Mavic Air 2, DJI Mavic Mini, DJI Phantom 4,
and Parrot Disco.

**Classification** — Three classifiers are implemented and compared:
- Quantum Support Vector Machine (QSVM)
- Quantum Neural Network (QNN)
- Deep Neural Network (DNN)

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

Dependencies:
- `numpy`, `sympy`, `scipy` — signal generation and mathematics
- `qiskit` / `pennylane` — quantum circuit implementations
- `torch` — deep neural network classifier
- `matplotlib` — visualisation


## Usage

```python
drone  = Drone(name="DJI_Phantom_4", N=2, L_1=0.006, L_2=0.05, f_rot=116)
radar  = Radar(λ=0.03, f_c=10e9)
gen    = SyntheticSignalGenerator(drone, radar)

context = Context(R=200, V_rad=10, θ=1.57, Φ_p=1.57, A_r=1,
                  t_start=0, t_stop=0.5, dt=0.001)
                  
t, signal = gen.generate_signal(context)
```

## Status

Work in progress. Description to be expanded.