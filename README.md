# Compressed Sensing With Priors

This repository studies sparse source recovery for underdetermined microphone-array systems in the frequency domain.

# Carl-Inge:
mest relevante filene å se på for pseudoinverser er
- [test_pinv_noise.ipynb](notebooks/test_pinv_noise.ipynb) som illustrerer hvordan kjøre metodene og simuleringer med forskjellig pseudoinvers og støy.
- [mixing_model.py](src/cs_priors/simulation/mixing_model.py) her er `moore_penrose_inverse` og `ridge_inverse`.
- [test_simulation.py](src/cs_priors/tests/test_simulation.py) som sjekker at inversene oppfører seg slik jeg tror de skal.
- [benchmark.py](src/cs_priors/metrics/benchmark.py) hvordan metodene evalueres

## Installation

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Model

The simulations use the frequency-domain model

\[
Y = A(X + \delta_X) + \eta_Y
\]

where:
- `X` is the true source matrix
- `A` is the complex mixing model
- `delta_X` is source-model perturbation
- `eta_Y` is sensor noise
- `Y` is the observed microphone data

The maintained code uses the following shapes:
- `A`: `(M, S, F)`
- `Y`: `(M, F)`
- `X`: `(S, F)`

## Main Code

- [src/cs_priors/simulation/mixing_model.py](src/cs_priors/simulation/mixing_model.py): simulation pipeline, noise models, and inverse initialization
- [src/cs_priors/solvers/frequency_lasso.py](src/cs_priors/solvers/frequency_lasso.py): augmented-real LASSO baseline
- [src/cs_priors/solvers/frequency_group_lasso.py](src/cs_priors/solvers/frequency_group_lasso.py): Group LASSO with source-frequency grouping
- [src/cs_priors/solvers/multifrequency_sparse_prior.py](src/cs_priors/solvers/multifrequency_sparse_prior.py): multi-frequency sparse-prior solver
- [src/cs_priors/metrics/benchmark.py](src/cs_priors/metrics/benchmark.py): grid benchmarking over seeds and simulation parameters

## Minimal Example

```python
from cs_priors.simulation.mixing_model import quick_sim
from cs_priors.solvers.frequency_lasso import frequency_lasso_solve

sim = quick_sim(
    num_mics=7,
    num_sources=10,
    num_active=2,
    seed=0,
    sensor_snr_db=20.0,
    min_freq_hz=1.0,
)

X_hat = frequency_lasso_solve(sim.Y, sim.A, alpha=1e-4)
```

## Notes

This repository is still being cleaned up for thesis use. The `notebooks/` folder contains the current entry points. Older exploratory material is kept only as historical reference and is not the recommended place to start.
