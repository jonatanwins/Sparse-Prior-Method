# Compressed Sensing With Priors

This repository contains the code, notebooks, and generated figures used for a
thesis project on sparse source recovery in underdetermined microphone-array
systems.

The maintained implementation is the Python package in
[`src/cs_priors`](src/cs_priors). The main thesis-facing notebooks and figure
scripts are in [`notebooks/how_to_run_the_methods.ipynb`](notebooks/how_to_run_the_methods.ipynb)
and [`notebooks/illustrations`](notebooks/illustrations). Historical notebooks
are retained for provenance, but are not the recommended starting point.

## Repository Layout

- [`src/cs_priors`](src/cs_priors): maintained package code.
- [`src/cs_priors/simulation/mixing_model.py`](src/cs_priors/simulation/mixing_model.py): frequency-domain simulation pipeline, geometry construction, noise models, and inverse initialization.
- [`src/cs_priors/solvers/freq_lasso.py`](src/cs_priors/solvers/freq_lasso.py): augmented-real LASSO baseline.
- [`src/cs_priors/solvers/freq_group_lasso.py`](src/cs_priors/solvers/freq_group_lasso.py): Group LASSO baseline with configurable grouping.
- [`src/cs_priors/solvers/freq_sparse_prior.py`](src/cs_priors/solvers/freq_sparse_prior.py): sparse-prior solver used in the thesis experiments.
- [`src/cs_priors/tests`](src/cs_priors/tests): package tests.
- [`notebooks/illustrations`](notebooks/illustrations): thesis figure notebooks and runnable visualization scripts.
- [`results/illustrations`](results/illustrations): generated sparse-prior geometry figures.
- [`results/benchmarks`](results/benchmarks): generated benchmark artifacts.

## Installation

Use Python 3.11 if possible. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

If you use Conda instead, create and activate an environment first, then run the
same `pip install` commands.

If the editable install is run without internet access and build dependencies
are already available in the environment, use:

```bash
python -m pip install -e . --no-build-isolation
```

The PyVista-based isosurface viewer is optional. Install PyVista only if you
want to rerun the interactive 3D isosurface visualization:

```bash
python -m pip install pyvista
```

## Model

The maintained simulations use the frequency-domain model

$$
Y = A(X + \delta_X) + \eta_Y
$$

where `X` is the true source matrix, `A` is the complex mixing model,
`\delta_X` is source-model perturbation, `\eta_Y` is sensor noise, and `Y` is
the observed microphone data.

The main array shapes are:

- `A`: `(M, S, F)` for microphones, candidate sources, and frequency bins.
- `Y`: `(M, F)`.
- `X`: `(S, F)`.

## Quick Start

```python
from cs_priors.simulation.mixing_model import quick_sim
from cs_priors.solvers.freq_lasso import frequency_lasso_solve
from cs_priors.solvers.freq_group_lasso import frequency_group_lasso_solve
from cs_priors.solvers.freq_sparse_prior import sparse_prior_solve

sim = quick_sim(
    num_mics=5,
    num_sources=7,
    num_active=2,
    seed=0,
    mode="noise",
    min_freq_hz=1.0,
    sensor_snr_db=100.0,
    inverse_method="ridge",
)

X_lasso = frequency_lasso_solve(sim.Y, sim.A)
X_group_lasso = frequency_group_lasso_solve(sim.Y, sim.A, grouping="frequency")
X_sparse_prior = sparse_prior_solve(sim.X_pinv, sim.A, grouping="frequency")
```

For a notebook walkthrough of the methods, open
[`notebooks/how_to_run_the_methods.ipynb`](notebooks/how_to_run_the_methods.ipynb).

## Reproducing Thesis Figures

The sparse-prior geometry figures are generated from
[`notebooks/illustrations`](notebooks/illustrations). The most relevant entry
points are:

```bash
python notebooks/illustrations/sparse_prior_visuals.py
python notebooks/illustrations/sparse_prior_surface_viewer.py --save results/illustrations/sparse_prior_surface_view.png --no-show
python notebooks/illustrations/sparse_prior_unrolled_viewer.py --save results/illustrations/sparse_prior_unrolled_view.png --no-show
```

The PyVista isosurface workflow is also exposed through
[`notebooks/illustrations/interactive_grouped_prior_3d.py`](notebooks/illustrations/interactive_grouped_prior_3d.py).
It requires the optional `pyvista` dependency.

Generated figure files are written to [`results/illustrations`](results/illustrations).

## Reproducing Benchmark Artifacts

Benchmark notebooks and generated outputs are kept separately:

- benchmark notebooks: [`notebooks/benchmarks_v1`](notebooks/benchmarks_v1)
- generated benchmark figures and tables: [`results/benchmarks`](results/benchmarks)

The benchmark notebooks can be run from the repository root after installing the
package in editable mode. Some notebooks are computationally heavier than the
minimal examples, so they are not part of the default test command.

## Tests

Run the maintained package tests with:

```bash
python -m pytest src/cs_priors/tests
```

These tests cover the frequency-domain simulation pipeline, sparse-prior solver
behavior, and support-leakage metrics.

If pytest crashes before collecting tests because of local terminal/capture
setup, rerun with capture disabled:

```bash
python -m pytest -p no:capture src/cs_priors/tests
```

## Maintained and Historical Material

Use these paths first:

- [`src/cs_priors`](src/cs_priors)
- [`notebooks/how_to_run_the_methods.ipynb`](notebooks/how_to_run_the_methods.ipynb)
- [`notebooks/illustrations`](notebooks/illustrations)
- [`results/illustrations`](results/illustrations)
- [`results/benchmarks`](results/benchmarks)

The folders [`notebooks/legacy`](notebooks/legacy),
[`notebooks/legacy_visualize_sparse_prior`](notebooks/legacy_visualize_sparse_prior),
[`notebooks/recent_meeting_showcases`](notebooks/recent_meeting_showcases), and
[`unused`](unused) are retained as historical working material. They may contain
outdated imports, exploratory code, or intermediate results, and should not be
treated as the maintained thesis entry points.

## Generated Results

Files in [`results`](results) are generated artifacts used for the thesis
write-up and review. They are useful for checking expected outputs, but the
source of truth is the package code and the maintained notebooks/scripts listed
above.
