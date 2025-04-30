## Installation
In the root folder run

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

```python
from cs_model.app import run_simulation
from cs_model.plotting import plot_geometry_auto, plot_composite_overview

basic = run_simulation(array_type="linear")
plot_geometry_auto(basic.x_mics, basic.y_mics, basic.sources, figsize=(4,4))
plot_composite_overview(basic.t, basic.composite_waveforms, figsize=(6,4))
```