import numpy as np

SINGLE_FREQUENCY_HZ = 2e4  # 20 kHz,
SAMPLES_PER_PERIOD = 4
SAMPLING_RATE = SAMPLES_PER_PERIOD * SINGLE_FREQUENCY_HZ
DURATION = 1 / SINGLE_FREQUENCY_HZ

SOURCE_DISTANCE = 0.5
MIC_RADIUS = 0.025  # 5cm in diameter
SOURCE_SPACING_DEG = 5.0
NUM_ACTIVE = 2
COMPONENT_AMPLITUDE = 10.0
MAGNITUDE_JITTER = 0.0

PHASE_SEEDS = np.arange(15)
LASSO_ALPHA = 1e-3
LASSO_MAX_ITER = 10_000
FULL_RING_SPAN = 2 * np.pi

METHOD_STYLE = {
    "Sparse Prior from MP": {
        "label": "Sparse Prior, MP",
        "color": "tab:red",
        "linestyle": "-",
    },
    "Sparse Prior from Ridge": {
        "label": "Sparse Prior, Ridge",
        "color": "tab:red",
        "linestyle": "--",
    },
    "LASSO from MP": {
        "label": "LASSO, MP",
        "color": "tab:orange",
        "linestyle": "-",
    },
    "LASSO from Ridge": {
        "label": "LASSO, Ridge",
        "color": "tab:orange",
        "linestyle": "--",
    },
    "MP": {
        "label": "MP pseudoinverse",
        "color": "tab:blue",
        "linestyle": ":",
    },
    "Ridge": {
        "label": "Ridge pseudoinverse",
        "color": "#009E73",
        "linestyle": ":",
    },
    # Old notebook method names
    "Sparse Prior": {
        "label": "Sparse Prior",
        "color": "tab:red",
        "linestyle": "-",
    },
    "r-LASSO": {
        "label": "LASSO",
        "color": "tab:orange",
        "linestyle": "-",
    },
    "X_pinv": {
        "label": r"Initializer $\boldsymbol{x}_0$",
        "color": "tab:blue",
        "linestyle": ":",
    },
}

METHOD_LABELS = {k: v["label"] for k, v in METHOD_STYLE.items()}
METHOD_COLORS = {k: v["color"] for k, v in METHOD_STYLE.items()}
METHOD_LINESTYLES = {k: v["linestyle"] for k, v in METHOD_STYLE.items()}


def sector_span_deg(num_sources):
    return (num_sources - 1) * SOURCE_SPACING_DEG


def base_sim_kwargs(
    *,
    num_sources,
    num_mics=None,
    num_active=NUM_ACTIVE,
    mic_angle_span=None,
    source_angle_span_deg=None,
):
    kwargs = dict(
        num_sources=num_sources,
        num_active=num_active,
        sampling_rate=SAMPLING_RATE,
        duration=DURATION,
        source_distance=SOURCE_DISTANCE,
        mic_radius=MIC_RADIUS,
        mic_angle_start=0.0,
        source_angle_start=0.0,
        component_amplitude=COMPONENT_AMPLITUDE,
        magnitude_jitter=MAGNITUDE_JITTER,
        single_frequency_hz=SINGLE_FREQUENCY_HZ,
    )
    if num_mics is not None:
        kwargs["num_mics"] = num_mics
    if mic_angle_span is not None:
        kwargs["mic_angle_span"] = mic_angle_span
    if source_angle_span_deg is not None:
        kwargs["source_angle_span"] = np.deg2rad(source_angle_span_deg)
    return kwargs
