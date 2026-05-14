import numpy as np

NUM_SOURCES = 10
NUM_ACTIVE = 2
MIC_COUNTS = np.array([2, 3, 4, 5, 6, 7])
SEEDS = np.arange(15)

SOURCE_SPACING_DEG = 5.0
SOURCE_ANGLE_START_DEG = 0.0
SOURCE_ANGLE_SPAN = np.deg2rad((NUM_SOURCES - 1) * SOURCE_SPACING_DEG)

ALPHA = 1e-3
MAX_ITER = 1000

METHOD_ORDER = ["LASSO", "Group LASSO, no groups", "Group LASSO, frequency groups"]
COMPARISON_METHOD_ORDER = [
    "Sparse Prior, no groups",
    "Sparse Prior, frequency groups",
    "LASSO",
    "Group LASSO, frequency groups",
    "MP",
]
METHOD_STYLES = {
    "LASSO": {"color": "tab:orange", "linestyle": "-"},
    "Group LASSO, no groups": {"color": "tab:orange", "linestyle": ":"},
    "Group LASSO, frequency groups": {"color": "tab:green", "linestyle": "-"},
    "MP": {"color": "tab:blue", "linestyle": ":"},
    "Sparse Prior, no groups": {"color": "tab:red", "linestyle": ":"},
    "Sparse Prior, frequency groups": {"color": "tab:red", "linestyle": "-"},
}
ALL_METHOD_ORDER = METHOD_ORDER + [
    method for method in COMPARISON_METHOD_ORDER if method not in METHOD_ORDER
]

METHOD_COLORS = {k: v["color"] for k, v in METHOD_STYLES.items()}
METHOD_LINESTYLES = {k: v["linestyle"] for k, v in METHOD_STYLES.items()}
METHOD_LABELS = {
    "MP": "Moore-Penrose",
    "LASSO": "LASSO",
    "Group LASSO, no groups": "Group LASSO, no groups",
    "Group LASSO, frequency groups": "Group LASSO, frequency groups",
    "Sparse Prior, no groups": "Sparse Prior, no groups",
    "Sparse Prior, frequency groups": "Sparse Prior, frequency groups",
}


def base_sim_kwargs():
    return dict(
        num_sources=NUM_SOURCES,
        num_active=NUM_ACTIVE,
        array_type="circular",
        mic_radius=0.025,
        mic_angle_span=2 * np.pi,
        source_distance=0.5,
        source_angle_start=np.deg2rad(SOURCE_ANGLE_START_DEG),
        source_angle_span=SOURCE_ANGLE_SPAN,
        sampling_rate=80_000.0,
        duration=2.5e-4,
        min_freq_hz=4_000.0,
        single_frequency_hz=None,
        component_amplitude=1.0,
        magnitude_jitter=0.0,
        sensor_snr_db=None,
        model_snr_db=None,
        inverse_method="mp",
    )
