import numpy as np
from numpy.testing import assert_allclose

from ..metrics.count_sparsity import (
    format_source_leakage_summary,
    source_leakage_metrics,
)


def test_source_leakage_metrics_capture_off_support_energy():
    x_true = np.array(
        [
            [1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ]
    )
    x_leaky = np.array(
        [
            [1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j],
            [0.6 + 0.0j, 0.6 + 0.0j, 0.6 + 0.0j],
            [0.2 + 0.0j, 0.2 + 0.0j, 0.2 + 0.0j],
        ]
    )

    metrics = source_leakage_metrics(x_true, x_leaky, tol=1e-12, row_floor_ratio=0.1)

    assert_allclose(metrics["active_energy_ratio"], 3.0 / 4.2)
    assert_allclose(metrics["inactive_energy_ratio"], 1.2 / 4.2)
    assert_allclose(metrics["inactive_to_active_energy_ratio"], 1.2 / 3.0)
    assert metrics["inactive_rows_above_floor"] == 2
    assert_allclose(metrics["inactive_row_fraction"], 1.0)
    assert_allclose(metrics["max_inactive_row_ratio"], 0.6)


def test_format_source_leakage_summary_reports_expected_values():
    x_true = np.array(
        [
            [1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ]
    )
    x_pred = np.array(
        [
            [1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j],
            [0.6 + 0.0j, 0.6 + 0.0j, 0.6 + 0.0j],
            [0.2 + 0.0j, 0.2 + 0.0j, 0.2 + 0.0j],
        ]
    )

    summary = format_source_leakage_summary(
        source_leakage_metrics(x_true, x_pred, tol=1e-12, row_floor_ratio=0.1)
    )

    assert (
        summary
        == "on=71.4%, off=28.6%\noff-rows>10%=2, max off/true=60.0%"
    )
