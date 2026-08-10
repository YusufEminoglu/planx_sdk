"""Tests for the reusable PlanX quality and decision utilities."""

import numpy as np
import pytest

from planx.quality import (
    array_summary,
    bootstrap_mean_ci,
    classify_values,
    gini_coefficient,
    normalize_weights,
    pareto_front,
    quantile_breaks,
    rank_values,
    top_k_indices,
    weighted_quantile,
    zscore,
)


def test_array_summary_excludes_missing_and_adds_weighted_mean():
    summary = array_summary([1.0, 2.0, np.nan, 4.0], weights=[1.0, 1.0, 3.0, 2.0])
    assert summary["count"] == 4
    assert summary["valid_count"] == 3
    assert summary["missing_count"] == 1
    assert summary["mean"] == pytest.approx(7.0 / 3.0)
    assert summary["weighted_mean"] == pytest.approx(11.0 / 4.0)


def test_weighted_quantile_and_gini():
    assert weighted_quantile([1.0, 2.0, 4.0], [0.0, 0.5, 1.0]).tolist() == [1.0, 2.0, 4.0]
    assert weighted_quantile([1.0, 2.0, 4.0], [0.5], [1.0, 1.0, 2.0])[0] == pytest.approx(2.0)
    assert gini_coefficient([0.0, 1.0]) == pytest.approx(0.5)
    assert gini_coefficient([0.0, 0.0, 0.0]) == 0.0
    assert gini_coefficient([1.0, 2.0], [1.0, 3.0]) == pytest.approx(6.0 / 56.0)


def test_weight_normalization_and_break_classification():
    assert np.allclose(normalize_weights([1.0, 2.0, 1.0]), [0.25, 0.5, 0.25])
    breaks = quantile_breaks(np.arange(1.0, 11.0), n_classes=4)
    assert np.allclose(breaks, [3.25, 5.5, 7.75])
    assert np.array_equal(classify_values([1.0, 3.25, 5.5, 10.0, np.nan], breaks), [1, 2, 3, 4, -1])


def test_rank_is_stable_and_averages_ties():
    ranked = rank_values([10.0, 20.0, 20.0, 5.0, np.nan])
    assert np.allclose(ranked, [3.0, 1.5, 1.5, 4.0, 0.0], equal_nan=False)
    assert np.array_equal(rank_values([1.0, 3.0, 2.0], descending=False), [1.0, 3.0, 2.0])


def test_bootstrap_ci_is_reproducible():
    values = np.arange(1.0, 11.0)
    first = bootstrap_mean_ci(values, n_boot=400, seed=7)
    second = bootstrap_mean_ci(values, n_boot=400, seed=7)
    assert first == second
    assert first[0] == pytest.approx(5.5)
    assert first[1] <= first[0] <= first[2]


def test_zscore_and_top_k_handle_missing_values():
    scores = zscore([1.0, 2.0, 3.0, np.nan])
    assert np.allclose(scores[:3], [-np.sqrt(1.5), 0.0, np.sqrt(1.5)])
    assert np.isnan(scores[3])
    assert np.array_equal(top_k_indices([1.0, np.nan, 3.0, 2.0, 3.0], 3), [2, 4, 3])
    assert np.array_equal(top_k_indices([1.0, 4.0, 2.0], 2, largest=False), [0, 2])


def test_pareto_front_supports_mixed_objectives_and_nodata():
    values = np.array([[1.0, 4.0], [2.0, 3.0], [1.0, 5.0], [np.nan, 2.0]])
    assert np.array_equal(pareto_front(values), [False, True, True, False])
    mixed = np.array([[2.0, 5.0], [3.0, 4.0], [1.0, 6.0]])
    assert np.array_equal(pareto_front(mixed, maximize=[True, False]), [False, True, False])


@pytest.mark.parametrize(
    "call",
    [
        lambda: normalize_weights([0.0, 0.0]),
        lambda: weighted_quantile([1.0, 2.0], [1.2]),
        lambda: quantile_breaks([1.0], 1),
        lambda: classify_values([1.0], [2.0, 1.0]),
        lambda: bootstrap_mean_ci([1.0, 2.0], n_boot=10),
        lambda: zscore([np.nan]),
        lambda: top_k_indices([1.0], 0),
        lambda: pareto_front(np.array([1.0, 2.0])),
    ],
)
def test_quality_utilities_reject_invalid_input(call):
    with pytest.raises(ValueError):
        call()
