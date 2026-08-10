# -*- coding: utf-8 -*-
"""Reusable data-quality and decision-support utilities.

These small, dependency-light helpers are shared by headless PlanX engines
when they need consistent summaries, weighting, classification, and ranking
behaviour. Missing values are preserved where possible and invalid input is
rejected with an actionable ``ValueError``.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "array_summary",
    "weighted_quantile",
    "gini_coefficient",
    "normalize_weights",
    "quantile_breaks",
    "classify_values",
    "rank_values",
    "bootstrap_mean_ci",
    "zscore",
    "top_k_indices",
    "pareto_front",
]


def _finite_values(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0:
        raise ValueError("values must contain at least one item")
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("values must contain at least one finite item")
    return finite


def _values_and_weights(values, weights=None) -> tuple[np.ndarray, np.ndarray | None]:
    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0:
        raise ValueError("values must contain at least one item")
    if weights is None:
        finite = np.isfinite(array)
        if not np.any(finite):
            raise ValueError("values must contain at least one finite item")
        return array[finite], None
    weight_array = np.asarray(weights, dtype=float).ravel()
    if weight_array.shape != array.shape:
        raise ValueError("weights must have the same number of items as values")
    finite = np.isfinite(array) & np.isfinite(weight_array)
    if np.any(weight_array[finite] < 0.0):
        raise ValueError("weights must be non-negative")
    if not np.any(finite & (weight_array > 0.0)):
        raise ValueError("weights must contain at least one positive finite item")
    return array[finite], weight_array[finite]


def array_summary(values, weights=None) -> dict[str, float | int]:
    """Return robust descriptive statistics for a numeric array.

    Non-finite values are excluded from statistics and reported as
    ``missing_count``. When weights are supplied, ``weighted_mean`` is added
    while the unweighted statistics remain available for diagnostics.
    """

    raw = np.asarray(values, dtype=float).ravel()
    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        raise ValueError("values must contain at least one finite item")
    result: dict[str, float | int] = {
        "count": int(raw.size),
        "valid_count": int(finite.size),
        "missing_count": int(raw.size - finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "p25": float(np.quantile(finite, 0.25)),
        "p75": float(np.quantile(finite, 0.75)),
        "p95": float(np.quantile(finite, 0.95)),
        "sum": float(np.sum(finite)),
    }
    if weights is not None:
        clean, clean_weights = _values_and_weights(values, weights)
        assert clean_weights is not None
        result["weighted_mean"] = float(np.average(clean, weights=clean_weights))
    return result


def weighted_quantile(values, quantiles, weights=None) -> np.ndarray:
    """Compute weighted quantiles with deterministic linear interpolation."""

    clean, clean_weights = _values_and_weights(values, weights)
    q = np.asarray(quantiles, dtype=float)
    if np.any(~np.isfinite(q)) or np.any((q < 0.0) | (q > 1.0)):
        raise ValueError("quantiles must be finite values between 0 and 1")
    if clean_weights is None:
        result = np.quantile(clean, q)
    else:
        positive = clean_weights > 0.0
        clean = clean[positive]
        clean_weights = clean_weights[positive]
        order = np.argsort(clean, kind="mergesort")
        sorted_values = clean[order]
        sorted_weights = clean_weights[order]
        cumulative = np.cumsum(sorted_weights) / np.sum(sorted_weights)
        result = np.interp(q, cumulative, sorted_values, left=sorted_values[0])
    return np.asarray(result, dtype=float).reshape(q.shape)


def gini_coefficient(values, weights=None) -> float:
    """Return the (optionally weighted) Gini inequality coefficient."""

    clean, clean_weights = _values_and_weights(values, weights)
    if np.any(clean < 0.0):
        raise ValueError("Gini coefficient requires non-negative values")
    if clean_weights is None:
        clean_weights = np.ones(clean.size, dtype=float)
    order = np.argsort(clean, kind="mergesort")
    clean = clean[order]
    clean_weights = clean_weights[order]
    total_value = float(np.sum(clean * clean_weights))
    if total_value == 0.0:
        return 0.0
    total_weight = float(np.sum(clean_weights))
    cumulative_weight = np.cumsum(clean_weights)
    cumulative_value = np.cumsum(clean * clean_weights)
    previous_weight = np.r_[0.0, cumulative_weight[:-1]] / total_weight
    lorenz = cumulative_value / total_value
    previous_lorenz = np.r_[0.0, lorenz[:-1]]
    area = float(
        np.sum(
            (previous_lorenz + lorenz)
            * 0.5
            * (cumulative_weight / total_weight - previous_weight)
        )
    )
    return float(np.clip(1.0 - 2.0 * area, 0.0, 1.0))


def normalize_weights(weights) -> np.ndarray:
    """Validate and normalize a non-negative weight vector to sum to one."""

    array = np.asarray(weights, dtype=float).ravel()
    if array.size == 0 or np.any(~np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("weights must be a non-empty finite non-negative vector")
    total = float(np.sum(array))
    if total <= 0.0:
        raise ValueError("weights must contain a positive total")
    return array / total


def quantile_breaks(values, n_classes: int = 5) -> np.ndarray:
    """Return unique internal quantile thresholds for classification."""

    if not isinstance(n_classes, (int, np.integer)) or n_classes < 2:
        raise ValueError("n_classes must be an integer greater than or equal to 2")
    clean = _finite_values(values)
    breaks = np.quantile(clean, np.linspace(0.0, 1.0, int(n_classes) + 1)[1:-1])
    return np.unique(np.asarray(breaks, dtype=float))


def classify_values(values, breaks: Sequence[float], nodata: int = -1) -> np.ndarray:
    """Classify values into one-based bins, preserving non-finite values as nodata."""

    array = np.asarray(values, dtype=float)
    threshold = np.asarray(breaks, dtype=float).ravel()
    if np.any(~np.isfinite(threshold)) or np.any(np.diff(threshold) <= 0.0):
        raise ValueError("breaks must be finite and strictly increasing")
    result = np.full(array.shape, nodata, dtype=int)
    finite = np.isfinite(array)
    result[finite] = np.digitize(array[finite], threshold, right=False) + 1
    return result


def rank_values(values, descending: bool = True) -> np.ndarray:
    """Return stable one-based average ranks; non-finite values receive zero."""

    original = np.asarray(values, dtype=float)
    array = original.ravel()
    result = np.zeros(array.size, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(array))
    if finite_indices.size == 0:
        return result.reshape(original.shape)
    sort_values = -array[finite_indices] if descending else array[finite_indices]
    order = np.argsort(sort_values, kind="mergesort")
    ordered_indices = finite_indices[order]
    ordered_values = array[ordered_indices]
    start = 0
    while start < ordered_values.size:
        end = start + 1
        while end < ordered_values.size and ordered_values[end] == ordered_values[start]:
            end += 1
        result[ordered_indices[start:end]] = (start + 1 + end) / 2.0
        start = end
    return result.reshape(original.shape)


def bootstrap_mean_ci(
    values,
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int | None = 42,
) -> tuple[float, float, float]:
    """Return ``(mean, lower, upper)`` from a reproducible bootstrap CI."""

    clean = _finite_values(values)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if not isinstance(n_boot, (int, np.integer)) or n_boot < 100:
        raise ValueError("n_boot must be an integer of at least 100")
    rng = np.random.default_rng(seed)
    means = np.empty(int(n_boot), dtype=float)
    for start in range(0, int(n_boot), 256):
        stop = min(start + 256, int(n_boot))
        sample = rng.integers(0, clean.size, size=(stop - start, clean.size))
        means[start:stop] = np.mean(clean[sample], axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha])
    return float(np.mean(clean)), float(lower), float(upper)


def zscore(values, ddof: int = 0) -> np.ndarray:
    """Standardize finite values while preserving non-finite entries."""

    original = np.asarray(values, dtype=float)
    array = original.ravel()
    finite = np.isfinite(array)
    finite_count = int(np.count_nonzero(finite))
    if ddof < 0 or ddof >= finite_count:
        raise ValueError("ddof must be non-negative and smaller than the finite count")
    if finite_count == 0:
        raise ValueError("values must contain at least one finite item")
    mean = float(np.mean(array[finite]))
    scale = float(np.std(array[finite], ddof=ddof))
    result = np.full(array.shape, np.nan, dtype=float)
    result[finite] = 0.0 if scale == 0.0 else (array[finite] - mean) / scale
    return result.reshape(original.shape)


def top_k_indices(values, k: int, largest: bool = True) -> np.ndarray:
    """Return stable indices of the ``k`` largest or smallest finite values."""

    array = np.asarray(values, dtype=float).ravel()
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    finite = np.flatnonzero(np.isfinite(array))
    if finite.size == 0:
        return np.empty(0, dtype=int)
    order_values = -array[finite] if largest else array[finite]
    order = np.argsort(order_values, kind="mergesort")
    return finite[order[: int(k)]]


def pareto_front(values, maximize=True) -> np.ndarray:
    """Return a mask for non-dominated rows in a multi-objective matrix."""

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("values must be a non-empty two-dimensional matrix")
    directions = np.asarray(maximize, dtype=bool)
    if directions.ndim == 0:
        directions = np.full(matrix.shape[1], bool(directions))
    if directions.shape != (matrix.shape[1],):
        raise ValueError("maximize must be a scalar or one flag per objective")
    valid = np.all(np.isfinite(matrix), axis=1)
    oriented = np.where(directions, matrix, -matrix)
    front = np.zeros(matrix.shape[0], dtype=bool)
    valid_indices = np.flatnonzero(valid)
    for i in valid_indices:
        dominates = np.all(oriented[valid_indices] >= oriented[i], axis=1) & np.any(
            oriented[valid_indices] > oriented[i], axis=1
        )
        front[i] = not bool(np.any(dominates))
    return front
