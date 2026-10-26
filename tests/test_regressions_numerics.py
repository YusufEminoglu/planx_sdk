# -*- coding: utf-8 -*-
"""Regression tests for numerical defects fixed against the textbook formulas,
plus coverage of previously untested edge paths in the engine modules."""

from __future__ import annotations

import itertools
import math
import zipfile

import numpy as np
import pytest

from planx.engine import population, report, scenario, syntax, transit
from planx.generative.parcel_generator import generate_optimized_parcels
from planx.geostats import (
    calculate_general_g,
    calculate_global_geary,
    calculate_global_moran,
    calculate_ols,
    calculate_sde,
    mann_kendall_test,
)
from planx.suitability.tod_optimizer import evaluate_tod_potential


# ---------------------------------------------------------------------------
# Global Moran's I / Geary's C: S1 with asymmetric (k-nearest) weights
# ---------------------------------------------------------------------------
def _knn_fixture(n: int = 12, k: int = 2, seed: int = 1):
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 2))
    y = rng.random(n)
    d = ((pts[:, None] - pts[None]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    nb = {i: [int(j) for j in np.argsort(d[i])[:k]] for i in range(n)}
    w = {i: [1.0] * k for i in range(n)}
    W = np.zeros((n, n))
    for i, js in nb.items():
        W[i, js] = 1.0
    return y, nb, w, W


def _cliff_ord_reference(y: np.ndarray, W: np.ndarray) -> tuple[float, float]:
    """Dense-matrix Cliff & Ord randomisation variances of Moran's I and Geary's C."""
    n = len(y)
    z = y - y.mean()
    s0 = W.sum()
    s1 = 0.5 * ((W + W.T) ** 2).sum()
    s2 = ((W.sum(1) + W.sum(0)) ** 2).sum()
    b2 = n * (z**4).sum() / (z @ z) ** 2
    e_i = -1.0 / (n - 1)
    var_i = (
        n * ((n * n - 3 * n + 3) * s1 - n * s2 + 3 * s0**2)
        - b2 * ((n * n - n) * s1 - 2 * n * s2 + 6 * s0**2)
    ) / ((n - 1) * (n - 2) * (n - 3) * s0**2) - e_i**2
    n2 = n * n
    var_c = (
        (n - 1) * s1 * (n2 - 3 * n + 3 - (n - 1) * b2)
        - 0.25 * (n - 1) * s2 * (n2 + 3 * n - 6 - (n2 - n + 2) * b2)
        + s0**2 * (n2 - 3 - (n - 1) ** 2 * b2)
    ) / (n * (n - 2) * (n - 3) * s0**2)
    return float(var_i), float(var_c)


def test_global_moran_variance_asymmetric_knn_weights():
    y, nb, w, W = _knn_fixture()
    ref_var_i, _ = _cliff_ord_reference(y, W)
    _, _, variance, _, _ = calculate_global_moran(y, nb, w, list(range(len(y))))
    # Before the fix S1 dropped one direction of every one-way pair: 0.0556.
    assert variance == pytest.approx(ref_var_i, rel=1e-10)


def test_global_geary_variance_asymmetric_knn_weights():
    y, nb, w, W = _knn_fixture()
    _, ref_var_c = _cliff_ord_reference(y, W)
    _, _, variance, _, _ = calculate_global_geary(y, nb, w, list(range(len(y))))
    assert variance == pytest.approx(ref_var_c, rel=1e-10)


# ---------------------------------------------------------------------------
# General G: randomisation variance vs the exact permutation distribution
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("symmetric", [True, False])
def test_general_g_variance_matches_exhaustive_permutations(symmetric):
    rng = np.random.default_rng(3)
    n = 6
    x = rng.random(n) * 10.0 + 1.0
    W = (rng.random((n, n)) < 0.5).astype(float) * rng.random((n, n))
    np.fill_diagonal(W, 0.0)
    if symmetric:
        W = np.maximum(W, W.T)
    nb = {i: [j for j in range(n) if W[i, j]] for i in range(n)}
    w = {i: [W[i, j] for j in nb[i]] for i in range(n)}

    g, expected, variance, _, _ = calculate_general_g(x, nb, w, list(range(n)))

    den = x.sum() ** 2 - (x**2).sum()
    sims = [
        (W * np.outer(x[list(p)], x[list(p)])).sum() / den for p in itertools.permutations(range(n))
    ]
    assert expected == pytest.approx(np.mean(sims), rel=1e-12)
    # Before the fix the variance was an order of magnitude too large.
    assert variance == pytest.approx(np.var(sims), rel=1e-9)


# ---------------------------------------------------------------------------
# Standard deviational ellipse orientation
# ---------------------------------------------------------------------------
def test_sde_diagonal_cloud_is_oriented_along_the_diagonal():
    t = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    jitter = np.array([0.0, 0.1, 0.0, -0.1, 0.0])
    x, y = t + jitter, t - jitter
    _, _, angle, semi_major, semi_minor = calculate_sde(x, y)
    # Points along y = x: the major axis is at ~45 deg (it was 135 deg).
    assert math.degrees(angle) == pytest.approx(45.0, abs=1.0)
    assert semi_major == pytest.approx(math.sqrt(np.sum(x**2 + y**2) / 5 - semi_minor**2))
    assert semi_minor < 0.1 * semi_major

    _, _, angle_neg, _, _ = calculate_sde(-x, y)
    assert math.degrees(angle_neg) == pytest.approx(-45.0, abs=1.0)


# ---------------------------------------------------------------------------
# OLS residual Moran's I with non-row-standardised weights
# ---------------------------------------------------------------------------
def test_ols_residual_moran_uses_n_over_s0_for_binary_weights():
    n = 8
    x = np.arange(n, dtype=float)
    y = 2.0 * x + 0.5 * np.array([1, -1, 1, -1, 1, -1, 1, -1.0])
    y += np.array([0, 0, 0.3, 0, 0, 0.2, 0, 0])
    nb = {i: [j for j in (i - 1, i + 1) if 0 <= j < n] for i in range(n)}
    w = {i: [1.0] * len(v) for i, v in nb.items()}
    res = calculate_ols(y, x[:, None], nb, w, list(range(n)), ["x"])

    e = res["residuals"]
    W = np.zeros((n, n))
    for i, js in nb.items():
        W[i, js] = 1.0
    expected = n / W.sum() * (e @ W @ e) / (e @ e)
    # Without the n / S0 factor the statistic was -1.73 (outside [-1, 1]).
    assert res["residuals_moran"] == pytest.approx(expected)
    assert -1.0 <= res["residuals_moran"] <= 1.0


# ---------------------------------------------------------------------------
# Mann-Kendall / Sen slope
# ---------------------------------------------------------------------------
def test_sen_slope_counts_tied_pairs_as_zero_slopes():
    # 6 of the 10 pairwise slopes are 0, so the Sen median is 0 (was 0.5).
    _, _, slope = mann_kendall_test([1.0, 1.0, 1.0, 1.0, 2.0])
    assert slope == 0.0


def test_sen_slope_even_count_averages_middle_slopes():
    # Slopes 1, 1.5, 2, 7/3, 3, 4 -> median (2 + 7/3) / 2 (was 7/3).
    _, _, slope = mann_kendall_test([0.0, 1.0, 3.0, 7.0])
    assert slope == pytest.approx((2.0 + 7.0 / 3.0) / 2.0)


# ---------------------------------------------------------------------------
# GTFS: first_service_day honours removal exceptions
# ---------------------------------------------------------------------------
def test_first_service_day_skips_a_removed_start_date():
    feed = {
        "calendar": {
            "WK": {
                "days": (True,) * 5 + (False, False),
                "start": "20260101",
                "end": "20261231",
            }
        },
        "calendar_dates": [("WK", "20260101", 2)],
    }
    day = transit.first_service_day(feed)
    assert day == "20260102"
    assert transit.active_services(feed, day) == {"WK"}


def test_first_service_day_fallbacks_and_errors():
    only_added = {"calendar": {}, "calendar_dates": [("X", "20260310", 1), ("X", "20260305", 1)]}
    assert transit.first_service_day(only_added) == "20260305"
    bad_start = {
        "calendar": {"Z": {"days": (True,) * 7, "start": "bad", "end": "x"}},
        "calendar_dates": [],
    }
    with pytest.raises(ValueError):
        transit.first_service_day(bad_start)


# ---------------------------------------------------------------------------
# Report: non-finite metrics from from_json
# ---------------------------------------------------------------------------
def test_compare_report_renders_non_finite_metrics():
    a = scenario.from_json('{"name": "A", "metrics": {"pop": NaN, "x": Infinity}}')
    b = scenario.from_json('{"name": "B", "metrics": {"pop": 5, "x": 1}}')
    html = report.build_compare_html("t", scenario.compare(a, b), generated="now")
    assert '<td class="num">nan</td>' in html
    assert '<td class="num">inf</td>' in html
    assert report._fmt_metric(float("-inf")) == "-inf"


def test_build_rank_html_sections():
    snaps = [
        scenario.snapshot("Base", {"access_mean": 40.0, "pop": 10.0}),
        scenario.snapshot("Plan <B>", {"access_mean": 70.0, "pop": 10.0}),
    ]
    result = scenario.rank(snaps)
    html = report.build_rank_html("Rank", result, weights_note="w note", generated="g")
    assert "Scenario Scoreboard" in html
    assert "Plan &lt;B&gt;" in html
    assert "w note" in html
    assert "Skipped metrics:" in html

    no_skip = dict(result, skipped=[])
    assert "Skipped metrics: none" in report.build_rank_html("R", no_skip)


# ---------------------------------------------------------------------------
# TOD land-use entropy
# ---------------------------------------------------------------------------
def test_tod_entropy_is_scale_invariant_in_the_shares():
    def entropy(mix):
        res = evaluate_tod_potential([{"id": "s", "landuse_shares": mix}])
        return res.station_scores[0].landuse_diversity_entropy

    base = entropy({"r": 0.5, "c": 0.3, "v": 0.2})
    # Percentages used to give a negative "entropy" clamped to 0.
    assert entropy({"r": 50, "c": 30, "v": 20}) == pytest.approx(base)
    assert entropy({"r": 5000.0, "c": 3000.0, "v": 2000.0}) == pytest.approx(base)
    assert entropy({"r": 0.0, "c": 0.0}) == 0.0
    assert entropy({"r": 1.0}) == 0.0


def test_tod_typology_bands():
    stations = [
        {
            "id": "low",
            "peak_trips_per_hour": 1,
            "intersection_density_km2": 5,
            "landuse_shares": {"r": 1.0},
        },
        {
            "id": "mid",
            "peak_trips_per_hour": 12,
            "intersection_density_km2": 60,
            "landuse_shares": {"r": 0.9, "c": 0.1},
        },
    ]
    res = evaluate_tod_potential(stations)
    assert res.station_scores[0].tod_typology == "Suburban Node"
    assert res.station_scores[0].recommended_far_density == 1.2
    assert res.station_scores[1].tod_typology == "Neighborhood Station"
    assert res.high_potential_tod_count == 0
    assert evaluate_tod_potential([]).mean_tod_index == 0.0


# ---------------------------------------------------------------------------
# Coverage: parcel generator, space syntax radii, population, GTFS reader
# ---------------------------------------------------------------------------
def test_parcel_generator_tall_block_and_degenerate_input():
    tall = [(0.0, 0.0), (40.0, 0.0), (40.0, 120.0), (0.0, 120.0)]
    layout = generate_optimized_parcels(tall, target_parcel_area_m2=400.0)
    assert layout.total_parcels == 9
    assert all(p.coordinates[0][0] == 0.0 for p in layout.parcels)
    assert sum(p.area_m2 for p in layout.parcels) == pytest.approx(40.0 * 120.0)
    empty = generate_optimized_parcels([(0.0, 0.0), (1.0, 1.0)])
    assert empty.total_parcels == 0 and empty.parcels == []


def test_parse_radii_variants_and_labels():
    assert syntax.parse_radii("400; ;Global, rn") == [400.0, None, None]
    assert syntax.parse_radii("  ") == [None]
    with pytest.raises(ValueError):
        syntax.parse_radii("400, -5")
    assert syntax.radius_label(None) == "n"
    assert syntax.radius_label(800.0) == "800"


def test_population_validation_paths():
    with pytest.raises(ValueError):
        population.leslie_matrix(np.array([0.9]), np.array([0.1, 0.2]))
    with pytest.raises(ValueError):
        population.cohort_projection(np.ones(3), np.ones(2), np.zeros(2))
    with pytest.raises(ValueError):
        population.cohort_projection(np.ones(2), np.ones(2), np.zeros(2), migration=np.ones(3))
    with pytest.raises(ValueError):
        population.housing_needs(100.0, 0.0, 10.0)
    with pytest.raises(ValueError):
        population.residential_capacity(np.ones(2), np.ones(3))
    with pytest.raises(ValueError):
        population.residential_capacity(np.ones(2), np.ones(2), unit_size=0.0)
    with pytest.raises(ValueError):
        population.residential_capacity(np.ones(2), np.ones(2), existing_floor=np.ones(3))
    assert population.allocate_growth(5, np.array([])).size == 0
    assert population.allocate_growth(0, np.ones(3)).tolist() == [0, 0, 0]
    assert population.allocate_growth(4, np.zeros(4)).tolist() == [1, 1, 1, 1]


def _zip(path, files):
    with zipfile.ZipFile(path, "w") as zf:
        for name, text in files.items():
            zf.writestr(name, text)
    return str(path)


_BASE_GTFS = {
    "stops.txt": "stop_id,stop_name,stop_lat,stop_lon\nA,,0,0\n,skip,0,0\nB,Stop B,0,0.01\n",
    "routes.txt": "route_id,route_short_name,route_long_name,route_type\nR1,1,L,bus\n,x,y,3\n",
    "trips.txt": "trip_id,route_id,service_id\nT1,R1,S\n,R1,S\n",
    "stop_times.txt": (
        "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
        "T1,08:00:00,08:00:00,A,1\nT1,,,B,2\nT1,08:10:00,,B,3\nT9,08:00:00,,A,1\n"
    ),
    "calendar_dates.txt": "service_id,date,exception_type\nS,20260105,1\nS,20260106,x\n",
}


def test_read_gtfs_tolerant_paths(tmp_path):
    feed = transit.read_gtfs(_zip(tmp_path / "ok.zip", _BASE_GTFS))
    assert feed["stop_ids"] == ["A", "B"]
    assert feed["stop_names"][0] == "A"  # empty name falls back to the id
    assert feed["routes"]["R1"]["type"] == -1  # non-numeric route_type
    assert list(feed["trips"]) == ["T1"]
    assert feed["stop_times"]["T1"] == [(28800, 28800, 0), (29400, 29400, 1)]
    assert feed["calendar"] == {}
    assert feed["calendar_dates"] == [("S", "20260105", 1)]
    assert transit.first_service_day(feed) == "20260105"


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"calendar_dates.txt": None}, "neither calendar.txt"),
        ({"stops.txt": "stop_id,stop_lat,stop_lon\n"}, "no stops"),
        ({"stops.txt": "stop_id,stop_lat,stop_lon\nA,north,0\n"}, "non-numeric"),
        (
            {
                "stop_times.txt": "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
                "T1,8h00,,A,1\n"
            },
            "trip 'T1'",
        ),
    ],
)
def test_read_gtfs_errors(tmp_path, patch, message):
    files = dict(_BASE_GTFS)
    for name, text in patch.items():
        if text is None:
            files.pop(name)
        else:
            files[name] = text
    with pytest.raises(ValueError, match=message):
        transit.read_gtfs(_zip(tmp_path / "bad.zip", files))


def test_read_gtfs_not_a_zip_and_bad_inputs(tmp_path):
    bad = tmp_path / "x.zip"
    bad.write_text("not a zip")
    with pytest.raises(ValueError, match="could not open"):
        transit.read_gtfs(str(bad))
    with pytest.raises(ValueError):
        transit.parse_time("08:61:00")
    with pytest.raises(ValueError):
        transit.active_services({"calendar": {}, "calendar_dates": []}, "2026-1-1")
    feed = transit.read_gtfs(_zip(tmp_path / "ok.zip", _BASE_GTFS))
    with pytest.raises(ValueError):
        transit.stop_frequencies(feed, "20260105", window=(10.0, 5.0))
