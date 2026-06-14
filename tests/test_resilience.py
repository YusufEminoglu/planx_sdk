# -*- coding: utf-8 -*-
"""Tests for the resilience submodule."""

import numpy as np
import pytest

from planx.resilience import (
    pluvial_flood_susceptibility,
    simulate_seismic_debris,
    social_vulnerability_index,
    urban_heat_comfort_risk,
)


def test_simulate_seismic_debris():
    # 4 buildings
    areas = np.array([100.0, 150.0, 200.0, 300.0])
    floors = np.array([2.0, 5.0, 10.0, 1.0])
    years = np.array([1980, 1995, 2010, 2025])

    # 1. Run with Mw = 7.0 (baseline)
    # 1980: base_p = 0.85
    # 1995: base_p = 0.60
    # 2010: base_p = 0.25
    # 2025: base_p = 0.05
    probs, collapsed, radii, volumes = simulate_seismic_debris(
        areas, floors, years, magnitude=7.0, seed=42
    )

    np.testing.assert_allclose(probs, [0.85, 0.60, 0.25, 0.05])
    assert len(collapsed) == 4
    assert np.all((collapsed == 0) | (collapsed == 1))

    # Radii should be: H * debris_factor if collapsed, else 0
    # H = floors * 3.0
    # debris_factor = 0.4
    # For collapsed: H * 0.4 = floors * 1.2
    for i in range(4):
        if collapsed[i] == 1:
            assert np.isclose(radii[i], floors[i] * 3.0 * 0.4)
            assert np.isclose(volumes[i], areas[i] * floors[i] * 3.0 * 0.3)
        else:
            assert radii[i] == 0.0
            assert volumes[i] == 0.0

    # 2. Check dimension mismatch error
    with pytest.raises(ValueError, match="must have identical length"):
        simulate_seismic_debris(areas[:-1], floors, years, magnitude=7.0)


def test_pluvial_flood_susceptibility():
    dem = np.array([[10.0, 12.0, 15.0], [8.0, 9.0, 11.0], [5.0, 7.0, 8.0]])
    scores, classes = pluvial_flood_susceptibility(dem, cell_size=10.0, neighborhood_radius=15.0)

    assert scores.shape == (3, 3)
    assert len(classes) == 3
    assert len(classes[0]) == 3
    # Low elevations should have higher susceptibility scores
    assert scores[2, 0] > scores[0, 2]


def test_social_vulnerability_index():
    indicators = {
        "elderly": np.array([10.0, 50.0, 100.0]),
        "low_income": np.array([200.0, 100.0, 50.0]),
    }
    weights = {"elderly": 0.5, "low_income": 0.5}

    scores, classes = social_vulnerability_index(indicators, weights)

    np.testing.assert_allclose(scores, [50.0, 38.888889, 50.0], rtol=1e-5)
    assert classes == ["Moderate", "Moderate", "Moderate"]


def test_urban_heat_comfort_risk():
    imp = np.array([[0.8, 0.2], [0.5, 0.1]])
    bld = np.array([[0.6, 0.1], [0.4, 0.05]])
    grn = np.array([[0.1, 0.8], [0.3, 0.9]])
    dst = np.array([[300.0, 50.0], [200.0, 20.0]])
    vuln = np.array([[2, 0], [1, 0]])

    scores, classes = urban_heat_comfort_risk(imp, bld, grn, dst, vuln, cooling_distance=400.0)

    assert scores.shape == (2, 2)
    assert len(classes) == 2
    assert len(classes[0]) == 2
    assert scores[0, 0] > scores[1, 1]
