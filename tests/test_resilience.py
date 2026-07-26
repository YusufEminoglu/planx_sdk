# -*- coding: utf-8 -*-
"""Tests for the resilience submodule."""

import numpy as np
import pytest

from planx.resilience import (
    calculate_grid_sky_view_factor,
    calculate_solar_access,
    classify_local_climate_zones,
    coastal_flood_inundation,
    coastal_surge_inundation,
    debris_clearance_routing,
    equity_adjusted_priority,
    evacuation_route_optimization,
    identify_critical_bottlenecks,
    infrastructure_service_loss,
    landslide_susceptibility,
    multi_hazard_composite,
    network_criticality_index,
    optimize_canopy_placement,
    pluvial_flood_susceptibility,
    prioritize_debris_clearance,
    simulate_interdependent_infrastructure_cascade,
    simulate_network_disruption,
    simulate_seismic_debris,
    social_vulnerability_index,
    socio_economic_flood_risk,
    tree_canopy_microclimate_cooling,
    urban_heat_comfort_risk,
    urban_heat_island_intensity,
    wildfire_risk_index,
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


def test_multi_hazard_composite():
    heat = np.array([80.0, 20.0, 50.0])
    flood = np.array([40.0, 10.0, np.nan])

    hazards = {"heat": heat, "flood": flood}
    weights = {"heat": 0.6, "flood": 0.4}

    scores, classes, dominant, diversity, drivers = multi_hazard_composite(hazards, weights)

    assert scores.shape == (3,)
    assert diversity.shape == (3,)
    assert len(classes) == 3
    assert len(dominant) == 3
    assert len(drivers) == 3

    # Check score calculations
    # Index 0: (80*0.6 + 40*0.4) / 1.0 = 48 + 16 = 64
    assert np.isclose(scores[0], 64.0)
    # Index 2: only heat is valid, so score = 50.0
    assert np.isclose(scores[2], 50.0)
    assert dominant[0] == "heat"
    assert "heat" in drivers[0]
    assert "flood" in drivers[0]
    assert drivers[2] == ["heat"]


def test_equity_adjusted_priority():
    hazard = np.array([[40.0, 60.0], [20.0, 80.0]])
    svi = np.array([[10.0, 90.0], [50.0, 30.0]])

    scores, raw, factors, classes = equity_adjusted_priority(hazard, svi, equity_weight=0.5)

    assert scores.shape == (2, 2)
    assert raw.shape == (2, 2)
    assert factors.shape == (2, 2)
    assert len(classes) == 2
    assert len(classes[0]) == 2

    # Index (0, 0): factor = 1 + 0.5*0.1 = 1.05
    # raw = 40.0 * 1.05 = 42.0
    # score = 100 * 42 / 150 = 28.0
    assert np.isclose(factors[0, 0], 1.05)
    assert np.isclose(raw[0, 0], 42.0)
    assert np.isclose(scores[0, 0], 28.0)


def test_simulate_network_disruption():
    # 0 - 1 - 2
    indptr = np.array([0, 1, 3, 4], dtype=np.int64)
    adj = np.array([1, 0, 2, 1], dtype=np.int64)
    weights = np.array([1.5, 1.5, 2.5, 2.5], dtype=np.float64)

    # 1. Block an edge
    w_disrupt = simulate_network_disruption(indptr, adj, weights, n=3, blocked_edges=[0])
    assert w_disrupt[0] == np.inf
    assert w_disrupt[1] == 1.5

    # 2. Block a node (node 1)
    w_disrupt2 = simulate_network_disruption(indptr, adj, weights, n=3, blocked_nodes=[1])
    # Node 1 outgoing: edges 1 (1->0) and 2 (1->2) should be inf
    # Node 1 incoming: edges 0 (0->1) and 3 (2->1) should be inf
    np.testing.assert_allclose(w_disrupt2, [np.inf, np.inf, np.inf, np.inf])

    # 3. Invalid inputs
    with pytest.raises(ValueError, match="Blocked edge indices"):
        simulate_network_disruption(indptr, adj, weights, n=3, blocked_edges=[10])
    with pytest.raises(ValueError, match="Blocked node indices"):
        simulate_network_disruption(indptr, adj, weights, n=3, blocked_nodes=[5])


def test_infrastructure_service_loss():
    # 2 origins, 2 destinations
    dists_pre = np.array([[10.0, 20.0], [15.0, 30.0]])
    dists_post = np.array([[10.0, np.inf], [np.inf, np.inf]])  # Origin 1 is isolated

    results = infrastructure_service_loss(dists_pre, dists_post, demands=np.array([100.0, 50.0]))
    assert np.isclose(results["isolation_rate"], 50.0 / 150.0)
    assert np.isclose(results["pop_isolated"], 50.0)
    assert np.isclose(results["mean_delay"], 0.0)

    # With delay
    dists_post_delay = np.array([[15.0, 20.0], [np.inf, np.inf]])
    results2 = infrastructure_service_loss(
        dists_pre, dists_post_delay, demands=np.array([100.0, 50.0])
    )
    assert np.isclose(results2["mean_delay"], 5.0)
    assert np.isclose(
        results2["service_vulnerability_index"], 100.0 * (0.7 * (1.0 / 3.0) + 0.3 * 0.5)
    )


def test_identify_critical_bottlenecks():
    pre = np.array([10.0, 5.0, 20.0])
    post = np.array([12.0, 5.0, 35.0])

    indices, load = identify_critical_bottlenecks(pre, post, top_k=2)
    np.testing.assert_array_equal(indices, [2, 0])
    np.testing.assert_allclose(load, [15.0, 2.0])


def test_prioritize_debris_clearance():
    blocked = np.array([1, 4])
    debris = np.array([10.0, 100.0])
    criticality = np.array([10.0, 50.0, 20.0, 10.0, 200.0])

    order, scores = prioritize_debris_clearance(blocked, debris, criticality)
    np.testing.assert_array_equal(order, [1, 4])
    assert scores[0] > scores[1]

    with pytest.raises(ValueError, match="same length"):
        prioritize_debris_clearance(blocked, debris[:-1], criticality)


def test_coastal_flood_inundation():
    dem = np.array([[1.0, 5.0, 10.0], [1.0, 5.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float64)

    # 1. No seeds when no boundary cell <= 0.0
    flooded, depth = coastal_flood_inundation(dem, water_level=2.0)
    assert np.all(~flooded)

    # 2. With custom sea_mask starting at (0, 0)
    sea_mask = np.zeros((3, 3), dtype=bool)
    sea_mask[0, 0] = True

    flooded, depth = coastal_flood_inundation(dem, water_level=2.0, sea_mask=sea_mask)

    # Connected cells <= 2.0 should be flooded:
    # (0,0), (1,0), (2,0), (2,1), (2,2), (1,2)
    assert flooded[0, 0]
    assert flooded[1, 0]
    assert flooded[2, 0]
    assert flooded[2, 1]
    assert flooded[2, 2]
    assert flooded[1, 2]

    assert not flooded[0, 1]
    assert not flooded[1, 1]
    assert not flooded[0, 2]

    # Check depth: water_level - dem
    assert np.isclose(depth[0, 0], 1.0)
    assert np.isclose(depth[0, 1], 0.0)


def test_landslide_susceptibility():
    dem_flat = np.ones((3, 3), dtype=np.float64) * 10.0
    scores, classes = landslide_susceptibility(dem_flat, cell_size=10.0)
    assert np.allclose(scores, 0.0)
    assert classes == [["Low", "Low", "Low"], ["Low", "Low", "Low"], ["Low", "Low", "Low"]]

    dem_steep = np.array(
        [[100.0, 100.0, 100.0], [50.0, 50.0, 50.0], [0.0, 0.0, 0.0]], dtype=np.float64
    )
    scores_steep, classes_steep = landslide_susceptibility(dem_steep, cell_size=10.0)
    assert np.isclose(scores_steep[1, 1], 100.0)
    assert classes_steep[1][1] == "Very High"

    with pytest.raises(ValueError, match="must match dem shape"):
        landslide_susceptibility(dem_flat, cell_size=10.0, soil_susceptibility=np.ones((2, 2)))


def test_wildfire_risk_index():
    dem_flat = np.ones((3, 3), dtype=np.float64) * 10.0
    veg = np.zeros((3, 3), dtype=np.float64)

    scores, classes = wildfire_risk_index(dem_flat, cell_size=10.0, vegetation_density=veg)
    assert np.allclose(scores, 0.0)
    assert classes == [["Low", "Low", "Low"], ["Low", "Low", "Low"], ["Low", "Low", "Low"]]

    veg_high = np.ones((3, 3), dtype=np.float64)
    scores_veg, classes_veg = wildfire_risk_index(
        dem_flat, cell_size=10.0, vegetation_density=veg_high
    )
    assert np.allclose(scores_veg, 45.0)
    assert classes_veg[1][1] == "Moderate"

    with pytest.raises(ValueError, match="must match dem shape"):
        wildfire_risk_index(dem_flat, cell_size=10.0, vegetation_density=np.zeros((2, 2)))


def test_network_criticality_index():
    # Simple line graph: 0 - 1 - 2
    indptr = np.array([0, 1, 3, 4], dtype=np.int64)
    adj = np.array([1, 0, 2, 1], dtype=np.int64)
    weights = np.array([1.5, 1.5, 2.5, 2.5], dtype=np.float64)
    n = 3

    # If we evaluate target edge 0 (connecting 0 -> 1)
    res = network_criticality_index(indptr, adj, weights, n, target_edges=[0], target_nodes=[1])

    assert "edges_nci" in res
    assert "nodes_nci" in res
    # Blocking edge 0 should cause drop in efficiency, NCI should be positive
    assert res["edges_nci"][0] > 0.0
    # Blocking node 1 (the center hub connecting 0 and 2) should completely segment the network,
    # so efficiency drops significantly
    assert res["nodes_nci"][0] > 0.0


def test_urban_heat_island_intensity():
    # 2x2 grid
    albedo = np.array([[0.1, 0.2], [0.15, 0.8]])
    ndvi = np.array([[0.0, 0.5], [-0.2, 0.9]])
    bh = np.array([[10.0, 5.0], [20.0, 0.0]])
    bf = np.array([[0.5, 0.3], [0.8, 0.0]])

    intensity = urban_heat_island_intensity(albedo, ndvi, bh, bf)

    assert intensity.shape == (2, 2)
    # The cell with low albedo, low vegetation, and high building density (0, 0 or 1, 0)
    # should have higher UHI intensity than the green park cell (1, 1)
    assert intensity[1, 0] > intensity[1, 1]
    assert np.all(intensity >= 0.0)

    # Argument validation
    with pytest.raises(ValueError):
        urban_heat_island_intensity(albedo, ndvi[:-1], bh, bf)


def test_socio_economic_flood_risk():
    # 2x2 grid
    hazard = np.array([[1.0, 0.0], [0.5, 0.2]])
    exposure = np.array([[10.0, 20.0], [5.0, 1.0]])
    svi = np.array([[0.8, 0.2], [0.6, 0.4]])

    # 1. Multiplicative method
    scores, classes = socio_economic_flood_risk(hazard, exposure, svi, method="multiplicative")
    assert scores.shape == (2, 2)
    assert len(classes) == 2
    # Since inputs are normalized internally via min-max:
    # Max hazard is 1.0 -> 100, min is 0.0 -> 0
    # Max exposure is 20.0 -> 100, min is 1.0 -> 0
    # Max SVI is 0.8 -> 100, min is 0.2 -> 0
    # At (0,0): h_norm=100, e_norm=9/19*100=47.37, v_norm=100 -> score = 100*47.37*100/10000 = 47.37
    # At (0,1): h_norm=0, e_norm=100, v_norm=0 -> score = 0
    assert np.isclose(scores[0, 1], 0.0)
    assert classes[0][1] == "Low"

    # 2. Additive method
    scores_add, classes_add = socio_economic_flood_risk(hazard, exposure, svi, method="additive")
    assert scores_add.shape == (2, 2)
    assert scores_add[0, 0] > 0.0

    # Error handling
    with pytest.raises(ValueError):
        socio_economic_flood_risk(hazard, exposure[:-1], svi)


def test_debris_clearance_routing():
    # Simple line graph: 0 - 1 - 2
    # CSR representations:
    # 0 -> 1 (edge 0, weight 1.5)
    # 1 -> 0 (edge 1, weight 1.5)
    # 1 -> 2 (edge 2, weight 2.5)
    # 2 -> 1 (edge 3, weight 2.5)
    indptr = np.array([0, 1, 3, 4], dtype=np.int64)
    adj = np.array([1, 0, 2, 1], dtype=np.int64)
    weights = np.array([1.5, 1.5, 2.5, 2.5], dtype=np.float64)
    n = 3

    # Let's say edges 0 and 2 are blocked by debris
    blocked_edges = np.array([0, 2])
    debris_volumes = np.array([10.0, 50.0])  # Edge 0 has less debris (easier to clear)
    edge_criticality = np.ones(4)

    # Run routing starting from depot_node = 0
    clearance_order, total_dist = debris_clearance_routing(
        indptr, adj, weights, n, blocked_edges, debris_volumes, edge_criticality, depot_node=0
    )

    # The vehicle should clear edge 0 first because it's closer to the depot and has less debris,
    # then move to 1, then clear edge 2.
    assert np.array_equal(clearance_order, [0, 2])
    assert total_dist > 0.0

    # Error checking
    with pytest.raises(ValueError):
        debris_clearance_routing(
            indptr,
            adj,
            weights,
            n,
            blocked_edges,
            debris_volumes[:-1],
            edge_criticality,
            depot_node=0,
        )


# ---------------------------------------------------------------------------
# Additional coverage: social_vulnerability_index
# ---------------------------------------------------------------------------


def test_social_vulnerability_index_errors():
    with pytest.raises(ValueError, match="At least one indicator"):
        social_vulnerability_index({}, {})

    indicators = {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([1.0, 2.0])}
    with pytest.raises(ValueError, match="length must match"):
        social_vulnerability_index(indicators, {"a": 1.0, "b": 1.0})


def test_social_vulnerability_index_edge_cases():
    # All-NaN indicator -> normalized to zeros
    indicators = {
        "a": np.array([np.nan, np.nan, np.nan]),
        "b": np.array([10.0, 20.0, 30.0]),
    }
    scores, classes = social_vulnerability_index(indicators, {"a": 1.0, "b": 1.0})
    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))

    # Uniform (non-varying) indicator -> max_v <= min_v branch
    indicators_uniform = {
        "a": np.array([5.0, 5.0, 5.0]),
        "b": np.array([0.0, 50.0, 100.0]),
    }
    scores_u, _ = social_vulnerability_index(indicators_uniform, {"a": 1.0, "b": 1.0})
    # 'a' contributes 0 everywhere, so score = b_norm / 2
    np.testing.assert_allclose(scores_u, [0.0, 25.0, 50.0])

    # Zero/negative weight is skipped; all-zero weights fall back to weight_sum=1.0
    indicators_w = {"a": np.array([0.0, 100.0]), "b": np.array([100.0, 0.0])}
    scores_skip, _ = social_vulnerability_index(indicators_w, {"a": 1.0, "b": 0.0})
    np.testing.assert_allclose(scores_skip, [0.0, 100.0])

    scores_allzero, _ = social_vulnerability_index(indicators_w, {"a": 0.0, "b": 0.0})
    np.testing.assert_allclose(scores_allzero, [0.0, 0.0])

    # NaN score (one indicator invalid for a unit, causing NaN propagation) -> "Low"
    indicators_nan_unit = {
        "a": np.array([1.0, 2.0, np.nan]),
        "b": np.array([5.0, 10.0, 15.0]),
    }
    scores_nan, classes_nan = social_vulnerability_index(indicators_nan_unit, {"a": 1.0, "b": 1.0})
    assert not np.isfinite(scores_nan[2])
    assert classes_nan[2] == "Low"

    # Full classification range: Low / Moderate / High / Very High
    indicators_range = {"x": np.array([0.0, 40.0, 60.0, 80.0, 100.0])}
    scores_range, classes_range = social_vulnerability_index(indicators_range, {"x": 1.0})
    np.testing.assert_allclose(scores_range, [0.0, 40.0, 60.0, 80.0, 100.0])
    assert classes_range == ["Low", "Moderate", "High", "Very High", "Very High"]


# ---------------------------------------------------------------------------
# Additional coverage: landslide_susceptibility
# ---------------------------------------------------------------------------


def test_landslide_susceptibility_errors():
    with pytest.raises(ValueError, match="2D array"):
        landslide_susceptibility(np.array([1.0, 2.0, 3.0]), cell_size=10.0)

    with pytest.raises(ValueError, match="cell_size must be greater than 0"):
        landslide_susceptibility(np.ones((3, 3)), cell_size=0.0)

    with pytest.raises(ValueError, match="lulc_susceptibility shape must match dem shape"):
        landslide_susceptibility(
            np.ones((3, 3)), cell_size=10.0, lulc_susceptibility=np.ones((2, 2))
        )


def test_landslide_susceptibility_weight_sum_fallback():
    dem = np.array([[100.0, 100.0, 100.0], [50.0, 50.0, 50.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    scores, classes = landslide_susceptibility(
        dem, cell_size=10.0, slope_weight=0.0, soil_weight=0.0, lulc_weight=0.0
    )
    np.testing.assert_allclose(scores, 0.0)
    assert all(c == "Low" for row in classes for c in row)


def test_landslide_susceptibility_edge_cases():
    # All-NaN DEM
    dem_nan = np.full((2, 2), np.nan)
    scores, classes = landslide_susceptibility(dem_nan, cell_size=10.0)
    np.testing.assert_allclose(scores, 0.0)
    assert classes == [["Low", "Low"], ["Low", "Low"]]

    # Isolate soil component (slope_weight=0, lulc_weight=0) across full class range
    dem_flat = np.ones((1, 5)) * 10.0
    soil = np.array([[0.0, 40.0, 60.0, 80.0, 100.0]])
    scores_soil, classes_soil = landslide_susceptibility(
        dem_flat,
        cell_size=10.0,
        soil_susceptibility=soil,
        slope_weight=0.0,
        soil_weight=1.0,
        lulc_weight=0.0,
    )
    np.testing.assert_allclose(scores_soil, soil)
    assert classes_soil == [["Low", "Moderate", "High", "Very High", "Very High"]]

    # Isolate LULC component (valid shape, non-default weighting)
    lulc = np.array([[10.0, 50.0, 90.0]])
    dem_flat3 = np.ones((1, 3)) * 5.0
    scores_lulc, _ = landslide_susceptibility(
        dem_flat3,
        cell_size=10.0,
        lulc_susceptibility=lulc,
        slope_weight=0.0,
        soil_weight=0.0,
        lulc_weight=1.0,
    )
    np.testing.assert_allclose(scores_lulc, lulc)

    # NaN preserved from the DEM alongside otherwise valid cells
    dem_partial_nan = np.array([[10.0, np.nan], [10.0, 10.0]])
    scores_partial, classes_partial = landslide_susceptibility(dem_partial_nan, cell_size=10.0)
    assert not np.isfinite(scores_partial[0, 1])
    assert classes_partial[0][1] == "Low"


# ---------------------------------------------------------------------------
# Additional coverage: wildfire_risk_index
# ---------------------------------------------------------------------------


def test_wildfire_risk_index_errors():
    veg = np.zeros((3, 3))
    with pytest.raises(ValueError, match="2D array"):
        wildfire_risk_index(np.array([1.0, 2.0]), cell_size=10.0, vegetation_density=veg[0])

    with pytest.raises(ValueError, match="cell_size must be greater than 0"):
        wildfire_risk_index(np.ones((3, 3)), cell_size=0.0, vegetation_density=veg)


def test_wildfire_risk_index_weight_sum_fallback():
    dem = np.array([[100.0, 100.0, 100.0], [50.0, 50.0, 50.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    veg = np.ones((3, 3))
    scores, classes = wildfire_risk_index(
        dem,
        cell_size=10.0,
        vegetation_density=veg,
        slope_weight=0.0,
        aspect_weight=0.0,
        veg_weight=0.0,
    )
    np.testing.assert_allclose(scores, 0.0)
    assert all(c == "Low" for row in classes for c in row)


def test_wildfire_risk_index_edge_cases():
    # All-NaN DEM
    dem_nan = np.full((2, 2), np.nan)
    scores, classes = wildfire_risk_index(
        dem_nan, cell_size=10.0, vegetation_density=np.zeros((2, 2))
    )
    np.testing.assert_allclose(scores, 0.0)
    assert classes == [["Low", "Low"], ["Low", "Low"]]

    # Isolate vegetation factor to hit the full classification range
    dem_flat = np.ones((1, 4)) * 10.0
    veg_range = np.array([[0.0, 0.4, 0.6, 0.9]])
    scores_range, classes_range = wildfire_risk_index(
        dem_flat,
        cell_size=10.0,
        vegetation_density=veg_range,
        slope_weight=0.0,
        aspect_weight=0.0,
        veg_weight=1.0,
    )
    np.testing.assert_allclose(scores_range, [[0.0, 40.0, 60.0, 90.0]])
    assert classes_range == [["Low", "Moderate", "High", "Very High"]]

    # Southern hemisphere aspect scoring differs from northern for a sloped DEM
    dem_slope = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [20.0, 20.0, 20.0]])
    veg_zero = np.zeros((3, 3))
    scores_north, _ = wildfire_risk_index(
        dem_slope, cell_size=10.0, vegetation_density=veg_zero, hemisphere="northern"
    )
    scores_south, _ = wildfire_risk_index(
        dem_slope, cell_size=10.0, vegetation_density=veg_zero, hemisphere="southern"
    )
    assert not np.allclose(scores_north, scores_south)

    # NaN preserved from the DEM
    dem_partial_nan = np.array([[10.0, np.nan], [10.0, 10.0]])
    scores_partial, classes_partial = wildfire_risk_index(
        dem_partial_nan, cell_size=10.0, vegetation_density=np.zeros((2, 2))
    )
    assert not np.isfinite(scores_partial[0, 1])
    assert classes_partial[0][1] == "Low"


# ---------------------------------------------------------------------------
# Additional coverage: multi_hazard_composite / equity_adjusted_priority
# ---------------------------------------------------------------------------


def test_multi_hazard_composite_errors():
    with pytest.raises(ValueError, match="At least two hazard"):
        multi_hazard_composite({"heat": np.array([1.0, 2.0])})

    with pytest.raises(ValueError, match="does not match"):
        multi_hazard_composite({"heat": np.array([1.0, 2.0]), "flood": np.array([1.0, 2.0, 3.0])})


def test_multi_hazard_composite_edge_cases():
    # Default (equal) weights
    hazards = {"heat": np.array([80.0, 20.0]), "flood": np.array([20.0, 80.0])}
    scores, classes, dominant, diversity, drivers = multi_hazard_composite(hazards)
    np.testing.assert_allclose(scores, [50.0, 50.0])

    # Very High classification
    hazards_high = {"heat": np.array([100.0]), "flood": np.array([100.0])}
    scores_high, classes_high, _, _, _ = multi_hazard_composite(hazards_high)
    assert classes_high == ["Very High"]

    # 2D input reshaping of list results
    hazards_2d = {
        "heat": np.array([[80.0, 20.0], [10.0, 90.0]]),
        "flood": np.array([[40.0, 10.0], [5.0, 95.0]]),
    }
    scores_2d, classes_2d, dominant_2d, diversity_2d, drivers_2d = multi_hazard_composite(
        hazards_2d, {"heat": 0.5, "flood": 0.5}
    )
    assert scores_2d.shape == (2, 2)
    assert len(classes_2d) == 2 and len(classes_2d[0]) == 2
    assert len(dominant_2d) == 2 and len(dominant_2d[0]) == 2
    assert len(drivers_2d) == 2 and len(drivers_2d[0]) == 2

    # All-NaN unit for a given position
    hazards_nan = {
        "heat": np.array([np.nan, 50.0]),
        "flood": np.array([np.nan, 50.0]),
    }
    scores_nan, classes_nan, dominant_nan, diversity_nan, drivers_nan = multi_hazard_composite(
        hazards_nan
    )
    assert not np.isfinite(scores_nan[0])
    assert classes_nan[0] == "Low"
    assert dominant_nan[0] == ""
    assert not np.isfinite(diversity_nan[0])
    assert drivers_nan[0] == []

    # Higher-dimensional (3D) shape falls back to a flat list for classes/dominant/drivers
    hazards_3d = {
        "heat": np.full((1, 2, 2), 80.0),
        "flood": np.full((1, 2, 2), 20.0),
    }
    scores_3d, classes_3d, dominant_3d, diversity_3d, drivers_3d = multi_hazard_composite(
        hazards_3d
    )
    assert scores_3d.shape == (1, 2, 2)
    assert isinstance(classes_3d, list) and len(classes_3d) == 4
    assert isinstance(dominant_3d, list) and len(dominant_3d) == 4
    assert isinstance(drivers_3d, list) and len(drivers_3d) == 4


def test_equity_adjusted_priority_errors():
    with pytest.raises(ValueError, match="same shape"):
        equity_adjusted_priority(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_equity_adjusted_priority_edge_cases():
    # 1D shape branch, including a NaN hazard value and full classification range
    hazard = np.array([0.0, 40.0, 60.0, 90.0, np.nan])
    svi = np.zeros(5)
    scores, raw, factors, classes = equity_adjusted_priority(hazard, svi, equity_weight=0.0)
    assert classes == ["Low", "Moderate", "High", "Very High", "Low"]
    assert not np.isfinite(scores[4])

    # Higher-dimensional (3D) shape falls back to a flat list of classes
    hazard_3d = np.zeros((1, 2, 2))
    svi_3d = np.zeros((1, 2, 2))
    _, _, _, classes_3d = equity_adjusted_priority(hazard_3d, svi_3d, equity_weight=0.5)
    assert isinstance(classes_3d, list)
    assert len(classes_3d) == 4
    assert all(isinstance(c, str) for c in classes_3d)


# ---------------------------------------------------------------------------
# Additional coverage: pluvial_flood_susceptibility
# ---------------------------------------------------------------------------


def test_pluvial_flood_susceptibility_errors():
    with pytest.raises(ValueError, match="2D array"):
        pluvial_flood_susceptibility(np.array([1.0, 2.0, 3.0]), cell_size=10.0)

    dem = np.ones((3, 3)) * 10.0
    with pytest.raises(ValueError, match="drainage_dists shape must match dem shape"):
        pluvial_flood_susceptibility(dem, cell_size=10.0, drainage_dists=np.ones((2, 2)))


def test_pluvial_flood_susceptibility_edge_cases():
    # All-NaN DEM
    dem_nan = np.full((2, 2), np.nan)
    scores, classes = pluvial_flood_susceptibility(dem_nan, cell_size=10.0)
    np.testing.assert_allclose(scores, 0.0)
    assert classes == [["Low", "Low"], ["Low", "Low"]]

    # Uniform elevation -> elev_range <= 0 fallback
    dem_uniform = np.full((3, 3), 5.0)
    scores_u, _ = pluvial_flood_susceptibility(dem_uniform, cell_size=10.0)
    assert np.all(np.isfinite(scores_u))

    # All weights zero -> weight_sum <= 0 fallback
    dem = np.array([[10.0, 12.0, 15.0], [8.0, 9.0, 11.0], [5.0, 7.0, 8.0]])
    scores_zero, classes_zero = pluvial_flood_susceptibility(
        dem, cell_size=10.0, elevation_weight=0.0, slope_weight=0.0, drainage_weight=0.0
    )
    np.testing.assert_allclose(scores_zero, 0.0)
    assert all(c == "Low" for row in classes_zero for c in row)

    # Isolate drainage proximity across the full classification range
    dem_flat = np.zeros((1, 4))
    drainage = np.array([[0.0, 140.0, 260.0, 400.0]])
    scores_drain, classes_drain = pluvial_flood_susceptibility(
        dem_flat,
        cell_size=10.0,
        neighborhood_radius=100.0,
        drainage_dists=drainage,
        elevation_weight=0.0,
        slope_weight=0.0,
        drainage_weight=1.0,
    )
    np.testing.assert_allclose(scores_drain, [[100.0, 65.0, 35.0, 0.0]])
    assert classes_drain == [["Very High", "High", "Moderate", "Low"]]

    # NaN preserved from the DEM
    dem_partial_nan = np.array([[10.0, np.nan], [10.0, 10.0]])
    scores_partial, classes_partial = pluvial_flood_susceptibility(dem_partial_nan, cell_size=10.0)
    assert not np.isfinite(scores_partial[0, 1])
    assert classes_partial[0][1] == "Low"


# ---------------------------------------------------------------------------
# Additional coverage: coastal_flood_inundation
# ---------------------------------------------------------------------------


def test_coastal_flood_inundation_errors_and_edge_cases():
    with pytest.raises(ValueError, match="2D array"):
        coastal_flood_inundation(np.array([1.0, 2.0]), water_level=1.0)

    dem = np.ones((3, 3))
    with pytest.raises(ValueError, match="sea_mask shape must match dem shape"):
        coastal_flood_inundation(dem, water_level=1.0, sea_mask=np.ones((2, 2), dtype=bool))

    # Degenerate (empty) dimension: boundary-seeding shortcut is skipped
    dem_empty = np.zeros((0, 3))
    flooded, depth = coastal_flood_inundation(dem_empty, water_level=1.0)
    assert flooded.shape == (0, 3)
    assert depth.shape == (0, 3)


# ---------------------------------------------------------------------------
# Additional coverage: socio_economic_flood_risk
# ---------------------------------------------------------------------------


def test_socio_economic_flood_risk_edge_cases():
    # All cells invalid
    nan_grid = np.full((2, 2), np.nan)
    scores, classes = socio_economic_flood_risk(nan_grid, nan_grid, nan_grid)
    np.testing.assert_allclose(scores, 0.0)
    assert all(c == "Low" for row in classes for c in row)

    # Uniform hazard array -> rng <= 0 fallback inside min_max_normalize
    hazard_uniform = np.full((2, 2), 3.0)
    exposure = np.array([[1.0, 2.0], [3.0, 4.0]])
    svi = np.array([[10.0, 20.0], [30.0, 40.0]])
    scores_u, _ = socio_economic_flood_risk(hazard_uniform, exposure, svi, method="multiplicative")
    assert scores_u.shape == (2, 2)
    assert np.all(np.isfinite(scores_u))

    # Unknown method
    with pytest.raises(ValueError, match="Unknown risk calculation method"):
        socio_economic_flood_risk(exposure, exposure, svi, method="bogus")

    # Additive method with all-zero weights -> w_sum <= 0 fallback
    scores_add_zero, _ = socio_economic_flood_risk(
        exposure,
        exposure,
        svi,
        method="additive",
        w_hazard=0.0,
        w_exposure=0.0,
        w_vulnerability=0.0,
    )
    np.testing.assert_allclose(scores_add_zero, 0.0)

    # NaN classification (one invalid cell) plus a Very High score cell
    hazard = np.array([[100.0, np.nan], [50.0, 20.0]])
    exposure2 = np.array([[100.0, 50.0], [50.0, 10.0]])
    svi2 = np.array([[100.0, 50.0], [50.0, 5.0]])
    scores2, classes2 = socio_economic_flood_risk(hazard, exposure2, svi2, method="multiplicative")
    assert not np.isfinite(scores2[0, 1])
    assert classes2[0][1] == "Low"
    assert np.isclose(scores2[0, 0], 100.0)
    assert classes2[0][0] == "Very High"

    # "High" classification band ([50, 75)): isolate the hazard weight so the
    # additive score equals the normalized hazard value directly. Using min=0 and
    # max=100 keeps the normalization a no-op for the middle value.
    hazard4 = np.array([[0.0, 60.0, 100.0]])
    other4 = np.array([[1.0, 2.0, 3.0]])
    scores3, classes3 = socio_economic_flood_risk(
        hazard4,
        other4,
        other4,
        method="additive",
        w_hazard=1.0,
        w_exposure=0.0,
        w_vulnerability=0.0,
    )
    np.testing.assert_allclose(scores3, [[0.0, 60.0, 100.0]])
    assert classes3[0][1] == "High"


# ---------------------------------------------------------------------------
# Additional coverage: urban_heat_comfort_risk / urban_heat_island_intensity
# ---------------------------------------------------------------------------


def test_urban_heat_comfort_risk_errors_and_edge_cases():
    imp = np.zeros((2, 2))
    bld = np.zeros((2, 2))
    grn = np.zeros((2, 2))
    dst = np.zeros((2, 2))
    vuln = np.zeros((2, 2))

    with pytest.raises(ValueError, match="same shape"):
        urban_heat_comfort_risk(imp, bld[0], grn, dst, vuln)

    # All weights zero -> weight_sum <= 0 fallback
    scores_zero, classes_zero = urban_heat_comfort_risk(
        imp, bld, grn, dst, vuln, w_imperv=0.0, w_green=0.0, w_build=0.0, w_vuln=0.0
    )
    np.testing.assert_allclose(scores_zero, 0.0)
    assert all(c == "Low" for row in classes_zero for c in row)

    # Isolate impervious factor across the full classification range
    imp_range = np.array([[0.0, 0.4], [0.6, 0.9]])
    scores_range, classes_range = urban_heat_comfort_risk(
        imp_range, bld, grn, dst, vuln, w_imperv=1.0, w_green=0.0, w_build=0.0, w_vuln=0.0
    )
    np.testing.assert_allclose(scores_range, [[0.0, 40.0], [60.0, 90.0]])
    assert classes_range == [["Low", "Moderate"], ["High", "Very High"]]

    # NaN propagation from an input array
    imp_nan = np.array([[np.nan, 0.2], [0.3, 0.4]])
    scores_nan, classes_nan = urban_heat_comfort_risk(imp_nan, bld, grn, dst, vuln)
    assert not np.isfinite(scores_nan[0, 0])
    assert classes_nan[0][0] == "Low"


def test_urban_heat_island_intensity_wind_speed_variants():
    albedo = np.array([[0.1, 0.2], [0.15, 0.8]])
    ndvi = np.array([[0.0, 0.5], [-0.2, 0.9]])
    bh = np.array([[10.0, 5.0], [20.0, 0.0]])
    bf = np.array([[0.5, 0.3], [0.8, 0.0]])

    # Valid wind_speed array (matching shape)
    wind = np.array([[2.0, 3.0], [0.0, 5.0]])
    intensity = urban_heat_island_intensity(albedo, ndvi, bh, bf, wind_speed=wind)
    assert intensity.shape == (2, 2)
    assert np.all(intensity >= 0.0)

    # Mismatched wind_speed shape
    with pytest.raises(ValueError, match="wind_speed array must have the same shape"):
        urban_heat_island_intensity(albedo, ndvi, bh, bf, wind_speed=wind[0])


# ---------------------------------------------------------------------------
# Additional coverage: simulate_network_disruption
# ---------------------------------------------------------------------------


def test_simulate_network_disruption_empty_lists():
    indptr = np.array([0, 1, 3, 4], dtype=np.int64)
    adj = np.array([1, 0, 2, 1], dtype=np.int64)
    weights = np.array([1.5, 1.5, 2.5, 2.5], dtype=np.float64)

    # Explicit empty blocked_edges list: should skip the edge-blocking branch
    w1 = simulate_network_disruption(indptr, adj, weights, n=3, blocked_edges=[])
    np.testing.assert_allclose(w1, weights)

    # Explicit empty blocked_nodes list: should skip the node-blocking branch
    w2 = simulate_network_disruption(indptr, adj, weights, n=3, blocked_nodes=[])
    np.testing.assert_allclose(w2, weights)


# ---------------------------------------------------------------------------
# Additional coverage: infrastructure_service_loss
# ---------------------------------------------------------------------------


def test_infrastructure_service_loss_edge_cases():
    dists_pre = np.array([[1.0, 2.0], [3.0, 4.0]])
    dists_post = np.array([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(ValueError, match="identical shapes"):
        infrastructure_service_loss(dists_pre, dists_post[:-1])

    with pytest.raises(ValueError, match="must match number of origins"):
        infrastructure_service_loss(dists_pre, dists_post, demands=np.array([1.0]))

    # demands=None default -> all origins weighted equally, no isolation
    results = infrastructure_service_loss(dists_pre, dists_post)
    assert np.isclose(results["isolation_rate"], 0.0)

    # total_demand <= 0 fallback (all demands zero); make every origin unreachable
    # post-disruption so the "connected" mean-delay branch is not exercised with an
    # all-zero weights array.
    dists_post_isolated = np.array([[np.inf, np.inf], [np.inf, np.inf]])
    results_zero_demand = infrastructure_service_loss(
        dists_pre, dists_post_isolated, demands=np.zeros(2)
    )
    assert np.isclose(results_zero_demand["isolation_rate"], 0.0)
    assert np.isclose(results_zero_demand["mean_delay"], 0.0)

    # Zero demand on origins that remain connected must not raise ZeroDivisionError
    # from np.average(weights=...) summing to zero; falls back to unweighted average.
    # dists_pre min-per-origin = [1.0, 3.0]; dists_post_connected min-per-origin = [1.0, 5.0]
    # -> diff = [0.0, 2.0] -> unweighted mean_delay = 1.0
    dists_post_connected = np.array([[1.0, 2.0], [5.0, 6.0]])
    results_zero_weight_connected = infrastructure_service_loss(
        dists_pre, dists_post_connected, demands=np.zeros(2)
    )
    assert np.isclose(results_zero_weight_connected["mean_delay"], 1.0)

    # cutoff provided: origin becomes "isolated" once travel exceeds the cutoff
    dists_pre_cutoff = np.array([[5.0], [3.0]])
    dists_post_cutoff = np.array([[12.0], [3.0]])
    results_cutoff = infrastructure_service_loss(dists_pre_cutoff, dists_post_cutoff, cutoff=10.0)
    assert np.isclose(results_cutoff["isolation_rate"], 0.5)
    assert np.isclose(results_cutoff["pop_isolated"], 1.0)


# ---------------------------------------------------------------------------
# Additional coverage: identify_critical_bottlenecks / prioritize_debris_clearance
# ---------------------------------------------------------------------------


def test_identify_critical_bottlenecks_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        identify_critical_bottlenecks(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_prioritize_debris_clearance_edge_cases():
    blocked = np.array([0, 1])
    criticality = np.array([10.0, 20.0, 30.0])

    with pytest.raises(ValueError, match="non-negative"):
        prioritize_debris_clearance(blocked, np.array([-1.0, 5.0]), criticality)

    with pytest.raises(ValueError, match="valid range of edge_criticality"):
        prioritize_debris_clearance(np.array([0, 100]), np.array([1.0, 1.0]), criticality)


# ---------------------------------------------------------------------------
# Additional coverage: network_criticality_index branch scenarios
# ---------------------------------------------------------------------------


def test_network_criticality_index_branch_scenarios():
    indptr_line = np.array([0, 1, 3, 4], dtype=np.int64)
    adj_line = np.array([1, 0, 2, 1], dtype=np.int64)
    weights_line = np.array([1.5, 1.5, 2.5, 2.5], dtype=np.float64)

    # origins/destinations subset (node 0 is an origin but not a destination) exercises
    # both branches of the "is this origin also a destination" check, and calling with
    # only target_nodes (no target_edges) exercises the target_edges=None skip branch.
    res_subset = network_criticality_index(
        indptr_line,
        adj_line,
        weights_line,
        n=3,
        target_nodes=[1],
        origins=[0, 1, 2],
        destinations=[1, 2],
    )
    assert "nodes_nci" in res_subset
    assert "edges_nci" not in res_subset

    # eff_base <= 0 (origins == destinations == a single self node) skips the
    # NCI-computation loops for both edges and nodes.
    res_zero_eff = network_criticality_index(
        indptr_line,
        adj_line,
        weights_line,
        n=3,
        target_edges=[0],
        target_nodes=[0],
        origins=[0],
        destinations=[0],
    )
    np.testing.assert_allclose(res_zero_eff["edges_nci"], [0.0])
    np.testing.assert_allclose(res_zero_eff["nodes_nci"], [0.0])

    # Directed triangle mesh: node1's outgoing edges are ordered so the search for
    # edge0's (0 -> 1) reverse counterpart at node 1 first checks a non-matching edge
    # (continue) before finding the match (break).
    indptr_tri = np.array([0, 2, 4, 6], dtype=np.int64)
    adj_tri = np.array([1, 2, 2, 0, 0, 1], dtype=np.int64)
    weights_tri = np.ones(6, dtype=np.float64)
    res_tri = network_criticality_index(
        indptr_tri, adj_tri, weights_tri, n=3, target_edges=[0], target_nodes=[1]
    )
    assert res_tri["edges_nci"][0] >= 0.0
    assert res_tri["nodes_nci"][0] >= 0.0

    # Directed cycle (no reverse edges at all): the counterpart search never finds a
    # match and simply falls through the loop.
    indptr_cycle = np.array([0, 1, 2, 3], dtype=np.int64)
    adj_cycle = np.array([1, 2, 0], dtype=np.int64)
    weights_cycle = np.ones(3, dtype=np.float64)
    res_cycle = network_criticality_index(
        indptr_cycle, adj_cycle, weights_cycle, n=3, target_edges=[0]
    )
    assert res_cycle["edges_nci"][0] >= 0.0


# ---------------------------------------------------------------------------
# Additional coverage: debris_clearance_routing branch scenarios
# ---------------------------------------------------------------------------


def test_debris_clearance_routing_unreachable_fallback():
    # Two disconnected directed edges: 0 -> 1 and 2 -> 3. Once the vehicle clears
    # edge 0 it becomes stranded at node 1, so the remaining blocked edge (2 -> 3)
    # is permanently unreachable, exercising the large-distance fallback branches.
    indptr = np.array([0, 1, 1, 2, 2], dtype=np.int64)
    adj = np.array([1, 3], dtype=np.int64)
    weights = np.array([1.0, 1.0], dtype=np.float64)
    n = 4

    blocked_edges = np.array([0, 1])
    debris_volumes = np.array([1.0, 1.0])
    edge_criticality = np.array([1.0, 1.0])

    clearance_order, total_distance = debris_clearance_routing(
        indptr, adj, weights, n, blocked_edges, debris_volumes, edge_criticality, depot_node=0
    )
    np.testing.assert_array_equal(clearance_order, [0, 1])
    # Edge 0 is reached at distance 0 (cost 0 + 1.0); edge 1's origin node is
    # unreachable, so only its own edge weight is added as a fallback cost.
    assert np.isclose(total_distance, 2.0)


def test_debris_clearance_routing_counterpart_search_branches():
    # Directed cycle (no reverse edges): the counterpart-restoration search never
    # finds a match for any cleared edge.
    indptr = np.array([0, 1, 2, 3], dtype=np.int64)
    adj = np.array([1, 2, 0], dtype=np.int64)
    weights = np.ones(3, dtype=np.float64)
    n = 3

    clearance_order, total_distance = debris_clearance_routing(
        indptr,
        adj,
        weights,
        n,
        blocked_edges=np.array([0, 1, 2]),
        debris_volumes=np.array([1.0, 1.0, 1.0]),
        edge_criticality=np.array([1.0, 1.0, 1.0]),
        depot_node=0,
    )
    np.testing.assert_array_equal(clearance_order, [0, 1, 2])
    assert np.isclose(total_distance, 3.0)

    # Directed triangle mesh: restoring edge 0's counterpart at node 1 first checks
    # a non-matching edge (continue) before finding the match (break).
    indptr_tri = np.array([0, 2, 4, 6], dtype=np.int64)
    adj_tri = np.array([1, 2, 2, 0, 0, 1], dtype=np.int64)
    weights_tri = np.ones(6, dtype=np.float64)

    clearance_order_tri, total_distance_tri = debris_clearance_routing(
        indptr_tri,
        adj_tri,
        weights_tri,
        n=3,
        blocked_edges=np.array([0]),
        debris_volumes=np.array([1.0]),
        edge_criticality=np.ones(6),
        depot_node=0,
    )
    np.testing.assert_array_equal(clearance_order_tri, [0])
    assert np.isclose(total_distance_tri, 1.0)


def test_optimize_canopy_placement():
    # Setup simple network: 3 nodes
    indptr = np.array([0, 1, 3, 4], dtype=np.int64)
    adj = np.array([1, 0, 2, 1], dtype=np.int64)
    edge_w = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float64)

    # Segment indicators
    flow = np.array([100.0, 100.0, 20.0, 20.0])  # high flow on edge 0->1
    canopy = np.array([0.1, 0.1, 0.8, 0.8])  # edge 1->2 already has high canopy
    heat = np.array([45.0, 45.0, 35.0, 35.0])  # higher heat index on 0->1

    # 1. Budget of 1 tree placement
    optimal_sites = optimize_canopy_placement(
        indptr,
        adj,
        edge_w,
        n=3,
        pedestrian_flow=flow,
        existing_canopy=canopy,
        heat_index=heat,
        num_trees=1,
    )
    # Edge index 0 (or 1) should be chosen first because it has high flow and low canopy
    assert len(optimal_sites) == 1
    assert optimal_sites[0] in (0, 1)

    # 2. Budget is zero or negative
    assert (
        len(
            optimize_canopy_placement(
                indptr,
                adj,
                edge_w,
                n=3,
                pedestrian_flow=flow,
                existing_canopy=canopy,
                heat_index=heat,
                num_trees=0,
            )
        )
        == 0
    )
    assert (
        len(
            optimize_canopy_placement(
                indptr,
                adj,
                edge_w,
                n=3,
                pedestrian_flow=flow,
                existing_canopy=canopy,
                heat_index=heat,
                num_trees=-5,
            )
        )
        == 0
    )

    with pytest.raises(ValueError, match="match number of edges"):
        optimize_canopy_placement(
            indptr,
            adj,
            edge_w,
            n=3,
            pedestrian_flow=flow[:-1],
            existing_canopy=canopy,
            heat_index=heat,
            num_trees=1,
        )


def test_calculate_grid_sky_view_factor():
    # 1. Open area
    flat_grid = np.zeros((5, 5))
    svf_open = calculate_grid_sky_view_factor(
        flat_grid, resolution=1.0, max_radius=3.0, num_directions=4
    )
    np.testing.assert_allclose(svf_open, 1.0)

    # 2. Obstructed center canyon
    canyon_grid = np.zeros((3, 3))
    canyon_grid[1, 1] = 10.0  # tall building in center
    svf = calculate_grid_sky_view_factor(
        canyon_grid, resolution=1.0, max_radius=2.0, num_directions=8
    )
    # The edges should see the central obstruction, SVF should be < 1.0
    assert svf[0, 0] < 1.0
    assert svf[0, 1] < 1.0
    assert svf[1, 1] == 1.0  # top of building has no obstruction above it

    # 3. Validation errors
    with pytest.raises(ValueError, match="height_grid must be a 2D array"):
        calculate_grid_sky_view_factor(np.zeros(5))

    with pytest.raises(ValueError, match="resolution must be greater"):
        calculate_grid_sky_view_factor(flat_grid, resolution=0.0)


def test_classify_local_climate_zones():
    # 3x1 grid representing different cell characteristics
    # Row 0: Compact high-rise (BSF=0.5, H=30.0, ISF=0.8) -> LCZ 1
    # Row 1: Open low-rise (BSF=0.3, H=5.0, ISF=0.4) -> LCZ 6
    # Row 2: Pervious (BSF=0.01, H=0.0, ISF=0.05) -> LCZ 11
    bsf = np.array([[0.5], [0.3], [0.01]])
    isf = np.array([[0.8], [0.4], [0.05]])
    h = np.array([[30.0], [5.0], [0.0]])

    lcz = classify_local_climate_zones(bsf, isf, h)
    assert lcz.shape == (3, 1)
    assert lcz[0, 0] == 1
    assert lcz[1, 0] == 6
    assert lcz[2, 0] == 11

    # Validation errors
    with pytest.raises(ValueError, match="same shape"):
        classify_local_climate_zones(bsf, isf, np.zeros((2, 1)))


def test_calculate_solar_access():
    # 1. Flat grid - open area
    flat_grid = np.zeros((3, 3))
    alts = np.array([45.0])
    azis = np.array([180.0])
    access = calculate_solar_access(
        flat_grid, resolution=1.0, sun_altitudes=alts, sun_azimuths=azis
    )
    np.testing.assert_allclose(access, 100.0)

    # 2. Obstructed grid - canyon shadow
    canyon_grid = np.zeros((3, 3))
    canyon_grid[1, 1] = 10.0  # building in center
    # Sun at 45 deg altitude, 180 deg (South) azimuth -> shadow cast North (ox=0, oy=-1)
    access = calculate_solar_access(
        canyon_grid,
        resolution=1.0,
        sun_altitudes=alts,
        sun_azimuths=azis,
        max_shadow_dist=5.0,
    )
    # The cell at (0, 1) (North of center building) should be shaded (0.0% solar access)
    assert access[0, 1] == 0.0
    # The cell at (2, 1) (South of center building) should be sunlit (100.0% solar access)
    assert access[2, 1] == 100.0

    # 3. Empty steps
    access_empty = calculate_solar_access(
        canyon_grid,
        resolution=1.0,
        sun_altitudes=np.array([]),
        sun_azimuths=np.array([]),
    )
    np.testing.assert_allclose(access_empty, 100.0)

    # 4. Validation errors
    with pytest.raises(ValueError, match="height_grid must be a 2D array"):
        calculate_solar_access(np.zeros(3), 1.0, alts, azis)

    with pytest.raises(ValueError, match="identical length"):
        calculate_solar_access(flat_grid, 1.0, np.array([45.0]), np.array([180.0, 90.0]))


def test_evacuation_route_optimization():
    adj = np.array([[0.0, 10.0, 0.0], [0.0, 0.0, 10.0], [0.0, 0.0, 0.0]])
    cap = np.array([[0.0, 50.0, 0.0], [0.0, 0.0, 50.0], [0.0, 0.0, 0.0]])
    demands = {0: 100.0}
    depots = [2]

    res = evacuation_route_optimization(adj, cap, demands, depots, vehicle_speed=40.0)

    assert res["assigned_flows"][0, 1] == 100.0
    assert res["assigned_flows"][1, 2] == 100.0
    assert len(res["bottleneck_edges"]) == 2
    assert res["clearance_time_hours"] > 0.0

    with pytest.raises(ValueError, match="square 2D arrays"):
        evacuation_route_optimization(adj[:2, :2], cap, demands, depots)

    with pytest.raises(ValueError, match="depot node must be specified"):
        evacuation_route_optimization(adj, cap, demands, [])


def test_coastal_surge_inundation():
    dem = np.array([[0.0, 1.0, 3.0], [0.5, 2.0, 5.0], [1.5, 4.0, 6.0]])
    sea = np.array([[True, False, False], [False, False, False], [False, False, False]])

    depth, flooded = coastal_surge_inundation(dem, surge_height=2.5, sea_mask=sea, cell_size=10.0)

    assert flooded[0, 0] == 1
    assert flooded[0, 1] == 1
    assert depth[0, 1] == 1.5

    with pytest.raises(ValueError, match="2D array"):
        coastal_surge_inundation(np.zeros(3), 2.5, sea)

    with pytest.raises(ValueError, match="sea_mask shape must match"):
        coastal_surge_inundation(dem, 2.5, sea[:2, :2])


def test_simulate_interdependent_infrastructure_cascade():
    p_adj = np.array([[0, 1], [1, 0]])
    w_adj = np.array([[0, 1], [1, 0]])
    dep = np.array([[1, 0], [0, 0]])

    res = simulate_interdependent_infrastructure_cascade(
        p_adj, w_adj, dep, initial_failed_power=[0]
    )

    assert 0 in res["failed_power_nodes"]
    assert 0 in res["failed_water_nodes"]
    assert res["cascade_iterations"] > 0
    assert 0.0 <= res["power_operability_ratio"] <= 1.0

    with pytest.raises(ValueError, match="shape must be"):
        simulate_interdependent_infrastructure_cascade(
            p_adj, w_adj, dep[:1, :], initial_failed_power=[0]
        )


def test_tree_canopy_microclimate_cooling():
    trees = np.array([[0.0, 0.0], [50.0, 50.0]])
    radii = np.array([5.0, 8.0])
    lai = np.array([3.5, 4.0])
    grid = np.array([[0.0, 0.0], [10.0, 0.0], [200.0, 200.0]])

    dt = tree_canopy_microclimate_cooling(trees, radii, lai, grid, max_cooling_dist=30.0)

    assert len(dt) == 3
    assert dt[0] == pytest.approx(0.6 * 3.5)
    assert dt[0] > dt[1] > dt[2]
    assert dt[2] == 0.0



