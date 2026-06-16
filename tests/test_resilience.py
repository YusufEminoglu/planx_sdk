# -*- coding: utf-8 -*-
"""Tests for the resilience submodule."""

import numpy as np
import pytest

from planx.resilience import (
    coastal_flood_inundation,
    debris_clearance_routing,
    equity_adjusted_priority,
    identify_critical_bottlenecks,
    infrastructure_service_loss,
    landslide_susceptibility,
    multi_hazard_composite,
    network_criticality_index,
    pluvial_flood_susceptibility,
    prioritize_debris_clearance,
    simulate_network_disruption,
    simulate_seismic_debris,
    social_vulnerability_index,
    socio_economic_flood_risk,
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
