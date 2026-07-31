# -*- coding: utf-8 -*-
"""Tests for the resilience submodule."""

import numpy as np
import pytest

from planx.resilience import (
    calculate_building_solar_radiation,
    calculate_grid_sky_view_factor,
    calculate_solar_access,
    classify_local_climate_zones,
    coastal_flood_inundation,
    coastal_surge_inundation,
    compound_hazard_cascade,
    debris_clearance_routing,
    detention_basin_sizing,
    earthquake_building_collapse_casualty,
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
    scs_unit_hydrograph,
    seismic_damage_loss_curve,
    seismic_road_blockage_simulation,
    simulate_interdependent_infrastructure_cascade,
    simulate_network_disruption,
    simulate_seismic_debris,
    social_vulnerability_index,
    socio_economic_flood_risk,
    stormwater_retention_basin_design,
    tree_canopy_microclimate_cooling,
    urban_heat_comfort_risk,
    urban_heat_island_intensity,
    urban_heat_vulnerability_index,
    urban_stormwater_peak_runoff,
    wildfire_evacuation_encroachment,
    wildfire_evacuation_front_buffer,
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


def test_seismic_road_blockage_simulation():
    # 3 segments
    coords = np.array(
        [
            [[0, 0], [10, 0]],
            [[10, 0], [20, 0]],
            [[20, 0], [30, 0]],
        ]
    )
    widths = np.array([10.0, 5.0, 8.0])
    heights = np.array([10.0, 20.0, 0.0])
    probs = np.array([0.5, 0.8, 0.1])

    res = seismic_road_blockage_simulation(
        street_segment_coords=coords,
        street_widths_m=widths,
        adjacent_building_heights=heights,
        building_collapse_probabilities=probs,
        debris_expansion_factor=0.5,
    )

    # Segment 0:
    # W_debris = 10 * 0.5 = 5.0
    # B = 5.0 / max(10, 1) = 0.5
    # P_block = 0.5 * min(1.0, 0.5) = 0.25
    # Status: P_block < 0.30 -> open

    # Segment 1:
    # W_debris = 20 * 0.5 = 10.0
    # B = 10.0 / max(5, 1) = 2.0
    # P_block = 0.8 * min(1.0, 2.0) = 0.8
    # Status: B >= 1.0 or P_block >= 0.70 -> blocked

    # Segment 2:
    # W_debris = 0.0
    # B = 0.0
    # P_block = 0.0
    # Status: open

    np.testing.assert_allclose(res["blockage_probabilities"], [0.25, 0.8, 0.0])
    np.testing.assert_allclose(res["debris_extents_m"], [5.0, 10.0, 0.0])
    np.testing.assert_allclose(res["blockage_ratios"], [0.5, 2.0, 0.0])
    assert res["blocked_segments_count"] == 1
    assert res["restricted_segments_count"] == 0
    assert res["open_segments_count"] == 2

    # Test restricted segment (0.30 <= P_block < 0.70 and not blocked)
    # P_block needs to be e.g. 0.5 -> let B = 0.5, probs = 1.0
    # W_debris = 5.0, widths = 10.0 -> B = 0.5
    # probs = 1.0, P_block = 1.0 * 0.5 = 0.5
    coords_r = np.array([[[0, 0], [10, 0]]])
    res_r = seismic_road_blockage_simulation(
        coords_r, np.array([10.0]), np.array([10.0]), np.array([1.0]), 0.5
    )
    assert res_r["restricted_segments_count"] == 1
    assert res_r["blocked_segments_count"] == 0
    assert res_r["open_segments_count"] == 0


def test_seismic_road_blockage_simulation_errors():
    coords = np.zeros((2, 2, 2))
    widths = np.array([10.0, 5.0])
    heights = np.array([10.0, 20.0])
    probs = np.array([0.5, 0.8])

    with pytest.raises(ValueError, match="must have shape"):
        seismic_road_blockage_simulation(np.zeros((2, 2)), widths, heights, probs)

    with pytest.raises(ValueError, match="must have identical length"):
        seismic_road_blockage_simulation(coords, widths, heights[:-1], probs)

    with pytest.raises(ValueError, match="Street widths must be greater than 0"):
        seismic_road_blockage_simulation(coords, np.array([0.0, 5.0]), heights, probs)

    with pytest.raises(ValueError, match="Building heights cannot be negative"):
        seismic_road_blockage_simulation(coords, widths, np.array([-1.0, 20.0]), probs)

    with pytest.raises(ValueError, match="Building collapse probabilities must be in"):
        seismic_road_blockage_simulation(coords, widths, heights, np.array([1.5, 0.8]))


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
    assert np.all(np.isfinite(scores_add_zero))

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


def test_wildfire_evacuation_encroachment():
    slope = np.zeros((5, 5))
    fuel = np.ones((5, 5)) * 0.8

    res = wildfire_evacuation_encroachment(
        fire_origin=(2, 2),
        wind_vector=(5.0, 0.0),
        slope_grid=slope,
        fuel_grid=fuel,
        cell_size=30.0,
        time_steps=5,
    )

    assert "burn_arrival_time" in res
    assert "flame_encroachment_mask" in res
    assert len(res["safe_evacuation_buffer_m"]) == 5
    assert res["burn_arrival_time"][2, 2] == 0.0

    with pytest.raises(ValueError, match="fuel_grid shape must match"):
        wildfire_evacuation_encroachment((2, 2), (5.0, 0.0), slope, fuel[:3, :3])


def test_earthquake_building_collapse_casualty():
    btypes = ["RC", "Masonry", "Steel"]
    stories = np.array([5, 3, 10])
    occ = np.array([50, 20, 200])

    res = earthquake_building_collapse_casualty(btypes, stories, occ, pga_g=0.45)

    assert "collapse_probability" in res
    assert "expected_collapsed_buildings" in res
    assert len(res["collapse_probability"]) == 3
    assert res["estimated_fatalities"] >= 0.0

    with pytest.raises(ValueError, match="equal length"):
        earthquake_building_collapse_casualty(btypes[:1], stories, occ, 0.45)


def test_urban_stormwater_peak_runoff():
    luse = {"roofs": 0.4, "pavement": 0.4, "lawns": 0.2}
    res = urban_stormwater_peak_runoff(
        catchment_area_ha=10.0, land_use_ratios=luse, rainfall_intensity_mm_hr=50.0
    )

    assert "composite_runoff_coefficient" in res
    assert "peak_discharge_m3_s" in res
    assert res["composite_runoff_coefficient"] > 0.5
    assert res["peak_discharge_m3_s"] > 0.0

    with pytest.raises(ValueError, match="positive"):
        urban_stormwater_peak_runoff(0.0, luse, 50.0)


def test_calculate_building_solar_radiation():
    areas = np.array([100.0, 200.0])
    svf = np.array([0.8, 0.5])

    res = calculate_building_solar_radiation(areas, svf, solar_irradiance_kwh_m2=1200.0)

    assert "annual_radiation_kwh" in res
    assert "annual_pv_generation_kwh" in res
    assert res["total_pv_generation_mwh"] > 0.0

    with pytest.raises(ValueError, match="identical length"):
        calculate_building_solar_radiation(areas[:1], svf)


def test_urban_heat_vulnerability_index():
    uhi = np.array([2.0, 5.0, 1.0])
    sens = np.array([500.0, 1200.0, 200.0])
    canopy = np.array([0.4, 0.1, 0.6])

    res = urban_heat_vulnerability_index(uhi, sens, canopy)

    assert "hvi_score" in res
    assert "vulnerability_category" in res
    assert len(res["vulnerability_category"]) == 3


def test_detention_basin_sizing():
    res = detention_basin_sizing(
        catchment_area_ha=5.0,
        cn_pre=65.0,
        cn_post=85.0,
        design_storm_mm=80.0,
    )

    assert "runoff_pre_mm" in res
    assert "runoff_post_mm" in res
    assert "detention_depth_mm" in res
    assert "detention_volume_m3" in res
    assert "peak_inflow_m3_s" in res
    assert res["runoff_post_mm"] > res["runoff_pre_mm"]
    assert res["detention_volume_m3"] > 0.0
    assert res["peak_inflow_m3_s"] > 0.0

    with pytest.raises(ValueError, match="positive"):
        detention_basin_sizing(0.0, 65.0, 85.0, 80.0)

    with pytest.raises(ValueError, match="cn_pre"):
        detention_basin_sizing(5.0, 0.5, 85.0, 80.0)

    with pytest.raises(ValueError, match="cn_post"):
        detention_basin_sizing(5.0, 65.0, 101.0, 80.0)


def test_scs_unit_hydrograph():
    # 1. Normal run with positive runoff
    res = scs_unit_hydrograph(
        watershed_area_km2=10.0,
        curve_number=75.0,
        rainfall_mm=100.0,
        storm_duration_hr=6.0,
        time_of_concentration_hr=2.5,
        dt_minutes=5.0,
    )

    assert "time_minutes" in res
    assert "discharge_m3s" in res
    assert "peak_discharge_m3s" in res
    assert "total_volume_m3" in res
    assert res["peak_discharge_m3s"] > 0.0
    assert res["total_runoff_mm"] > 0.0
    assert res["total_volume_m3"] > 0.0

    # Discharge should start at 0, rise to a peak, and fall to 0
    q = res["discharge_m3s"]
    assert np.isclose(q[0], 0.0)
    assert np.isclose(q[-1], 0.0)
    assert np.isclose(np.max(q), res["peak_discharge_m3s"], rtol=0.05)

    # 2. No runoff case (rainfall <= initial abstraction)
    # For CN=75, S = 25400/75 - 254 = 84.66 mm
    # Ia = 0.2 * S = 16.93 mm
    res_no_runoff = scs_unit_hydrograph(
        watershed_area_km2=10.0,
        curve_number=75.0,
        rainfall_mm=10.0,  # less than 16.93
        storm_duration_hr=6.0,
        time_of_concentration_hr=2.5,
    )
    assert np.isclose(res_no_runoff["total_runoff_mm"], 0.0)
    assert np.isclose(res_no_runoff["peak_discharge_m3s"], 0.0)
    assert np.allclose(res_no_runoff["discharge_m3s"], 0.0)

    # 3. Validation errors
    with pytest.raises(ValueError, match="watershed_area_km2"):
        scs_unit_hydrograph(-5.0, 75.0, 100.0, 6.0, 2.5)

    with pytest.raises(ValueError, match="curve_number"):
        scs_unit_hydrograph(10.0, 150.0, 100.0, 6.0, 2.5)

    with pytest.raises(ValueError, match="rainfall_mm"):
        scs_unit_hydrograph(10.0, 75.0, -10.0, 6.0, 2.5)

    with pytest.raises(ValueError, match="storm_duration_hr"):
        scs_unit_hydrograph(10.0, 75.0, 100.0, 0.0, 2.5)

    with pytest.raises(ValueError, match="time_of_concentration_hr"):
        scs_unit_hydrograph(10.0, 75.0, 100.0, 6.0, 0.0)

    with pytest.raises(ValueError, match="dt_minutes"):
        scs_unit_hydrograph(10.0, 75.0, 100.0, 6.0, 2.5, dt_minutes=-5.0)

    with pytest.raises(ValueError, match="peak_rate_factor"):
        scs_unit_hydrograph(10.0, 75.0, 100.0, 6.0, 2.5, peak_rate_factor=0.0)


def test_compound_hazard_cascade():
    # 2 locations, 3 hazards
    hazard_intensities = np.array([[10.0, 0.0, 0.0], [5.0, 2.0, 0.0]])
    trigger_thresholds = np.array([8.0, 5.0, 10.0])

    # Hazard 0 amplifies Hazard 1 by 0.5
    # Hazard 1 amplifies Hazard 2 by 1.0
    amplification_matrix = np.array([[0.0, 0.5, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])

    vulnerability = np.array([0.8, 0.5])

    res = compound_hazard_cascade(
        hazard_intensities,
        trigger_thresholds,
        amplification_matrix,
        vulnerability,
        max_iterations=10,
    )

    # Location 0: H0=10.0 > 8.0, so triggers H1.
    # New H1 = 0 + 10.0 * 0.5 = 5.0.
    # H1 is exactly 5.0, which is NOT > 5.0 (triggered = current > threshold).
    # Since max_iterations=10, the cascade grows repeatedly due to the algorithm logic:
    # Final intensities: [10.0, 50.0, 220.0]

    # Location 1: H0=5.0 (not > 8.0). H1=2.0 (not > 5.0).
    # Final intensities: [5.0, 2.0, 0.0]

    np.testing.assert_allclose(res["final_intensities"], [[10.0, 50.0, 220.0], [5.0, 2.0, 0.0]])

    # Check total intensity
    np.testing.assert_allclose(res["total_intensity"], [280.0, 7.0])

    # Check peak hazard
    np.testing.assert_array_equal(res["peak_hazard"], [2, 0])

    # Check damage index: vuln * (1 - exp(-total_intensity))
    expected_damage = vulnerability * (1.0 - np.exp(-res["total_intensity"]))
    np.testing.assert_allclose(res["damage_index"], expected_damage)

    # Check amplification ratio
    eps = 1e-12
    expected_ratio = res["final_intensities"] / np.maximum(hazard_intensities, eps)
    np.testing.assert_allclose(res["amplification_ratio"], expected_ratio)

    # Test invalid inputs
    with pytest.raises(ValueError):
        compound_hazard_cascade(
            hazard_intensities, trigger_thresholds[:-1], amplification_matrix, vulnerability
        )
    with pytest.raises(ValueError):
        compound_hazard_cascade(
            hazard_intensities, trigger_thresholds, amplification_matrix, vulnerability[:-1]
        )
    with pytest.raises(ValueError):
        invalid_amp = np.ones((3, 3))  # Diagonal not zero
        compound_hazard_cascade(hazard_intensities, trigger_thresholds, invalid_amp, vulnerability)


def test_seismic_damage_loss_curve():
    pga_values = np.array([0.0, 0.1, 0.5, 1.0])
    building_counts = np.array([10, 20])
    replacement_values = np.array([100000, 200000])

    res = seismic_damage_loss_curve(
        pga_values, building_counts, replacement_values, building_type="c2_medium"
    )

    assert "pga_values" in res
    assert "damage_state_probabilities" in res
    assert "expected_loss_ratio" in res
    assert "total_economic_loss" in res
    assert "building_collapse_count" in res

    np.testing.assert_allclose(res["pga_values"], pga_values)

    # For PGA=0, everything should be 0 except P_none
    np.testing.assert_allclose(res["damage_state_probabilities"][0, :], [1.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(res["expected_loss_ratio"][0], 0.0)
    np.testing.assert_allclose(res["total_economic_loss"][0], 0.0)
    np.testing.assert_allclose(res["building_collapse_count"][0], 0.0)

    # For high PGA, loss ratio should be close to 1
    assert res["expected_loss_ratio"][-1] > 0.5
    assert res["total_economic_loss"][-1] > 0

    # Total loss bounds check
    total_value = np.sum(building_counts * replacement_values)
    assert np.all(res["total_economic_loss"] >= 0)
    assert np.all(res["total_economic_loss"] <= total_value)

    # Test error cases
    with pytest.raises(ValueError, match="cannot be negative"):
        seismic_damage_loss_curve(np.array([-0.1, 0.5]), building_counts, replacement_values)

    with pytest.raises(ValueError, match="cannot be negative"):
        seismic_damage_loss_curve(pga_values, np.array([-10, 20]), replacement_values)

    with pytest.raises(ValueError, match="Unknown building_type"):
        seismic_damage_loss_curve(
            pga_values, building_counts, replacement_values, building_type="unknown"
        )


def test_dynamic_evacuation_bottlenecks():
    import numpy as np
    import pytest

    from planx.resilience import dynamic_evacuation_bottlenecks

    origin_demands = np.array([100.0, 50.0, 0.0])
    destination_capacities = np.array([0.0, 0.0, 200.0])
    edge_list = np.array([[0, 1], [1, 2]])
    edge_capacities = np.array([30.0, 40.0])
    edge_free_flow_times = np.array([1.0, 1.0])
    res = dynamic_evacuation_bottlenecks(
        origin_demands,
        destination_capacities,
        edge_list,
        edge_capacities,
        edge_free_flow_times,
        time_horizon_steps=10,
    )
    assert "total_evacuated" in res
    assert res["total_evacuated"] > 0
    assert "clearance_time_step" in res
    assert "edge_max_vcr" in res
    assert "edge_total_queues" in res
    assert "critical_bottlenecks" in res
    assert "time_series_evacuated" in res
    with pytest.raises(ValueError):
        dynamic_evacuation_bottlenecks(
            origin_demands[:-1],
            destination_capacities,
            edge_list,
            edge_capacities,
            edge_free_flow_times,
        )


def test_optimize_tree_canopy_greening():
    import numpy as np
    import pytest

    from planx.resilience import optimize_tree_canopy_greening

    lst = np.array([35.0, 36.0, 32.0, 38.0])
    aqi = np.array([50.0, 60.0, 40.0, 80.0])
    ped = np.array([100.0, 200.0, 50.0, 300.0])
    canopy = np.array([0.1, 0.0, 0.5, 0.0])
    coords = np.array([[0, 0], [0, 50], [0, 100], [0, 150]])

    # Normal case
    res = optimize_tree_canopy_greening(
        lst, aqi, ped, canopy, coords, budget_max_trees=2, cooling_radius=100.0
    )
    assert "selected_indices" in res
    assert "selected_coords" in res
    assert "total_heat_reduction_score" in res
    assert "priority_scores" in res
    assert "post_greening_heat_mitigation" in res

    assert len(res["selected_indices"]) == 2
    assert res["selected_coords"].shape == (2, 2)
    assert res["total_heat_reduction_score"] > 0
    assert len(res["priority_scores"]) == 4
    assert len(res["post_greening_heat_mitigation"]) == 4

    # Test budget larger than candidates
    res_large_budget = optimize_tree_canopy_greening(
        lst, aqi, ped, canopy, coords, budget_max_trees=10, cooling_radius=100.0
    )
    assert len(res_large_budget["selected_indices"]) <= 4

    # Test error cases
    with pytest.raises(ValueError, match="Input arrays must have matching shape"):
        optimize_tree_canopy_greening(lst[:-1], aqi, ped, canopy, coords, 2)

    with pytest.raises(ValueError, match="existing_canopy_ratio must be between 0 and 1"):
        bad_canopy = np.array([0.1, -0.2, 0.5, 1.2])
        optimize_tree_canopy_greening(lst, aqi, ped, bad_canopy, coords, 2)

    with pytest.raises(ValueError, match="budget_max_trees must be greater than 0"):
        optimize_tree_canopy_greening(lst, aqi, ped, canopy, coords, 0)

    with pytest.raises(ValueError, match="cooling_radius must be positive"):
        optimize_tree_canopy_greening(lst, aqi, ped, canopy, coords, 2, cooling_radius=0.0)


# ---------------------------------------------------------------------------
# Tests for stormwater_retention_basin_design
# ---------------------------------------------------------------------------


def test_stormwater_retention_basin_design():
    # Basic valid inputs
    res = stormwater_retention_basin_design(
        drainage_area_ha=1.0,  # 10,000 m2
        impervious_ratio=0.5,  # C = 0.05 + 0.9*0.5 = 0.5
        rainfall_depth_mm=50.0,  # R = 0.5 * 50 = 25 mm = 0.025 m
        soil_infiltration_rate_mmh=10.0,  # 10 mm/h
        max_allowable_drain_hours=48.0,
        basin_safety_factor=1.2,
    )
    # V_raw = 0.025 * 10000 = 250 m3
    # V_design = 250 * 1.2 = 300 m3
    # D_max = (10 / 1000) * 48 = 0.48 m
    # A_basin = 300 / 0.48 = 625 m2
    # t_drain = (0.48 * 1000) / 10 = 48.0 h
    assert np.isclose(res["runoff_volume_m3"], 250.0)
    assert np.isclose(res["design_storage_volume_m3"], 300.0)
    assert np.isclose(res["max_basin_depth_m"], 0.48)
    assert np.isclose(res["min_basin_surface_area_m2"], 625.0)
    assert np.isclose(res["actual_draindown_hours"], 48.0)
    assert np.isclose(res["runoff_coefficient"], 0.5)
    assert res["is_drain_time_compliant"] is True


def test_stormwater_retention_basin_design_errors():
    with pytest.raises(ValueError, match="drainage_area_ha must be positive"):
        stormwater_retention_basin_design(-1.0, 0.5, 50.0, 10.0)

    with pytest.raises(ValueError, match="impervious_ratio must be between 0 and 1"):
        stormwater_retention_basin_design(1.0, 1.5, 50.0, 10.0)

    with pytest.raises(ValueError, match="rainfall_depth_mm must be positive"):
        stormwater_retention_basin_design(1.0, 0.5, -50.0, 10.0)

    with pytest.raises(ValueError, match="soil_infiltration_rate_mmh must be positive"):
        stormwater_retention_basin_design(1.0, 0.5, 50.0, 0.0)


def test_stormwater_retention_basin_design_edge_cases():
    # Zero impervious ratio
    res = stormwater_retention_basin_design(1.0, 0.0, 50.0, 10.0)
    assert np.isclose(res["runoff_coefficient"], 0.05)

    # 1.0 impervious ratio
    res2 = stormwater_retention_basin_design(1.0, 1.0, 50.0, 10.0)
    assert np.isclose(res2["runoff_coefficient"], 0.95)


def test_wildfire_evacuation_front_buffer():
    ignition_coords = np.array([10.0, 20.0])
    res = wildfire_evacuation_front_buffer(
        ignition_coords=ignition_coords,
        wind_speed_kmh=20.0,
        wind_direction_deg=90.0,
        terrain_slope_deg=np.array([10.0]),
        time_elapsed_hours=2.0,
        buffer_safety_factor=1.5,
    )

    assert isinstance(res, dict)
    assert "forward_rate_of_spread_m_min" in res
    assert "forward_distance_m" in res
    assert "flank_distance_m" in res
    assert "safety_buffer_distance_m" in res
    assert "fire_ellipse_axes" in res

    assert res["forward_rate_of_spread_m_min"] > 0
    assert res["forward_distance_m"] > 0
    assert res["flank_distance_m"] > 0
    np.testing.assert_allclose(res["safety_buffer_distance_m"], res["forward_distance_m"] * 1.5)

    axes = res["fire_ellipse_axes"]
    assert axes["semi_major_m"] > 0
    assert axes["semi_minor_m"] > 0


def test_wildfire_evacuation_front_buffer_validation():
    ignition = np.array([10.0, 20.0])
    slope = np.array([10.0])

    with pytest.raises(ValueError, match="must be non-negative"):
        wildfire_evacuation_front_buffer(ignition, -10.0, 90.0, slope, 2.0)

    with pytest.raises(ValueError, match="time_elapsed_hours must be greater than 0"):
        wildfire_evacuation_front_buffer(ignition, 20.0, 90.0, slope, 0.0)

    with pytest.raises(ValueError, match="must be in"):
        wildfire_evacuation_front_buffer(ignition, 20.0, 400.0, slope, 2.0)

    with pytest.raises(ValueError, match="buffer_safety_factor must be positive"):
        wildfire_evacuation_front_buffer(ignition, 20.0, 90.0, slope, 2.0, -1.0)

    with pytest.raises(ValueError, match="ignition_coords must be of shape"):
        wildfire_evacuation_front_buffer(np.array([10.0]), 20.0, 90.0, slope, 2.0)


def test_coastal_storm_surge_inundation_engine_normal():
    from planx.resilience import coastal_storm_surge_inundation_engine

    dem = np.array([[0.0, 1.0, 3.0], [0.5, 2.0, 5.0], [1.5, 4.0, 6.0]])
    c_mask = np.array([[True, False, False], [False, False, False], [False, False, False]])

    res = coastal_storm_surge_inundation_engine(
        dem_grid=dem,
        coastal_mask=c_mask,
        storm_surge_m=2.5,
        sea_level_rise_m=0.5,
        cell_size_m=10.0,
    )

    assert "inundation_depth" in res
    assert "inundated_area_m2" in res
    assert "max_depth_m" in res
    assert "mean_depth_m" in res
    assert "volume_m3" in res
    assert "connectivity_mask" in res
    assert "hazard_classification_counts" in res

    assert res["inundation_depth"].shape == (3, 3)
    assert res["inundated_area_m2"] > 0
    assert res["max_depth_m"] > 0
    assert bool(res["connectivity_mask"][0, 0]) is True
    assert bool(res["connectivity_mask"][0, 1]) is True


def test_coastal_storm_surge_inundation_engine_validation():
    from planx.resilience import coastal_storm_surge_inundation_engine

    dem = np.zeros((3, 3))
    c_mask = np.zeros((3, 3), dtype=bool)

    with pytest.raises(ValueError, match="dem_grid must be a 2D array"):
        coastal_storm_surge_inundation_engine(np.zeros(3), c_mask, 2.0)

    with pytest.raises(ValueError, match="coastal_mask shape must match"):
        coastal_storm_surge_inundation_engine(dem, c_mask[:2, :2], 2.0)

    with pytest.raises(ValueError, match="storm_surge_m must be non-negative"):
        coastal_storm_surge_inundation_engine(dem, c_mask, -1.0)


def test_surface_cool_island_simulator_normal():
    from planx.resilience import surface_cool_island_simulator

    alb_base = np.array([[0.1, 0.2], [0.15, 0.2]])
    alb_targ = np.array([[0.5, 0.5], [0.4, 0.2]])
    g_frac = np.array([[0.2, 0.0], [0.1, 0.5]])

    res = surface_cool_island_simulator(
        albedo_grid=alb_base,
        target_albedo_grid=alb_targ,
        solar_irradiance_wm2=800.0,
        ambient_temp_c=35.0,
        cell_size_m=10.0,
        green_fraction_grid=g_frac,
    )

    assert "lst_reduction_c" in res
    assert "new_lst_grid" in res
    assert "net_radiation_change_wm2" in res
    assert "mean_cooling_c" in res
    assert "max_cooling_c" in res
    assert "total_heat_mitigated_mwh" in res
    assert "pet_comfort_improvement_c" in res

    assert res["lst_reduction_c"].shape == (2, 2)
    assert res["mean_cooling_c"] > 0.0
    assert res["total_heat_mitigated_mwh"] > 0.0


def test_surface_cool_island_simulator_validation():
    from planx.resilience import surface_cool_island_simulator

    alb_base = np.array([[0.1, 0.2], [0.15, 0.2]])

    with pytest.raises(ValueError, match="albedo_grid values must be between"):
        surface_cool_island_simulator(np.array([[-0.1, 0.2], [0.1, 0.2]]), np.zeros((2, 2)))


def test_wind_canopy_aerodynamic_drag_simulator_normal():
    from planx.resilience import wind_canopy_aerodynamic_drag_simulator

    lai = np.array([[2.0, 4.0], [1.0, 0.5]])
    lambda_f = np.array([[0.2, 0.4], [0.1, 0.05]])

    res = wind_canopy_aerodynamic_drag_simulator(
        inflow_wind_speed_ms=5.0,
        tree_lai_grid=lai,
        building_frontal_density_grid=lambda_f,
        tree_height_m=10.0,
        drag_coefficient=0.2,
    )

    assert "pedestrian_wind_speed_ms" in res
    assert "attenuation_ratio" in res
    assert "comfort_category_grid" in res
    assert "mean_wind_speed_ms" in res
    assert "max_wind_speed_ms" in res
    assert "comfortable_area_ratio" in res

    assert res["pedestrian_wind_speed_ms"].shape == (2, 2)
    assert res["comfort_category_grid"].shape == (2, 2)
    assert res["mean_wind_speed_ms"] < 5.0
    assert 0.0 <= res["comfortable_area_ratio"] <= 1.0


def test_pluvial_flash_flood_simulator_normal():
    from planx.resilience import pluvial_flash_flood_simulator

    dem = np.array([[100.0, 95.0], [90.0, 85.0]])
    cn = np.array([[80.0, 85.0], [90.0, 95.0]])

    res = pluvial_flash_flood_simulator(dem, cn, rainfall_mm=100.0, pipe_capacity_mm=25.0)

    assert "runoff_depth_grid" in res
    assert "ponding_depth_grid" in res
    assert "max_ponding_depth_mm" in res
    assert "hazard_level" in res
    assert res["runoff_depth_grid"].shape == (2, 2)


def test_heatwave_health_vulnerability_engine():
    from planx.resilience import heatwave_health_vulnerability_engine

    t = np.array([[30.0, 35.0], [28.0, 38.0]])
    rh = np.array([[60.0, 70.0], [50.0, 80.0]])
    vuln = np.array([[0.2, 0.4], [0.1, 0.5]])
    ac = np.array([[0.8, 0.4], [0.9, 0.2]])

    res = heatwave_health_vulnerability_engine(t, rh, vuln, ac)

    assert "heat_index_c_grid" in res
    assert "vulnerability_score_grid" in res
    assert "mean_heat_index_c" in res
    assert "alert_level" in res
    assert res["heat_index_c_grid"].shape == (2, 2)





