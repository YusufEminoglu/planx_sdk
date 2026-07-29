import numpy as np
import pytest
from scipy.sparse import csr_matrix

from planx.spatial.accessibility import (
    network_voronoi_allocation,
    transit_frequency_accessibility,
)


def test_transit_frequency_accessibility_basic():
    demand = np.array([[0.0, 0.0], [1000.0, 1000.0]])
    stops = np.array([[100.0, 0.0], [900.0, 1000.0]])
    headways = np.array([10.0, 5.0])
    routes = np.array([1, 3])

    res = transit_frequency_accessibility(
        demand_coords=demand,
        stop_coords=stops,
        headways_minutes=headways,
        num_routes=routes,
        catchment_radius=800.0,
        decay_function="linear",
        headway_benchmark=10.0,
        route_diversity_weight=0.3,
    )

    assert "accessibility_index" in res
    assert "num_stops_in_catchment" in res
    assert "nearest_stop_distance" in res
    assert "mean_headway_in_catchment" in res

    assert res["accessibility_index"].shape == (2,)
    assert res["num_stops_in_catchment"].shape == (2,)

    np.testing.assert_allclose(res["nearest_stop_distance"], [100.0, 100.0])
    np.testing.assert_array_equal(res["num_stops_in_catchment"], [1, 1])


def test_transit_frequency_accessibility_decay_methods():
    demand = np.array([[0.0, 0.0]])
    stops = np.array([[200.0, 0.0]])
    headways = np.array([10.0])
    routes = np.array([2])

    res_gauss = transit_frequency_accessibility(
        demand, stops, headways, routes, decay_function="gaussian"
    )
    res_exp = transit_frequency_accessibility(
        demand, stops, headways, routes, decay_function="exponential"
    )
    res_lin = transit_frequency_accessibility(
        demand, stops, headways, routes, decay_function="linear"
    )

    assert res_gauss["accessibility_index"][0] > 0
    assert res_exp["accessibility_index"][0] > 0
    assert res_lin["accessibility_index"][0] > 0


def test_transit_frequency_accessibility_no_stops_in_catchment():
    demand = np.array([[0.0, 0.0]])
    stops = np.array([[2000.0, 0.0]])
    headways = np.array([10.0])
    routes = np.array([2])

    res = transit_frequency_accessibility(demand, stops, headways, routes, catchment_radius=800.0)

    assert res["num_stops_in_catchment"][0] == 0
    assert res["accessibility_index"][0] == 0.0
    assert np.isnan(res["mean_headway_in_catchment"][0])
    np.testing.assert_allclose(res["nearest_stop_distance"][0], 2000.0)


def test_transit_frequency_accessibility_empty_stops():
    demand = np.array([[0.0, 0.0]])
    stops = np.empty((0, 2))
    headways = np.empty((0,))
    routes = np.empty((0,))

    res = transit_frequency_accessibility(demand, stops, headways, routes)

    assert res["num_stops_in_catchment"][0] == 0
    assert res["accessibility_index"][0] == 0.0
    assert np.isnan(res["mean_headway_in_catchment"][0])
    assert np.isinf(res["nearest_stop_distance"][0])


def test_transit_frequency_accessibility_validation():
    demand = np.array([[0.0, 0.0]])
    stops = np.array([[100.0, 0.0]])
    headways = np.array([10.0])
    routes = np.array([1])

    # Test bad shapes
    with pytest.raises(ValueError):
        transit_frequency_accessibility(np.array([0.0, 0.0]), stops, headways, routes)

    with pytest.raises(ValueError):
        transit_frequency_accessibility(demand, np.array([100.0, 0.0]), headways, routes)

    with pytest.raises(ValueError):
        transit_frequency_accessibility(demand, stops, np.array([10.0, 20.0]), routes)

    with pytest.raises(ValueError):
        transit_frequency_accessibility(demand, stops, headways, np.array([1, 2]))

    # Test invalid values
    with pytest.raises(ValueError):
        transit_frequency_accessibility(demand, stops, np.array([-5.0]), routes)

    with pytest.raises(ValueError):
        transit_frequency_accessibility(demand, stops, headways, np.array([0]))

    with pytest.raises(ValueError):
        transit_frequency_accessibility(demand, stops, headways, routes, catchment_radius=0)

    with pytest.raises(ValueError):
        transit_frequency_accessibility(demand, stops, headways, routes, headway_benchmark=-1.0)

    with pytest.raises(ValueError):
        transit_frequency_accessibility(demand, stops, headways, routes, route_diversity_weight=1.5)

    with pytest.raises(ValueError):
        transit_frequency_accessibility(demand, stops, headways, routes, decay_function="unknown")


def test_network_voronoi_allocation_basic():
    # 3-node linear graph: 0 --(10)-- 1 --(5)-- 2
    adj = np.array([[0, 10, 0], [10, 0, 5], [0, 5, 0]], dtype=float)
    graph_sparse = csr_matrix(adj)

    # Facilities at node 0 and 2
    facility_indices = np.array([0, 2])

    # uniform demand
    res = network_voronoi_allocation(graph_sparse, facility_indices)

    np.testing.assert_array_equal(res["assigned_facility"], [0, 1, 1])
    np.testing.assert_allclose(res["travel_cost"], [0, 5, 0])
    np.testing.assert_array_equal(res["facility_node_count"], [1, 2])
    np.testing.assert_allclose(res["facility_demand"], [1, 2])
    np.testing.assert_allclose(res["facility_mean_cost"], [0.0, 2.5])
    np.testing.assert_allclose(res["facility_max_cost"], [0.0, 5.0])
    assert res["coverage_ratio"] == 1.0


def test_network_voronoi_allocation_cutoff():
    # 4-node linear graph: 0 --(10)-- 1 --(10)-- 2 --(10)-- 3
    adj = np.array([[0, 10, 0, 0], [10, 0, 10, 0], [0, 10, 0, 10], [0, 0, 10, 0]], dtype=float)

    facility_indices = np.array([0])
    demand = np.array([10, 20, 30, 40])

    res = network_voronoi_allocation(
        adj, facility_indices, demand_values=demand, impedance_cutoff=15.0
    )

    # Node 0, 1 within 15. Node 2 is at 20, Node 3 is at 30.
    np.testing.assert_array_equal(res["assigned_facility"], [0, 0, -1, -1])

    assert res["travel_cost"][0] == 0.0
    assert res["travel_cost"][1] == 10.0
    assert np.isinf(res["travel_cost"][2])

    np.testing.assert_array_equal(res["facility_node_count"], [2])
    np.testing.assert_allclose(res["facility_demand"], [30.0])  # 10 + 20
    np.testing.assert_allclose(res["facility_max_cost"], [10.0])
    assert res["coverage_ratio"] == 0.3  # 30 / 100


def test_network_voronoi_allocation_validation():
    adj = np.eye(3)
    fac = np.array([0])

    with pytest.raises(ValueError):
        network_voronoi_allocation(np.array([1, 2]), fac)

    with pytest.raises(ValueError):
        network_voronoi_allocation(adj, np.array([[0], [1]]))

    with pytest.raises(ValueError):
        network_voronoi_allocation(adj, np.array([]))

    with pytest.raises(ValueError):
        network_voronoi_allocation(adj, np.array([-1]))

    with pytest.raises(ValueError):
        network_voronoi_allocation(adj, fac, demand_values=np.array([1, 2]))

    with pytest.raises(ValueError):
        network_voronoi_allocation(adj, fac, impedance_cutoff=-5.0)


def test_healthcare_equity_index_basic():
    from planx.spatial.accessibility import healthcare_equity_index

    demand = np.array([[0.0, 0.0], [0.0, 1000.0]])
    facs = np.array([[0.0, 500.0]])
    caps = np.array([100.0])
    pops = np.array([[50.0, 50.0], [20.0, 80.0]])
    weights = np.array([1.0, 2.0])

    res = healthcare_equity_index(demand, facs, caps, pops, weights, 2000.0)

    assert "accessibility_scores" in res
    assert "weighted_accessibility" in res
    assert "gini_coefficient" in res
    assert "group_accessibility_mean" in res
    assert "group_deficit" in res
    assert "equity_index" in res

    np.testing.assert_allclose(res["accessibility_scores"], [0.5, 0.5])
    assert res["gini_coefficient"] == 0.0
    assert res["equity_index"] == 1.0


def test_healthcare_equity_index_validation():
    from planx.spatial.accessibility import healthcare_equity_index

    demand = np.array([[0.0, 0.0]])
    facs = np.array([[0.0, 500.0]])
    caps = np.array([100.0])
    pops = np.array([[50.0, 50.0]])
    weights = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="demand_coords must be of shape"):
        healthcare_equity_index(np.array([0.0, 0.0]), facs, caps, pops, weights)

    with pytest.raises(ValueError, match="facility_coords must be of shape"):
        healthcare_equity_index(demand, np.array([0.0]), caps, pops, weights)

    with pytest.raises(ValueError, match="facility_capacities must be positive"):
        healthcare_equity_index(demand, facs, np.array([-10.0]), pops, weights)

    with pytest.raises(ValueError, match="population_groups must be of shape"):
        healthcare_equity_index(demand, facs, caps, np.array([50.0, 50.0]), weights)

    with pytest.raises(ValueError, match="catchment_distance must be positive"):
        healthcare_equity_index(demand, facs, caps, pops, weights, catchment_distance=-10.0)


def test_calculate_multimodal_15m_city_basic():
    from planx.spatial.accessibility import calculate_multimodal_15m_city

    demand = np.array([[0.0, 0.0], [5000.0, 0.0]])
    amenities = {
        "grocery": np.array([[1000.0, 0.0], [6000.0, 0.0]]),  # dist: 1km
        "school": np.array([[2000.0, 0.0], [7000.0, 0.0]]),  # dist: 2km
    }

    res = calculate_multimodal_15m_city(demand, amenities)

    assert "city_15m_score" in res
    assert "category_scores" in res
    assert "gini_equity_score" in res
    assert "threshold_compliance_pct" in res

    assert res["city_15m_score"].shape == (2,)
    assert "grocery" in res["category_scores"]
    assert "school" in res["category_scores"]
    assert res["category_scores"]["grocery"].shape == (2,)


def test_calculate_multimodal_15m_city_validation():
    from planx.spatial.accessibility import calculate_multimodal_15m_city

    demand = np.array([[0.0, 0.0]])
    amenities = {"grocery": np.array([[100.0, 0.0]])}

    with pytest.raises(ValueError, match="demand_coords must be a 2D array"):
        calculate_multimodal_15m_city(np.array([0.0, 0.0]), amenities)

    with pytest.raises(ValueError, match="demand_coords cannot be empty"):
        calculate_multimodal_15m_city(np.empty((0, 2)), amenities)

    with pytest.raises(ValueError, match="amenity_coords_dict cannot be empty"):
        calculate_multimodal_15m_city(demand, {})

    with pytest.raises(ValueError, match="Speed for mode walk must be positive"):
        calculate_multimodal_15m_city(demand, amenities, modal_speeds_kmh={"walk": -1.0})

    with pytest.raises(ValueError, match="Modal weights must sum to 1.0"):
        calculate_multimodal_15m_city(demand, amenities, modal_weights={"walk": 0.5})

    with pytest.raises(ValueError, match="No common modes"):
        calculate_multimodal_15m_city(
            demand, amenities, modal_speeds_kmh={"walk": 5.0}, modal_weights={"bike": 1.0}
        )

    with pytest.raises(ValueError, match="Coordinates for category 'grocery' must be shape"):
        calculate_multimodal_15m_city(demand, {"grocery": np.array([0.0, 0.0])})

    with pytest.raises(ValueError, match="Coordinates for category 'grocery' cannot be empty"):
        calculate_multimodal_15m_city(demand, {"grocery": np.empty((0, 2))})
