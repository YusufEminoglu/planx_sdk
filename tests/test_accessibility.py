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


def test_huff_retail_market_share_basic():
    from planx.spatial.accessibility import huff_retail_market_share

    origins = np.array([[0.0, 0.0], [10.0, 0.0]])
    stores = np.array([[5.0, 0.0], [20.0, 0.0]])
    attr = np.array([100.0, 200.0])
    pops = np.array([1000.0, 500.0])

    res = huff_retail_market_share(origins, stores, attr, pops)

    assert "probability_matrix" in res
    assert "store_captured_customers" in res
    assert "store_market_shares" in res
    assert "trade_area_zone_counts" in res

    assert res["probability_matrix"].shape == (2, 2)
    assert res["store_captured_customers"].shape == (2,)
    assert res["store_market_shares"].shape == (2,)
    assert res["trade_area_zone_counts"].shape == (2,)

    # P sums to 1 for each origin
    np.testing.assert_allclose(np.sum(res["probability_matrix"], axis=1), [1.0, 1.0])

    # total captured customers should equal total population
    np.testing.assert_allclose(np.sum(res["store_captured_customers"]), 1500.0)

    # total market share should equal 1.0
    np.testing.assert_allclose(np.sum(res["store_market_shares"]), 1.0)


def test_huff_retail_market_share_validation():
    from planx.spatial.accessibility import huff_retail_market_share

    origins = np.array([[0.0, 0.0]])
    stores = np.array([[10.0, 0.0]])
    attr = np.array([100.0])

    with pytest.raises(ValueError, match="origin_coords must be a 2D array"):
        huff_retail_market_share(np.array([0.0, 0.0]), stores, attr)

    with pytest.raises(ValueError, match="store_coords must be a 2D array"):
        huff_retail_market_share(origins, np.array([10.0, 0.0]), attr)

    with pytest.raises(ValueError, match="store_attractiveness length"):
        huff_retail_market_share(origins, stores, np.array([100.0, 200.0]))

    with pytest.raises(ValueError, match="store_attractiveness must be > 0"):
        huff_retail_market_share(origins, stores, np.array([-10.0]))

    with pytest.raises(ValueError, match="distance_exponent must be > 0"):
        huff_retail_market_share(origins, stores, attr, distance_exponent=0.0)

    with pytest.raises(ValueError, match="origin_populations length"):
        huff_retail_market_share(origins, stores, attr, origin_populations=np.array([10.0, 20.0]))

    with pytest.raises(ValueError, match="origin_populations must be non-negative"):
        huff_retail_market_share(origins, stores, attr, origin_populations=np.array([-10.0]))


def test_huff_retail_market_share_empty():
    from planx.spatial.accessibility import huff_retail_market_share

    origins = np.empty((0, 2))
    stores = np.empty((0, 2))
    attr = np.empty((0,))

    res = huff_retail_market_share(origins, stores, attr)
    assert res["probability_matrix"].shape == (0, 0)
    assert res["store_captured_customers"].shape == (0,)
    assert res["store_market_shares"].shape == (0,)
    assert res["trade_area_zone_counts"].shape == (0,)


def test_parking_spatial_mismatch_index_basic():
    from planx.spatial.accessibility import parking_spatial_mismatch_index

    demand_coords = np.array([[0.0, 0.0], [1000.0, 0.0]])
    facility_coords = np.array([[100.0, 0.0], [900.0, 0.0]])
    caps = np.array([50.0, 150.0])
    demand = np.array([100.0, 100.0])

    res = parking_spatial_mismatch_index(
        demand_coords=demand_coords,
        parking_facility_coords=facility_coords,
        parking_capacities=caps,
        zone_parking_demand=demand,
        walk_threshold_m=400.0,
    )

    assert "mismatch_ratios" in res
    assert "reachable_supply" in res
    assert "deficit_zones_count" in res
    assert "surplus_zones_count" in res
    assert "total_parking_deficit" in res
    assert "mismatch_gini" in res

    assert res["mismatch_ratios"].shape == (2,)
    assert res["reachable_supply"].shape == (2,)

    # Distances:
    # Zone 0 to Fac 0: 100m. W = 1 - (100/400)^2 = 1 - 1/16 = 15/16 = 0.9375
    # Zone 0 to Fac 1: 900m. W = 0
    # Zone 1 to Fac 0: 900m. W = 0
    # Zone 1 to Fac 1: 100m. W = 0.9375

    expected_S0 = 0.9375 * 50.0
    expected_S1 = 0.9375 * 150.0

    np.testing.assert_allclose(res["reachable_supply"], [expected_S0, expected_S1])
    np.testing.assert_allclose(res["mismatch_ratios"], [expected_S0 / 100.0, expected_S1 / 100.0])

    assert res["deficit_zones_count"] == 1  # Zone 0 is < 1
    assert res["surplus_zones_count"] == 1  # Zone 1 is >= 1

    expected_deficit = max(0.0, 100.0 - expected_S0) + max(0.0, 100.0 - expected_S1)
    np.testing.assert_allclose(res["total_parking_deficit"], expected_deficit)

    assert 0.0 <= res["mismatch_gini"] <= 1.0


def test_parking_spatial_mismatch_index_validation():
    from planx.spatial.accessibility import parking_spatial_mismatch_index

    demand_coords = np.array([[0.0, 0.0]])
    facility_coords = np.array([[100.0, 0.0]])
    caps = np.array([50.0])
    demand = np.array([100.0])

    with pytest.raises(ValueError, match="demand_coords must be a 2D array"):
        parking_spatial_mismatch_index(np.array([0.0, 0.0]), facility_coords, caps, demand)

    with pytest.raises(ValueError, match="parking_facility_coords must be a 2D array"):
        parking_spatial_mismatch_index(demand_coords, np.array([100.0, 0.0]), caps, demand)

    with pytest.raises(ValueError, match="parking_capacities length"):
        parking_spatial_mismatch_index(
            demand_coords, facility_coords, np.array([50.0, 50.0]), demand
        )

    with pytest.raises(ValueError, match="parking_capacities must be positive"):
        parking_spatial_mismatch_index(demand_coords, facility_coords, np.array([0.0]), demand)

    with pytest.raises(ValueError, match="zone_parking_demand length"):
        parking_spatial_mismatch_index(
            demand_coords, facility_coords, caps, np.array([100.0, 100.0])
        )

    with pytest.raises(ValueError, match="zone_parking_demand must be positive"):
        parking_spatial_mismatch_index(demand_coords, facility_coords, caps, np.array([-10.0]))

    with pytest.raises(ValueError, match="walk_threshold_m must be positive"):
        parking_spatial_mismatch_index(
            demand_coords, facility_coords, caps, demand, walk_threshold_m=-10.0
        )


def test_parking_spatial_mismatch_index_empty():
    from planx.spatial.accessibility import parking_spatial_mismatch_index

    res1 = parking_spatial_mismatch_index(
        np.empty((0, 2)), np.empty((0, 2)), np.empty((0,)), np.empty((0,))
    )
    assert res1["mismatch_ratios"].shape == (0,)
    assert res1["total_parking_deficit"] == 0.0

    res2 = parking_spatial_mismatch_index(
        np.array([[0.0, 0.0]]), np.empty((0, 2)), np.empty((0,)), np.array([100.0])
    )
    assert res2["deficit_zones_count"] == 1
    assert res2["total_parking_deficit"] == 100.0


def test_ev_charging_accessibility_index_normal():
    from planx.spatial import ev_charging_accessibility_index

    z_dem = np.array([10.0, 20.0, 30.0])
    z_xy = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    s_xy = np.array([[0.5, 0.5], [1.5, 1.5]])
    s_kw = np.array([100.0, 150.0])
    s_types = np.array([0, 1])
    t_cap = np.array([80.0, 120.0])

    res = ev_charging_accessibility_index(
        zone_demand=z_dem,
        zone_coords=z_xy,
        station_coords=s_xy,
        station_chargers_kw=s_kw,
        station_types=s_types,
        transformer_capacity_kw=t_cap,
        decay_beta=0.1,
    )

    assert "accessibility_score" in res
    assert "grid_stress_ratio" in res
    assert "station_capacity_ratios" in res
    assert "spatial_gini" in res
    assert "equity_index" in res
    assert "l2_accessibility" in res
    assert "dc_fast_accessibility" in res

    assert len(res["accessibility_score"]) == 3
    assert len(res["grid_stress_ratio"]) == 2
    assert res["l2_accessibility"] is not None
    assert res["dc_fast_accessibility"] is not None
    assert 0.0 <= res["spatial_gini"] <= 1.0


def test_ev_charging_accessibility_index_validation():
    from planx.spatial import ev_charging_accessibility_index

    z_dem = np.array([10.0, 20.0])
    z_xy = np.array([[0.0, 0.0], [1.0, 1.0]])
    s_xy = np.array([[0.5, 0.5]])
    s_kw = np.array([100.0])

    with pytest.raises(ValueError, match="zone_coords shape"):
        ev_charging_accessibility_index(z_dem, np.array([[0.0]]), s_xy, s_kw)

    with pytest.raises(ValueError, match="station_coords shape"):
        ev_charging_accessibility_index(z_dem, z_xy, np.array([0.5, 0.5]), s_kw)

    with pytest.raises(ValueError, match="zone_demand values must be non-negative"):
        ev_charging_accessibility_index(
            np.array([-10.0, 20.0]), z_xy, station_coords=s_xy, station_chargers_kw=s_kw
        )


def test_multimodal_transit_isochrone_profiler_normal():
    from planx.spatial import multimodal_transit_isochrone_profiler

    orig = np.array([0.0, 0.0])
    dests = np.array([[100.0, 100.0], [500.0, 500.0], [2000.0, 2000.0]])
    stops = np.array([[50.0, 50.0], [450.0, 450.0]])
    headways = np.array([10.0, 10.0])
    t_travel = np.array([[0.0, 5.0], [5.0, 0.0]])

    res = multimodal_transit_isochrone_profiler(
        origin_coord=orig,
        destination_coords=dests,
        transit_stop_coords=stops,
        transit_headways_min=headways,
        transit_travel_times=t_travel,
        walk_speed_kmh=4.8,
        transfer_penalty_min=5.0,
        max_time_budget_min=45.0,
    )

    assert "travel_times_min" in res
    assert "reachable_mask" in res
    assert "isochrone_bands" in res
    assert "mode_used" in res
    assert "reachable_count" in res
    assert "coverage_ratio" in res

    assert len(res["travel_times_min"]) == 3
    assert len(res["isochrone_bands"]) == 3
    assert res["reachable_count"] > 0
    assert 0.0 <= res["coverage_ratio"] <= 1.0


def test_multimodal_transit_isochrone_profiler_validation():
    from planx.spatial import multimodal_transit_isochrone_profiler

    orig = np.array([0.0, 0.0])
    dests = np.array([[100.0, 100.0]])
    stops = np.array([[50.0, 50.0]])
    headways = np.array([10.0])
    t_travel = np.array([[0.0]])

    with pytest.raises(ValueError, match="origin_coord must be a 1D array"):
        multimodal_transit_isochrone_profiler(
            np.array([[0.0, 0.0]]), dests, stops, headways, t_travel
        )

    with pytest.raises(ValueError, match="transit_headways_min length"):
        multimodal_transit_isochrone_profiler(orig, dests, stops, np.array([10.0, 5.0]), t_travel)


def test_ev_cvrp_multi_depot_routing_normal():
    from planx.spatial import ev_cvrp_multi_depot_routing

    depots = np.array([[0.0, 0.0], [10.0, 10.0]])
    customers = np.array([[1.0, 1.0], [2.0, 2.0], [9.0, 9.0]])
    demands = np.array([20.0, 30.0, 40.0])
    chargers = np.array([[5.0, 5.0]])

    res = ev_cvrp_multi_depot_routing(
        depot_coords=depots,
        customer_coords=customers,
        customer_demands=demands,
        charger_coords=chargers,
        vehicle_capacity=50.0,
        battery_capacity_kwh=60.0,
        energy_consumption_kwh_km=0.25,
    )

    assert "routes" in res
    assert "total_distance_km" in res
    assert "total_energy_kwh" in res
    assert "vehicles_used_count" in res
    assert "unserviced_customers_count" in res

    assert res["vehicles_used_count"] > 0
    assert res["total_distance_km"] > 0.0
    assert res["total_energy_kwh"] > 0.0


def test_ev_cvrp_multi_depot_routing_validation():
    from planx.spatial import ev_cvrp_multi_depot_routing

    depots = np.array([[0.0, 0.0]])
    customers = np.array([[1.0, 1.0]])
    demands = np.array([20.0])

    with pytest.raises(ValueError, match="depot_coords must be a 2D array"):
        ev_cvrp_multi_depot_routing(np.array([0.0, 0.0]), customers, demands)

    with pytest.raises(ValueError, match="vehicle_capacity must be positive"):
        ev_cvrp_multi_depot_routing(depots, customers, demands, vehicle_capacity=-10.0)


def test_micromobility_equity_index():
    from planx.spatial import micromobility_equity_index

    counts = np.array([10, 5, 2, 20])
    dists = np.array([100.0, 300.0, 800.0, 200.0])
    vuln = np.array([0.8, 0.5, 0.2, 0.9])

    res = micromobility_equity_index(counts, dists, vuln)

    assert "zone_equity_scores" in res
    assert "mean_equity_score" in res
    assert "equity_gini_index" in res
    assert len(res["zone_equity_scores"]) == 4
    assert 0.0 <= res["equity_gini_index"] <= 1.0


def test_transit_fleet_electrification_scheduler():
    from planx.spatial import transit_fleet_electrification_scheduler

    arrivals = np.array([22.0, 22.5, 23.0, 23.5])
    energy = np.array([120.0, 150.0, 100.0, 130.0])

    res = transit_fleet_electrification_scheduler(
        arrivals, energy, charger_power_kw=150.0, max_grid_power_kw=600.0
    )

    assert "peak_power_demand_kw" in res
    assert "total_energy_delivered_kwh" in res
    assert "grid_cap_compliant" in res
    assert res["grid_cap_compliant"] is True


def test_canopy_sky_view_factor_profiler():
    from planx.spatial import canopy_sky_view_factor_profiler

    grid = np.array([[0.0, 0.0], [10.0, 10.0]])
    b_heights = np.array([20.0, 30.0])
    b_coords = np.array([[5.0, 5.0], [15.0, 15.0]])

    res = canopy_sky_view_factor_profiler(grid, b_heights, b_coords)

    assert "sky_view_factor_grid" in res
    assert "mean_sky_view_factor" in res
    assert len(res["sky_view_factor_grid"]) == 2
    assert 0.0 <= res["mean_sky_view_factor"] <= 1.0


def test_drt_dispatch_optimizer():
    from planx.spatial import drt_dispatch_optimizer

    req_orig = np.array([[100.0, 100.0], [200.0, 200.0], [500.0, 500.0]])
    req_dest = np.array([[300.0, 300.0], [400.0, 400.0], [800.0, 800.0]])
    vehicles = np.array([[0.0, 0.0], [400.0, 400.0]])

    res = drt_dispatch_optimizer(req_orig, req_dest, vehicles, vehicle_capacity=8.0)

    assert "assigned_vehicle_indices" in res
    assert "vehicle_loads" in res
    assert "total_fleet_distance_km" in res
    assert len(res["assigned_vehicle_indices"]) == 3
    assert len(res["vehicle_loads"]) == 2


def test_fifteen_minute_city_equity_analyzer():
    from planx.spatial import fifteen_minute_city_equity_analyzer

    counts = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0],
        ]
    )
    times = np.array(
        [
            [10.0, 12.0, 14.0, 8.0, 11.0, 13.0],
            [20.0, 10.0, 30.0, 5.0, 40.0, 12.0],
            [12.0, 25.0, 10.0, 18.0, 14.0, 30.0],
        ]
    )
    vuln = np.array([0.2, 0.8, 0.6])

    res = fifteen_minute_city_equity_analyzer(counts, times, vuln)

    assert "zone_15m_city_scores" in res
    assert "mean_15m_city_score" in res
    assert "vulnerability_equity_gap" in res
    assert len(res["zone_15m_city_scores"]) == 3
