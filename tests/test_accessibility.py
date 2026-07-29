import numpy as np
import pytest

from planx.spatial.accessibility import transit_frequency_accessibility


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
