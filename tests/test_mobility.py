# -*- coding: utf-8 -*-
"""Unit tests for planx.mobility submodule."""

import numpy as np

from planx.mobility import (
    bpr_link_performance_function,
    frank_wolfe_user_equilibrium,
    furness_matrix_balancing,
    gravity_model_od_estimation,
)


def test_bpr_link_performance_function():
    flow = np.array([500.0, 1200.0, 2000.0])
    cap = np.array([1000.0, 1000.0, 1000.0])
    t0 = np.array([10.0, 10.0, 10.0])

    t_congested = bpr_link_performance_function(flow, cap, t0)
    assert len(t_congested) == 3
    assert t_congested[2] > t_congested[0]


def test_frank_wolfe_user_equilibrium():
    cap = np.array([1000.0, 1000.0, 1000.0])
    t0 = np.array([5.0, 10.0, 8.0])
    od = np.array([[100.0, 200.0], [150.0, 50.0]])

    res = frank_wolfe_user_equilibrium(3, cap, t0, od, max_iter=5)
    assert "equilibrium_flows" in res
    assert len(res["equilibrium_flows"]) == 3
    assert res["total_system_travel_time_hours"] > 0.0


def test_gravity_model_od_estimation():
    p = np.array([1000.0, 2000.0])
    a = np.array([1500.0, 1500.0])
    d = np.array([[2.0, 10.0], [10.0, 3.0]])

    res = gravity_model_od_estimation(p, a, d)
    assert "od_matrix" in res
    assert res["od_matrix"].shape == (2, 2)
    assert res["mean_trip_distance"] > 0.0


def test_furness_matrix_balancing():
    init = np.array([[10.0, 20.0], [30.0, 40.0]])
    o_target = np.array([50.0, 100.0])
    d_target = np.array([60.0, 90.0])

    res = furness_matrix_balancing(init, o_target, d_target, max_iter=10)
    assert "balanced_od_matrix" in res
    assert res["balanced_od_matrix"].shape == (2, 2)
    assert res["row_error_ratio"] < 0.1
