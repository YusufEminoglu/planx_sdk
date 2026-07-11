# -*- coding: utf-8 -*-
"""Tests for the active_travel resilience submodule."""

import numpy as np
import pytest

from planx.resilience.active_travel import (
    active_travel_equity_gini,
    calculate_tod_index,
    equity_weighted_accessibility,
    job_housing_spatial_mismatch,
    transport_mismatch_index,
)


def test_job_housing_spatial_mismatch():
    pop = np.array([100.0, 200.0])
    jobs = np.array([50.0, 150.0])
    costs = np.array([[5.0, 15.0], [20.0, 8.0]])

    # 1. Linear decay
    smi = job_housing_spatial_mismatch(pop, jobs, costs, cutoff=15.0, decay_method="linear")
    assert len(smi) == 2

    # 2. None / Uniform decay
    smi_none = job_housing_spatial_mismatch(pop, jobs, costs, cutoff=15.0, decay_method="none")
    assert len(smi_none) == 2

    # 3. Exponential decay
    smi_exp = job_housing_spatial_mismatch(
        pop, jobs, costs, cutoff=15.0, decay_method="exponential"
    )
    assert len(smi_exp) == 2

    # 4. Validation checks
    with pytest.raises(ValueError, match="residential_pop shape"):
        job_housing_spatial_mismatch(pop[:-1], jobs, costs, cutoff=15.0)

    with pytest.raises(ValueError, match="job_capacity shape"):
        job_housing_spatial_mismatch(pop, jobs[:-1], costs, cutoff=15.0)

    with pytest.raises(ValueError, match="cutoff must be greater than 0"):
        job_housing_spatial_mismatch(pop, jobs, costs, cutoff=-1.0)

    with pytest.raises(ValueError, match="Unknown decay method"):
        job_housing_spatial_mismatch(pop, jobs, costs, cutoff=15.0, decay_method="invalid")


def test_active_travel_equity_gini():
    # Perfect equality
    acc = np.array([10.0, 10.0])
    pop = np.array([100.0, 100.0])
    gini, cum_pop, cum_acc = active_travel_equity_gini(acc, pop)
    assert np.isclose(gini, 0.0, atol=1e-5)
    assert len(cum_pop) == 3
    assert len(cum_acc) == 3

    # Inequality
    acc_ineq = np.array([0.0, 10.0])
    gini_ineq, _, _ = active_travel_equity_gini(acc_ineq, pop)
    assert gini_ineq > 0.0

    # Validation checks
    with pytest.raises(ValueError, match="identical length"):
        active_travel_equity_gini(acc[:-1], pop)

    # Edge cases (zero population or zero accessibility)
    g1, _, _ = active_travel_equity_gini(acc, np.array([0.0, 0.0]))
    assert g1 == 0.0

    g2, _, _ = active_travel_equity_gini(np.array([0.0, 0.0]), pop)
    assert g2 == 0.0


def test_transport_mismatch_index():
    acc = np.array([10.0, 50.0, 100.0])
    vuln = np.array([100.0, 50.0, 10.0])

    mismatch = transport_mismatch_index(acc, vuln)
    assert 0.0 <= mismatch <= 100.0
    # High vulnerability in low accessibility zones -> high mismatch index
    assert mismatch > 50.0

    # Validation checks
    with pytest.raises(ValueError, match="identical length"):
        transport_mismatch_index(acc[:-1], vuln)

    # Edge case: zero vulnerable population
    m_zero = transport_mismatch_index(acc, np.array([0.0, 0.0, 0.0]))
    assert m_zero == 0.0


def test_calculate_tod_index():
    densities = np.array([100.0, 500.0])
    shares = np.array([[0.8, 0.2], [0.5, 0.5]])
    connectivity = np.array([10.0, 50.0])

    # 1. Equal weights
    tod = calculate_tod_index(densities, shares, connectivity)
    assert len(tod) == 2
    assert tod[1] > tod[0]

    # 2. Custom weights
    tod_custom = calculate_tod_index(densities, shares, connectivity, weights=(0.5, 0.2, 0.3))
    assert len(tod_custom) == 2

    # 3. Invalid/negative weights defaults to equal weights
    tod_invalid_w = calculate_tod_index(densities, shares, connectivity, weights=(-1, -2, -3))
    assert np.allclose(tod_invalid_w, tod)

    # 4. Single land use category diversity (entropy = 0)
    tod_single_share = calculate_tod_index(densities, np.array([[1.0], [1.0]]), connectivity)
    assert len(tod_single_share) == 2

    # 5. Validation errors
    with pytest.raises(ValueError, match="land_use_shares rows"):
        calculate_tod_index(densities, np.array([[1.0]]), connectivity)

    with pytest.raises(ValueError, match="connectivity size"):
        calculate_tod_index(densities, shares, np.array([10.0]))


def test_equity_weighted_accessibility():
    acc = np.array([50.0, 100.0])
    dep = np.array([2.0, 1.0])

    # 1. Base weights calculation
    res = equity_weighted_accessibility(acc, dep, alpha=1.0)
    assert len(res) == 2
    # mean_dep = 1.5, multiplier = [2/1.5, 1/1.5] = [4/3, 2/3]
    # res = [50 * 4/3, 100 * 2/3] = [66.666, 66.666]
    assert np.allclose(res, [200.0 / 3.0, 200.0 / 3.0])

    # 2. Zero deprivation fallback
    res_zero = equity_weighted_accessibility(acc, np.array([0.0, 0.0]))
    assert np.allclose(res_zero, acc)

    # 3. Validation errors
    with pytest.raises(ValueError, match="identical length"):
        equity_weighted_accessibility(acc[:-1], dep)

    with pytest.raises(ValueError, match="alpha must be a non-negative float"):
        equity_weighted_accessibility(acc, dep, alpha=-1.0)
