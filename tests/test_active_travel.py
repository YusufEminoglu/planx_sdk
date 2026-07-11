# -*- coding: utf-8 -*-
"""Tests for the active_travel resilience submodule."""

import numpy as np
import pytest

from planx.resilience.active_travel import (
    active_travel_equity_gini,
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
