# -*- coding: utf-8 -*-
"""Tests for the suitability submodule."""

import numpy as np
import pytest

from planx.suitability import (
    ahp_weights,
    aras_method,
    bwm_weights,
    capacitated_location_allocation,
    copras_method,
    critic_weights,
    decision_matrix_from_layers,
    dematel_method,
    edas_method,
    electre_i_method,
    electre_iii_method,
    entropy_weights,
    evaluate_tod_node_suitability,
    fucom_weights,
    fuzzy_ahp_weights,
    greedy_lscp,
    greedy_mclp,
    greedy_p_median,
    marcos_method,
    mcda_sensitivity_monte_carlo,
    mclp_distance_decay,
    normalize_array,
    pareto_facility_location,
    pca_weights,
    promethee_ii_method,
    topsis_method,
    vikor_method,
    waspas_method,
    weighted_linear_combination,
)


def test_normalize_array_benefit_minmax():
    arr = np.array([0.0, 50.0, 100.0], dtype=np.float32)
    norm = normalize_array(arr, "benefit_minmax", low=0.0, high=100.0)
    np.testing.assert_allclose(norm, [0.0, 50.0, 100.0])


def test_normalize_array_cost_minmax():
    arr = np.array([0.0, 50.0, 100.0], dtype=np.float32)
    norm = normalize_array(arr, "cost_minmax", low=0.0, high=100.0)
    np.testing.assert_allclose(norm, [100.0, 50.0, 0.0])


def test_normalize_array_sigmoid():
    arr = np.array([40.0, 50.0, 60.0], dtype=np.float32)
    norm = normalize_array(arr, "benefit_sigmoid", mid=50.0, spread=10.0)
    # At mid, sigmoid is 0.5 (which scales to 50.0)
    assert np.isclose(norm[1], 50.0)
    assert norm[2] > norm[1]
    assert norm[0] < norm[1]


def test_weighted_linear_combination():
    c1 = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    c2 = np.array([[50.0, 60.0], [70.0, 80.0]], dtype=np.float32)

    weights = [0.4, 0.6]
    result = weighted_linear_combination([c1, c2], weights)

    expected = 0.4 * c1 + 0.6 * c2
    np.testing.assert_allclose(result, expected)


def test_wlc_with_constraint():
    c1 = np.array([[50.0, 50.0], [50.0, 50.0]], dtype=np.float32)
    c2 = np.array([[50.0, 50.0], [50.0, 50.0]], dtype=np.float32)
    constraint = np.array([[1, 0], [1, 1]], dtype=np.uint8)

    weights = [0.5, 0.5]
    result = weighted_linear_combination(
        [c1, c2], weights, constraint_array=constraint, nodata=-999.0
    )

    expected = np.array([[50.0, 0.0], [50.0, 50.0]], dtype=np.float32)
    np.testing.assert_allclose(result, expected)


def test_greedy_mclp():
    candidates = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    demands = np.array([[1.0, 1.0], [11.0, 11.0], [25.0, 25.0]])
    pop = np.array([100.0, 200.0, 500.0])

    indices, added, cum = greedy_mclp(candidates, demands, pop, max_distance=5.0, k=2)
    assert indices == [1, 0]
    np.testing.assert_allclose(added, [200.0, 100.0])
    np.testing.assert_allclose(cum, [200.0, 300.0])

    existing = np.array([[0.0, 0.0]])
    indices, added, cum = greedy_mclp(
        candidates, demands, pop, max_distance=5.0, k=2, existing_coords=existing
    )
    assert indices == [1]
    np.testing.assert_allclose(added, [200.0])
    np.testing.assert_allclose(cum, [300.0])


def test_ahp_weights():
    # 3x3 consistency comparison matrix
    matrix = np.array([[1.0, 2.0, 3.0], [0.5, 1.0, 2.0], [0.3333333, 0.5, 1.0]])
    weights, cr = ahp_weights(matrix)

    assert weights.shape == (3,)
    assert cr < 0.10
    assert np.isclose(np.sum(weights), 1.0)
    assert weights[0] > weights[1] > weights[2]


def test_decision_matrix_from_layers():
    lyr1 = np.array([[1.0, 2.0], [np.nan, 4.0]])
    lyr2 = np.array([[10.0, 20.0], [30.0, 40.0]])

    dm, mask = decision_matrix_from_layers([lyr1, lyr2])

    # Position (1, 0) is nan in lyr1, so only 3 pixels should be valid
    assert dm.shape == (3, 2)
    assert np.all(mask == [[True, True], [False, True]])
    np.testing.assert_allclose(dm, [[1.0, 10.0], [2.0, 20.0], [4.0, 40.0]])


def test_entropy_weights():
    # 4 alternatives, 3 criteria
    decision_matrix = np.array(
        [[10.0, 100.0, 0.1], [20.0, 50.0, 0.2], [15.0, 80.0, 0.15], [30.0, 20.0, 0.3]]
    )
    weights = entropy_weights(decision_matrix)

    assert weights.shape == (3,)
    assert np.isclose(np.sum(weights), 1.0)


def test_critic_weights():
    # 5 alternatives, 3 criteria
    decision_matrix = np.array(
        [
            [10.0, 100.0, 1.0],
            [20.0, 80.0, 1.2],
            [15.0, 90.0, 1.1],
            [30.0, 70.0, 1.5],
            [25.0, 60.0, 1.3],
        ]
    )

    # 2 benefit criteria, 1 cost criterion (index 1 is cost)
    weights, sigmas, contrasts = critic_weights(decision_matrix, [1, -1, 1])

    assert weights.shape == (3,)
    assert sigmas.shape == (3,)
    assert contrasts.shape == (3,)
    assert np.isclose(np.sum(weights), 1.0)


def test_pca_weights():
    decision_matrix = np.array(
        [
            [10.0, 100.0, 1.0],
            [20.0, 80.0, 1.2],
            [15.0, 90.0, 1.1],
            [30.0, 70.0, 1.5],
            [25.0, 60.0, 1.3],
        ]
    )
    weights = pca_weights(decision_matrix)

    assert weights.shape == (3,)
    assert np.isclose(np.sum(weights), 1.0)


def test_greedy_p_median():
    candidates = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    demands = np.array([[1.0, 1.0], [11.0, 11.0], [25.0, 25.0]])
    pop = np.array([100.0, 200.0, 500.0])

    # 1. p=2 using coordinates
    selected, costs = greedy_p_median(
        candidate_coords=candidates, demand_coords=demands, demand_pop=pop, p=2
    )
    # The first greedy choice will pick candidates[2] (20,20) because it is closest
    # to the largest pop 500.
    # The second choice will pick candidates[1] (10,10) to cover the rest.
    assert selected == [2, 1]
    assert len(costs) == 2

    # 2. p=2 using precomputed distance matrix
    dists = np.array([[1.414, 15.556, 35.355], [12.728, 1.414, 21.213], [26.870, 12.728, 7.071]])
    selected_dist, costs_dist = greedy_p_median(dists=dists, demand_pop=pop, p=2)
    assert selected_dist == [2, 1]


def test_greedy_lscp():
    candidates = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    demands = np.array([[1.0, 1.0], [11.0, 11.0], [25.0, 25.0]])
    pop = np.array([100.0, 200.0, 500.0])

    # With max_distance = 15.0 and target_coverage = 0.8
    # Total pop = 800. 80% = 640.
    # Candidates[1] covers (1,1) (dist 14.14) and (11,11) (dist 1.414) -> 300 pop
    # Candidates[2] covers (11,11) (dist 12.72) and (25,25) (dist 7.07) -> 700 pop
    # Picking candidates[2] first covers 700 pop (which is >= 640).
    # So it should stop after picking 1 facility!
    selected, cov_frac = greedy_lscp(
        candidates, demands, demand_pop=pop, max_distance=15.0, target_coverage=0.8
    )
    assert selected == [2]
    assert cov_frac == 700.0 / 800.0


def test_capacitated_location_allocation():
    facilities = np.array([[0.0, 0.0], [10.0, 0.0]])
    capacities = np.array([150.0, 200.0])
    demands = np.array([[1.0, 0.0], [9.0, 0.0], [2.0, 0.0]])
    pop = np.array([100.0, 150.0, 80.0])

    # Minimum distances to facilities:
    # d((1,0)) to F0 is 1.0, to F1 is 9.0.
    # d((9,0)) to F0 is 9.0, to F1 is 1.0.
    # d((2,0)) to F0 is 2.0, to F1 is 8.0.
    # Sorted order of demands by min distance:
    # 1. d_idx=0 (dist 1.0)
    # 2. d_idx=1 (dist 1.0)
    # 3. d_idx=2 (dist 2.0)

    # Allocating d_idx=0 (pop 100) -> closest F0 (cap 150). Remaining F0 cap = 50.
    # Allocating d_idx=1 (pop 150) -> closest F1 (cap 200). Remaining F1 cap = 50.
    # Allocating d_idx=2 (pop 80) -> closest F0 (need 80, but remaining is 50 -> too small).
    # Next closest is F1 (need 80, but remaining is 50 -> too small).
    # So d_idx=2 is unassigned.

    allocations, unassigned, usage = capacitated_location_allocation(
        facilities, capacities, demands, pop
    )

    assert allocations[0] == [0]
    assert allocations[1] == [1]
    assert list(unassigned) == [2]
    assert np.allclose(usage, [100.0, 150.0])

    # Test max distance: F0 cannot serve d_idx=0 if max_distance < 1.0
    allocations_dist, unassigned_dist, usage_dist = capacitated_location_allocation(
        facilities, capacities, demands, pop, max_distance=0.5
    )
    assert len(allocations_dist[0]) == 0
    assert len(allocations_dist[1]) == 0
    assert len(unassigned_dist) == 3

    # Error handling
    with pytest.raises(ValueError, match="shape"):
        capacitated_location_allocation(np.ones((2, 3)), capacities, demands, pop)


def test_topsis_method():
    # 3 alternatives, 2 criteria (both benefit)
    # Alt 0 is clearly best, Alt 2 is clearly worst
    decision_matrix = np.array([[10.0, 100.0], [5.0, 50.0], [1.0, 10.0]])
    weights = np.array([0.5, 0.5])
    benefit_criteria = np.array([True, True])

    scores, ranks = topsis_method(decision_matrix, weights, benefit_criteria)

    assert len(scores) == 3
    assert len(ranks) == 3
    # Alt 0 should rank 1st, Alt 1 2nd, Alt 2 3rd
    assert ranks[0] == 1
    assert ranks[1] == 2
    assert ranks[2] == 3
    assert scores[0] > scores[1] > scores[2]
    # Check bounds
    assert np.all(scores >= 0.0) & np.all(scores <= 1.0)

    # Error checking
    with pytest.raises(ValueError):
        topsis_method(decision_matrix, weights[:-1], benefit_criteria)


def test_vikor_method():
    # 3 alternatives, 2 criteria (both benefit)
    decision_matrix = np.array([[10.0, 100.0], [5.0, 50.0], [1.0, 10.0]])
    weights = np.array([0.5, 0.5])
    benefit_criteria = np.array([True, True])

    scores, ranks = vikor_method(decision_matrix, weights, benefit_criteria, v=0.5)

    assert len(scores) == 3
    assert len(ranks) == 3
    # Lower is better in VIKOR compromise index Q
    # Alt 0 is closest to ideal best, so Q should be 0.0 (best)
    # Alt 2 is at ideal worst, so Q should be 1.0 (worst)
    assert np.isclose(scores[0], 0.0)
    assert np.isclose(scores[2], 1.0)
    assert ranks[0] == 1
    assert ranks[1] == 2
    assert ranks[2] == 3

    # Error checking
    with pytest.raises(ValueError):
        vikor_method(decision_matrix, weights, benefit_criteria[:-1])


def test_promethee_ii_method():
    decision_matrix = np.array([[10.0, 100.0], [5.0, 50.0], [1.0, 10.0]])
    weights = np.array([0.5, 0.5])
    benefit_criteria = np.array([True, True])

    net_flows, ranks = promethee_ii_method(decision_matrix, weights, benefit_criteria)

    assert len(net_flows) == 3
    assert len(ranks) == 3
    # Alt 0 is best -> net flow should be positive, rank 1
    # Alt 2 is worst -> net flow should be negative, rank 3
    assert ranks[0] == 1
    assert ranks[1] == 2
    assert ranks[2] == 3
    assert net_flows[0] > net_flows[1] > net_flows[2]

    # Test with custom preference thresholds
    p_thresh = np.array([5.0, 50.0])
    net_flows_thresh, ranks_thresh = promethee_ii_method(
        decision_matrix, weights, benefit_criteria, preference_thresholds=p_thresh
    )
    assert ranks_thresh[0] == 1

    # Single alternative edge case
    single_dm = np.array([[10.0, 100.0]])
    flows_single, ranks_single = promethee_ii_method(single_dm, weights, benefit_criteria)
    assert len(flows_single) == 1
    assert ranks_single[0] == 1

    # Validation errors
    with pytest.raises(ValueError):
        promethee_ii_method(decision_matrix, weights[:-1], benefit_criteria)
    with pytest.raises(ValueError):
        promethee_ii_method(decision_matrix, weights, benefit_criteria[:-1])
    with pytest.raises(ValueError):
        promethee_ii_method(
            decision_matrix, weights, benefit_criteria, preference_thresholds=p_thresh[:-1]
        )


def test_electre_i_method():
    dm = np.array([[10.0, 100.0], [5.0, 50.0], [1.0, 10.0]])
    weights = np.array([0.5, 0.5])
    benefit = np.array([True, True])

    C, D, non_dom = electre_i_method(
        dm, weights, benefit, concordance_threshold=0.5, discordance_threshold=0.5
    )

    assert C.shape == (3, 3)
    assert D.shape == (3, 3)
    assert 0 in non_dom  # Alt 0 is non-dominated

    # Validation errors
    with pytest.raises(ValueError):
        electre_i_method(dm, weights[:-1], benefit)


def test_electre_iii_method():
    dm = np.array([[10.0, 100.0], [5.0, 50.0], [1.0, 10.0]])
    weights = np.array([0.5, 0.5])
    benefit = np.array([True, True])
    q = np.array([1.0, 5.0])
    p = np.array([3.0, 15.0])
    v = np.array([8.0, 40.0])

    S, ranks = electre_iii_method(dm, weights, benefit, q, p, v)

    assert S.shape == (3, 3)
    assert len(ranks) == 3
    assert ranks[0] == 1

    # Validation errors
    with pytest.raises(ValueError):
        electre_iii_method(dm, weights[:-1], benefit, q, p, v)


def test_mclp_distance_decay():
    candidates = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    demands = np.array([[1.0, 1.0], [11.0, 11.0], [25.0, 25.0]])
    pop = np.array([100.0, 200.0, 500.0])

    selected, added, cum = mclp_distance_decay(
        candidates, demands, pop, max_distance=15.0, k=2, decay_method="exponential", beta=0.05
    )

    assert len(selected) <= 2
    assert len(added) == len(selected)
    assert len(cum) == len(selected)

    # Validation error
    with pytest.raises(ValueError, match="Unsupported decay_method"):
        mclp_distance_decay(
            candidates, demands, pop, max_distance=15.0, k=2, decay_method="invalid"
        )


def test_bwm_weights():
    best_to_others = np.array([1.0, 3.0, 9.0])
    others_to_worst = np.array([9.0, 3.0, 1.0])

    w, xi = bwm_weights(best_to_others, others_to_worst)

    assert len(w) == 3
    assert np.isclose(np.sum(w), 1.0)
    assert w[0] > w[1] > w[2]
    assert xi >= 0.0

    with pytest.raises(ValueError, match="equal length"):
        bwm_weights(best_to_others, others_to_worst[:-1])


def test_pareto_facility_location():
    candidates = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    demands = np.array([[1.0, 1.0], [11.0, 11.0], [25.0, 25.0]])
    pop = np.array([100.0, 200.0, 500.0])

    pareto_list = pareto_facility_location(candidates, demands, pop, k=2)

    assert isinstance(pareto_list, list)
    assert len(pareto_list) > 0
    assert "total_coverage" in pareto_list[0]
    assert "avg_distance" in pareto_list[0]


def test_fuzzy_ahp_weights():
    mat = np.array(
        [
            [[1.0, 1.0, 1.0], [2.0, 3.0, 4.0]],
            [[1 / 4.0, 1 / 3.0, 1 / 2.0], [1.0, 1.0, 1.0]],
        ]
    )

    w, ci = fuzzy_ahp_weights(mat)

    assert len(w) == 2
    assert np.isclose(np.sum(w), 1.0)
    assert w[0] > w[1]

    with pytest.raises(ValueError, match="shape"):
        fuzzy_ahp_weights(np.zeros((2, 2)))


def test_mcda_sensitivity_monte_carlo():
    X = np.array([[10.0, 2.0], [5.0, 8.0], [1.0, 10.0]])
    w_base = np.array([0.6, 0.4])

    res = mcda_sensitivity_monte_carlo(X, w_base, noise_level=0.05, n_simulations=100)

    assert "mean_ranks" in res
    assert "std_ranks" in res
    assert len(res["mean_ranks"]) == 3
    assert np.isclose(np.sum(res["rank_first_probability"]), 1.0)


def test_marcos_method():
    X = np.array([[10.0, 2.0], [5.0, 8.0], [1.0, 10.0]])
    w = np.array([0.6, 0.4])

    scores, ranks = marcos_method(X, w)

    assert len(scores) == 3
    assert len(ranks) == 3
    assert ranks[0] == 1

    with pytest.raises(ValueError, match="weights length"):
        marcos_method(X, w[:1])


def test_fucom_weights():
    phi = np.array([2.0, 1.5])
    ranks = np.array([0, 1, 2])

    w, chi = fucom_weights(phi, ranks)

    assert len(w) == 3
    assert np.isclose(np.sum(w), 1.0)
    assert w[0] > w[1] > w[2]
    assert chi >= 0.0

    with pytest.raises(ValueError, match="N-1"):
        fucom_weights(phi[:1], ranks)


def test_waspas_method():
    X = np.array([[10.0, 2.0], [5.0, 8.0], [1.0, 10.0]])
    w = np.array([0.6, 0.4])

    q, ranks = waspas_method(X, w, lambda_param=0.5)

    assert len(q) == 3
    assert len(ranks) == 3

    with pytest.raises(ValueError, match="lambda_param"):
        waspas_method(X, w, lambda_param=1.5)


def test_dematel_method():
    Z = np.array([[0.0, 3.0, 2.0], [1.0, 0.0, 3.0], [2.0, 1.0, 0.0]])

    res = dematel_method(Z)

    assert "total_influence_matrix" in res
    assert "prominence" in res
    assert "relation" in res
    assert len(res["cause_effect_class"]) == 3

    with pytest.raises(ValueError, match="square"):
        dematel_method(Z[:2, :3])


# ---------------------------------------------------------------------------
# Additional coverage: mcda.normalize_array
# ---------------------------------------------------------------------------


def test_normalize_array_validation_errors():
    arr = np.array([1.0, 2.0], dtype=np.float32)
    with pytest.raises(ValueError, match="high must be greater than low"):
        normalize_array(arr, "benefit_minmax", low=10.0, high=5.0)

    with pytest.raises(ValueError, match="spread must be greater than 0"):
        normalize_array(arr, "benefit_sigmoid", spread=0.0)

    with pytest.raises(ValueError, match="Unknown normalization method"):
        normalize_array(arr, "not_a_real_method")


def test_normalize_array_cost_sigmoid_and_gaussian():
    arr = np.array([40.0, 50.0, 60.0], dtype=np.float32)

    cost_sig = normalize_array(arr, "cost_sigmoid", mid=50.0, spread=10.0)
    assert np.isclose(cost_sig[1], 50.0)
    assert cost_sig[0] > cost_sig[1] > cost_sig[2]

    gauss = normalize_array(arr, "benefit_gaussian", mid=50.0, spread=10.0)
    # Peak at mid, symmetric decay on both sides
    assert gauss[1] > gauss[0]
    assert gauss[1] > gauss[2]
    assert np.isclose(gauss[0], gauss[2])


def test_normalize_array_nodata_preserved():
    arr = np.array([10.0, 50.0, np.nan], dtype=np.float32)
    norm = normalize_array(arr, "benefit_minmax", low=0.0, high=100.0, nodata=-1.0)
    # NaN input should be replaced with the nodata value in the output
    assert norm[2] == -1.0
    assert norm[0] == 10.0

    arr2 = np.array([10.0, 50.0, 999.0], dtype=np.float32)
    norm2 = normalize_array(arr2, "benefit_minmax", low=0.0, high=100.0, nodata=999.0)
    # The nodata sentinel value should be excluded and preserved
    assert norm2[2] == 999.0

    # No explicit nodata: NaN inputs become NaN outputs
    arr3 = np.array([10.0, np.nan], dtype=np.float32)
    norm3 = normalize_array(arr3, "benefit_minmax", low=0.0, high=100.0)
    assert np.isnan(norm3[1])


# ---------------------------------------------------------------------------
# Additional coverage: mcda.weighted_linear_combination
# ---------------------------------------------------------------------------


def test_wlc_validation_errors():
    with pytest.raises(ValueError, match="At least one criterion array"):
        weighted_linear_combination([], [1.0])

    with pytest.raises(ValueError, match="Number of weights"):
        weighted_linear_combination([np.ones((2, 2))], [0.5, 0.5])

    with pytest.raises(ValueError, match="identical shapes"):
        weighted_linear_combination([np.ones((2, 2)), np.ones((3, 3))], [0.5, 0.5])

    with pytest.raises(ValueError, match="Constraint array shape"):
        weighted_linear_combination(
            [np.ones((2, 2)), np.ones((2, 2))],
            [0.5, 0.5],
            constraint_array=np.ones((3, 3)),
        )


def test_wlc_renormalizes_weights_not_summing_to_one():
    c1 = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    c2 = np.array([[50.0, 60.0], [70.0, 80.0]], dtype=np.float32)

    # Weights [0.3, 0.3] sum to 0.6 -> should be re-normalized to [0.5, 0.5]
    result = weighted_linear_combination([c1, c2], [0.3, 0.3])
    expected = 0.5 * c1 + 0.5 * c2
    np.testing.assert_allclose(result, expected)


def test_wlc_criteria_nodatas():
    c1 = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    c2 = np.array([[10.0, 20.0], [30.0, 999.0]], dtype=np.float32)

    result = weighted_linear_combination([c1, c2], [0.5, 0.5], criteria_nodatas=[None, 999.0])
    # The pixel where c2 is nodata (999.0) should be excluded and set to output nodata
    assert result[1, 1] == -9999.0
    # Other pixels should compute normally
    assert np.isclose(result[0, 0], 10.0)


# ---------------------------------------------------------------------------
# Additional coverage: mcda.topsis_method / vikor_method
# ---------------------------------------------------------------------------


def test_topsis_method_mixed_benefit_cost_criteria():
    decision_matrix = np.array([[10.0, 100.0], [5.0, 10.0], [1.0, 50.0]])
    weights = np.array([0.5, 0.5])
    # Criterion 0 is a benefit, criterion 1 is a cost
    benefit_criteria = np.array([True, False])

    scores, ranks = topsis_method(decision_matrix, weights, benefit_criteria)
    assert scores.shape == (3,)
    assert ranks.shape == (3,)
    assert set(ranks.tolist()) == {1, 2, 3}

    with pytest.raises(ValueError, match="benefit_criteria length"):
        topsis_method(decision_matrix, weights, np.array([True]))


def test_topsis_method_zero_weights():
    decision_matrix = np.array([[10.0, 100.0], [5.0, 10.0], [1.0, 50.0]])
    weights = np.array([0.0, 0.0])
    benefit_criteria = np.array([True, True])

    scores, ranks = topsis_method(decision_matrix, weights, benefit_criteria)
    # With all-zero weights, every alternative collapses to the same
    # (degenerate) point, so scores should be equal (falls back to 0.5).
    np.testing.assert_allclose(scores, [0.5, 0.5, 0.5])


def test_vikor_method_mixed_benefit_cost_criteria():
    decision_matrix = np.array([[10.0, 100.0], [5.0, 10.0], [1.0, 50.0]])
    weights = np.array([0.5, 0.5])
    benefit_criteria = np.array([True, False])

    scores, ranks = vikor_method(decision_matrix, weights, benefit_criteria)
    assert scores.shape == (3,)
    assert ranks.shape == (3,)
    assert set(ranks.tolist()) == {1, 2, 3}

    with pytest.raises(ValueError, match="benefit_criteria length"):
        vikor_method(decision_matrix, weights, np.array([True]))

    with pytest.raises(ValueError, match="weights length"):
        vikor_method(decision_matrix, weights[:-1], benefit_criteria)


def test_vikor_method_zero_weights():
    decision_matrix = np.array([[10.0, 100.0], [5.0, 10.0], [1.0, 50.0]])
    weights = np.array([0.0, 0.0])
    benefit_criteria = np.array([True, True])

    scores, ranks = vikor_method(decision_matrix, weights, benefit_criteria)
    np.testing.assert_allclose(scores, [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Additional coverage: weights.ahp_weights
# ---------------------------------------------------------------------------


def test_ahp_weights_validation_errors():
    with pytest.raises(ValueError, match="must be square"):
        ahp_weights(np.ones((2, 3)))

    with pytest.raises(ValueError, match="cannot be empty"):
        ahp_weights(np.zeros((0, 0)))

    with pytest.raises(ValueError, match="must be positive"):
        ahp_weights(np.array([[1.0, -2.0], [0.5, 1.0]]))


def test_ahp_weights_large_matrix_uses_default_ri():
    # 11x11 matrix (n > 10) exercises the ri_map.get(n, 1.49) fallback
    n = 11
    rng = np.random.default_rng(42)
    matrix = np.ones((n, n))
    upper = rng.uniform(1.0, 5.0, size=(n, n))
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = upper[i, j]
            matrix[j, i] = 1.0 / upper[i, j]

    weights, cr = ahp_weights(matrix)
    assert weights.shape == (n,)
    assert np.isclose(np.sum(weights), 1.0)
    assert cr >= 0.0


# ---------------------------------------------------------------------------
# Additional coverage: weights.decision_matrix_from_layers
# ---------------------------------------------------------------------------


def test_decision_matrix_from_layers_validation_errors():
    with pytest.raises(ValueError, match="At least one layer"):
        decision_matrix_from_layers([])

    with pytest.raises(ValueError, match="Layer at index 1"):
        decision_matrix_from_layers([np.ones((2, 2)), np.ones((3, 3))])


def test_decision_matrix_from_layers_with_nodata():
    lyr1 = np.array([[1.0, 2.0], [3.0, 999.0]])
    lyr2 = np.array([[10.0, 20.0], [30.0, 40.0]])

    dm, mask = decision_matrix_from_layers([lyr1, lyr2], nodata=999.0)
    assert dm.shape == (3, 2)
    assert np.all(mask == [[True, True], [True, False]])


# ---------------------------------------------------------------------------
# Additional coverage: weights.entropy_weights
# ---------------------------------------------------------------------------


def test_entropy_weights_validation_error():
    with pytest.raises(ValueError, match="2D array"):
        entropy_weights(np.array([1.0, 2.0, 3.0]))


def test_entropy_weights_degenerate_shapes():
    # Zero alternatives: falls back to uniform weights
    weights = entropy_weights(np.zeros((0, 3)))
    np.testing.assert_allclose(weights, [1 / 3, 1 / 3, 1 / 3])

    # Zero criteria: returns an empty array
    weights_empty = entropy_weights(np.zeros((4, 0)))
    assert weights_empty.shape == (0,)


def test_entropy_weights_constant_matrix_falls_back_to_uniform():
    # All criteria are identical across alternatives -> zero diversification
    # degree for every criterion, triggering the uniform-weight fallback.
    decision_matrix = np.ones((4, 3)) * 5.0
    weights = entropy_weights(decision_matrix)
    np.testing.assert_allclose(weights, [1 / 3, 1 / 3, 1 / 3])


# ---------------------------------------------------------------------------
# Additional coverage: weights.critic_weights
# ---------------------------------------------------------------------------


def test_critic_weights_validation_errors():
    with pytest.raises(ValueError, match="2D array"):
        critic_weights(np.array([1.0, 2.0, 3.0]))

    decision_matrix = np.array([[10.0, 100.0], [20.0, 80.0], [15.0, 90.0]])
    with pytest.raises(ValueError, match="directions length"):
        critic_weights(decision_matrix, directions=[1])


def test_critic_weights_degenerate_shapes():
    weights, sigmas, contrasts = critic_weights(np.zeros((0, 3)))
    np.testing.assert_allclose(weights, [1 / 3, 1 / 3, 1 / 3])
    np.testing.assert_allclose(sigmas, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(contrasts, [1.0, 1.0, 1.0])

    weights_empty, _, _ = critic_weights(np.zeros((4, 0)))
    assert weights_empty.shape == (0,)


def test_critic_weights_default_directions():
    # directions=None should default to treating all criteria as benefit
    decision_matrix = np.array(
        [
            [10.0, 100.0, 1.0],
            [20.0, 80.0, 1.2],
            [15.0, 90.0, 1.1],
            [30.0, 70.0, 1.5],
            [25.0, 60.0, 1.3],
        ]
    )
    weights, sigmas, contrasts = critic_weights(decision_matrix)
    assert weights.shape == (3,)
    assert np.isclose(np.sum(weights), 1.0)


def test_critic_weights_constant_matrix_falls_back_to_uniform():
    # Zero variance across all criteria -> zero contrast score sum
    decision_matrix = np.ones((4, 3)) * 5.0
    weights, sigmas, contrasts = critic_weights(decision_matrix)
    np.testing.assert_allclose(weights, [1 / 3, 1 / 3, 1 / 3])
    np.testing.assert_allclose(sigmas, [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Additional coverage: weights.pca_weights
# ---------------------------------------------------------------------------


def test_pca_weights_validation_error():
    with pytest.raises(ValueError, match="2D array"):
        pca_weights(np.array([1.0, 2.0, 3.0]))


def test_pca_weights_too_few_alternatives():
    # Fewer than 3 alternatives falls back to uniform weights
    weights = pca_weights(np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_allclose(weights, [0.5, 0.5])


def test_pca_weights_single_criterion():
    # A single criterion column exercises the 0-d covariance reshape branch
    weights = pca_weights(np.array([[1.0], [2.0], [3.0], [4.0]]))
    np.testing.assert_allclose(weights, [1.0])


def test_pca_weights_constant_matrix_falls_back_to_uniform():
    # Zero variance in every criterion -> eigenvalues are all ~0
    decision_matrix = np.ones((5, 3)) * 3.0
    weights = pca_weights(decision_matrix)
    np.testing.assert_allclose(weights, [1 / 3, 1 / 3, 1 / 3])


# ---------------------------------------------------------------------------
# Additional coverage: facility.greedy_mclp
# ---------------------------------------------------------------------------


def test_greedy_mclp_validation_errors():
    candidates = np.array([[0.0, 0.0], [10.0, 10.0]])
    demands = np.array([[1.0, 1.0], [11.0, 11.0]])
    pop = np.array([100.0, 200.0])

    with pytest.raises(ValueError, match="candidate_coords"):
        greedy_mclp(np.ones((2, 3)), demands, pop, max_distance=5.0, k=1)

    with pytest.raises(ValueError, match="demand_coords"):
        greedy_mclp(candidates, np.ones((2, 3)), pop, max_distance=5.0, k=1)

    with pytest.raises(ValueError, match="demand_pop"):
        greedy_mclp(candidates, demands, np.ones(3), max_distance=5.0, k=1)

    with pytest.raises(ValueError, match="existing_coords"):
        greedy_mclp(
            candidates, demands, pop, max_distance=5.0, k=1, existing_coords=np.ones((1, 3))
        )


# ---------------------------------------------------------------------------
# Additional coverage: facility.greedy_p_median
# ---------------------------------------------------------------------------


def test_greedy_p_median_validation_errors():
    candidates = np.array([[0.0, 0.0], [10.0, 10.0]])
    demands = np.array([[1.0, 1.0], [11.0, 11.0]])

    with pytest.raises(ValueError, match="dists must be a 2D array"):
        greedy_p_median(dists=np.ones((2, 2, 2)), p=1)

    with pytest.raises(ValueError, match="Must provide either dists"):
        greedy_p_median(p=1)

    with pytest.raises(ValueError, match="candidate_coords"):
        greedy_p_median(candidate_coords=np.ones((2, 3)), demand_coords=demands, p=1)

    with pytest.raises(ValueError, match="demand_coords"):
        greedy_p_median(candidate_coords=candidates, demand_coords=np.ones((2, 3)), p=1)

    with pytest.raises(ValueError, match="demand_pop"):
        greedy_p_median(
            candidate_coords=candidates, demand_coords=demands, demand_pop=np.ones(3), p=1
        )

    with pytest.raises(ValueError, match="p must be greater than 0"):
        greedy_p_median(candidate_coords=candidates, demand_coords=demands, p=0)

    with pytest.raises(ValueError, match="existing_coords"):
        greedy_p_median(
            candidate_coords=candidates,
            demand_coords=demands,
            p=1,
            existing_coords=np.ones((1, 3)),
        )


def test_greedy_p_median_default_population_is_uniform():
    candidates = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    demands = np.array([[1.0, 1.0], [11.0, 11.0], [25.0, 25.0]])

    selected, costs = greedy_p_median(candidate_coords=candidates, demand_coords=demands, p=1)
    assert len(selected) == 1
    assert len(costs) == 1


def test_greedy_p_median_existing_indices():
    candidates = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    demands = np.array([[1.0, 1.0], [11.0, 11.0], [25.0, 25.0]])
    pop = np.array([100.0, 200.0, 500.0])

    selected, costs = greedy_p_median(
        candidate_coords=candidates,
        demand_coords=demands,
        demand_pop=pop,
        p=1,
        existing_indices=[2],
    )
    # Facility 2 is already selected/existing, so it should not reappear
    # in the "newly selected" output list.
    assert 2 not in selected
    assert len(selected) == 1

    with pytest.raises(ValueError, match="existing_indices"):
        greedy_p_median(
            candidate_coords=candidates,
            demand_coords=demands,
            demand_pop=pop,
            p=1,
            existing_indices=[99],
        )


def test_greedy_p_median_existing_coords():
    candidates = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    demands = np.array([[1.0, 1.0], [11.0, 11.0], [25.0, 25.0]])
    pop = np.array([100.0, 200.0, 500.0])

    selected, costs = greedy_p_median(
        candidate_coords=candidates,
        demand_coords=demands,
        demand_pop=pop,
        p=1,
        existing_coords=np.array([[20.0, 20.0]]),
    )
    assert len(selected) == 1
    assert len(costs) == 1


def test_greedy_p_median_exhausts_candidates():
    candidates = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    demands = np.array([[1.0, 1.0], [11.0, 11.0], [25.0, 25.0]])
    pop = np.array([100.0, 200.0, 500.0])

    # Requesting more facilities than available candidates should stop early
    selected, costs = greedy_p_median(
        candidate_coords=candidates, demand_coords=demands, demand_pop=pop, p=10
    )
    assert len(selected) == 3
    assert len(costs) == 3


# ---------------------------------------------------------------------------
# Additional coverage: facility.greedy_lscp
# ---------------------------------------------------------------------------


def test_greedy_lscp_validation_errors():
    candidates = np.array([[0.0, 0.0], [10.0, 10.0]])
    demands = np.array([[1.0, 1.0], [11.0, 11.0]])
    pop = np.array([100.0, 200.0])

    with pytest.raises(ValueError, match="candidate_coords"):
        greedy_lscp(np.ones((2, 3)), demands, demand_pop=pop)

    with pytest.raises(ValueError, match="demand_coords"):
        greedy_lscp(candidates, np.ones((2, 3)), demand_pop=pop)

    with pytest.raises(ValueError, match="demand_pop"):
        greedy_lscp(candidates, demands, demand_pop=np.ones(3))

    with pytest.raises(ValueError, match="existing_coords"):
        greedy_lscp(candidates, demands, demand_pop=pop, existing_coords=np.ones((1, 3)))


def test_greedy_lscp_default_population_is_uniform():
    candidates = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    demands = np.array([[1.0, 1.0], [11.0, 11.0], [25.0, 25.0]])

    selected, cov_frac = greedy_lscp(candidates, demands, max_distance=50.0, target_coverage=1.0)
    assert cov_frac == 1.0


def test_greedy_lscp_zero_population_fallback():
    candidates = np.array([[0.0, 0.0], [10.0, 10.0]])
    demands = np.array([[0.0, 0.0], [10.0, 10.0]])
    pop = np.array([0.0, 0.0])

    selected, cov_frac = greedy_lscp(
        candidates, demands, demand_pop=pop, max_distance=100.0, target_coverage=1.0
    )
    # No population can ever be covered, so the greedy loop breaks immediately
    assert selected == []
    assert cov_frac == 0.0


def test_greedy_lscp_existing_coords_already_meets_target():
    candidates = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    demands = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    pop = np.array([100.0, 100.0, 100.0])
    existing = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])

    selected, cov_frac = greedy_lscp(
        candidates,
        demands,
        demand_pop=pop,
        max_distance=1.0,
        target_coverage=1.0,
        existing_coords=existing,
    )
    # Existing facilities already cover 100% of demand, no new facilities needed
    assert selected == []
    assert cov_frac == 1.0


def test_greedy_lscp_requires_multiple_iterations():
    candidates = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    demands = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    pop = np.array([100.0, 100.0, 100.0])

    selected, cov_frac = greedy_lscp(
        candidates, demands, demand_pop=pop, max_distance=1.0, target_coverage=1.0
    )
    assert len(selected) == 3
    assert cov_frac == 1.0


def test_greedy_lscp_unreachable_target_stops_early():
    candidates = np.array([[0.0, 0.0], [10.0, 10.0]])
    demands = np.array([[1.0, 1.0], [100.0, 100.0]])
    pop = np.array([100.0, 900.0])

    # The second demand point is unreachable by any candidate within
    # max_distance, so 100% coverage can never be achieved.
    selected, cov_frac = greedy_lscp(
        candidates, demands, demand_pop=pop, max_distance=5.0, target_coverage=1.0
    )
    assert selected == [0]
    assert np.isclose(cov_frac, 0.1)


# ---------------------------------------------------------------------------
# Additional coverage: facility.capacitated_location_allocation
# ---------------------------------------------------------------------------


def test_capacitated_location_allocation_validation_errors():
    facilities = np.array([[0.0, 0.0], [10.0, 0.0]])
    capacities = np.array([150.0, 200.0])
    demands = np.array([[1.0, 0.0], [9.0, 0.0]])
    pop = np.array([100.0, 150.0])

    with pytest.raises(ValueError, match="facility_capacities"):
        capacitated_location_allocation(facilities, np.ones(3), demands, pop)

    with pytest.raises(ValueError, match="demand_coords"):
        capacitated_location_allocation(facilities, capacities, np.ones((2, 3)), pop)

    with pytest.raises(ValueError, match="demand_pop"):
        capacitated_location_allocation(facilities, capacities, demands, np.ones(3))


def test_capacitated_location_allocation_empty_facilities_or_demands():
    demands = np.array([[1.0, 1.0], [11.0, 11.0], [25.0, 25.0]])
    pop = np.array([100.0, 200.0, 500.0])

    # No facilities at all: every demand point should be unassigned
    allocations, unassigned, usage = capacitated_location_allocation(
        np.zeros((0, 2)), np.zeros(0), demands, pop
    )
    assert allocations == {}
    assert list(unassigned) == [0, 1, 2]
    assert usage.shape == (0,)

    # No demand points at all: nothing to allocate, no facilities used
    facilities = np.array([[0.0, 0.0], [10.0, 10.0]])
    capacities = np.array([10.0, 10.0])
    allocations2, unassigned2, usage2 = capacitated_location_allocation(
        facilities, capacities, np.zeros((0, 2)), np.zeros(0)
    )
    assert allocations2 == {}
    assert list(unassigned2) == []
    np.testing.assert_allclose(usage2, [0.0, 0.0])


def test_aras_method():
    X = np.array([[10.0, 2.0], [5.0, 8.0], [1.0, 10.0]])
    w = np.array([0.6, 0.4])

    k, ranks = aras_method(X, w)

    assert len(k) == 3
    assert len(ranks) == 3

    with pytest.raises(ValueError, match="weights length"):
        aras_method(X, w[:1])


def test_copras_method():
    X = np.array([[10.0, 2.0], [5.0, 8.0], [1.0, 10.0]])
    w = np.array([0.6, 0.4])

    n_ut, ranks = copras_method(X, w)

    assert len(n_ut) == 3
    assert len(ranks) == 3
    assert np.max(n_ut) == pytest.approx(100.0)

    with pytest.raises(ValueError, match="weights length"):
        copras_method(X, w[:1])


def test_edas_method():
    X = np.array([[10.0, 2.0], [5.0, 8.0], [1.0, 10.0]])
    w = np.array([0.6, 0.4])

    scores, ranks = edas_method(X, w)

    assert len(scores) == 3
    assert len(ranks) == 3
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)
    assert set(ranks) == {1, 2, 3}

    with pytest.raises(ValueError, match="weights length"):
        edas_method(X, w[:1])


def test_edas_method_cost_criteria():
    X = np.array([[10.0, 2.0], [5.0, 8.0], [1.0, 10.0]])
    w = np.array([0.5, 0.5])
    dirs = np.array([1.0, -1.0])

    scores, ranks = edas_method(X, w, directions=dirs)

    assert len(scores) == 3
    assert ranks[0] == 1  # alt 0 has highest benefit, lowest cost


def test_evaluate_tod_node_suitability():
    freq = np.array([10.0, 50.0, 100.0])
    density = np.array([20.0, 100.0, 200.0])
    entropy = np.array([0.2, 0.5, 0.8])
    walk = np.array([30.0, 60.0, 90.0])
    parking = np.array([0.5, 0.2, 0.1])

    res = evaluate_tod_node_suitability(
        station_transit_frequency=freq,
        surrounding_population_density=density,
        land_use_mix_entropy=entropy,
        walkability_pedestrian_score=walk,
        parking_supply_ratio=parking,
    )

    assert "tod_scores" in res
    assert "tier_1_count" in res
    assert "tier_2_count" in res
    assert "tier_3_count" in res
    assert "tod_ranking" in res

    assert res["tod_scores"].shape == (3,)
    assert len(res["tod_ranking"]) == 3

    # rank 1 should be the 3rd element as it has highest freq, density, entropy, walk
    assert res["tod_ranking"][2] == 1

    # Error checking
    with pytest.raises(ValueError, match="must be > 0"):
        evaluate_tod_node_suitability(
            np.array([0.0]), np.array([1.0]), np.array([0.5]), np.array([50.0]), np.array([0.2])
        )

    with pytest.raises(ValueError, match="must be > 0"):
        evaluate_tod_node_suitability(
            np.array([1.0]), np.array([0.0]), np.array([0.5]), np.array([50.0]), np.array([0.2])
        )

    with pytest.raises(ValueError, match="must be in"):
        evaluate_tod_node_suitability(
            np.array([1.0]), np.array([1.0]), np.array([1.5]), np.array([50.0]), np.array([0.2])
        )

    with pytest.raises(ValueError, match="must be in"):
        evaluate_tod_node_suitability(
            np.array([1.0]), np.array([1.0]), np.array([0.5]), np.array([150.0]), np.array([0.2])
        )

    with pytest.raises(ValueError, match="must be a 1D array"):
        evaluate_tod_node_suitability(
            np.array([[1.0]]), np.array([1.0]), np.array([0.5]), np.array([50.0]), np.array([0.2])
        )


def test_ev_fleet_charging_location_allocation_normal():
    from planx.suitability import ev_fleet_charging_location_allocation

    f_orig = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
    f_dest = np.array([[10.0, 0.0], [11.0, 1.0], [15.0, 5.0]])
    c_dep = np.array([[5.0, 0.0], [8.0, 2.0], [12.0, 4.0]])

    res = ev_fleet_charging_location_allocation(
        fleet_origins=f_orig,
        fleet_destinations=f_dest,
        candidate_depots=c_dep,
        num_depots_to_select=2,
        max_detour_km=15.0,
    )

    assert "selected_depot_indices" in res
    assert "trip_allocations" in res
    assert "fleet_coverage_ratio" in res
    assert "mean_detour_km" in res
    assert "depot_power_utilization_kw" in res
    assert "total_detour_km" in res

    assert len(res["selected_depot_indices"]) == 2
    assert len(res["trip_allocations"]) == 3
    assert 0.0 <= res["fleet_coverage_ratio"] <= 1.0


def test_ev_fleet_charging_location_allocation_validation():
    from planx.suitability import ev_fleet_charging_location_allocation

    f_orig = np.array([[0.0, 0.0], [1.0, 1.0]])
    f_dest = np.array([[10.0, 0.0], [11.0, 1.0]])
    c_dep = np.array([[5.0, 0.0]])

    with pytest.raises(ValueError, match="fleet_origins must be a 2D array"):
        ev_fleet_charging_location_allocation(np.array([0.0, 0.0]), f_dest, c_dep, 1)

    with pytest.raises(ValueError, match="num_depots_to_select must be between 1 and M"):
        ev_fleet_charging_location_allocation(f_orig, f_dest, c_dep, 2)


def test_tod_spatial_diversity_index():
    from planx.suitability import tod_spatial_diversity_index

    landuse = np.array(
        [
            [0.4, 0.4, 0.2],
            [0.8, 0.1, 0.1],
            [0.3, 0.3, 0.4],
        ]
    )
    far = np.array([2.5, 1.0, 3.5])
    dist = np.array([100.0, 600.0, 200.0])

    res = tod_spatial_diversity_index(landuse, far, dist)

    assert "shannon_entropy_scores" in res
    assert "tod_diversity_scores" in res
    assert "mean_tod_score" in res
    assert len(res["tod_diversity_scores"]) == 3


def test_logistics_microhub_location_allocation():
    from planx.suitability import logistics_microhub_location_allocation

    demand = np.array([[100.0, 100.0], [500.0, 500.0], [800.0, 800.0]])
    vols = np.array([10.0, 20.0, 15.0])
    cands = np.array([[0.0, 0.0], [400.0, 400.0], [900.0, 900.0]])

    res = logistics_microhub_location_allocation(
        demand, vols, cands, num_hubs_to_select=2, max_cargo_bike_range_km=5.0
    )

    assert "selected_hub_indices" in res
    assert "demand_allocations" in res
    assert "total_delivery_vkt" in res
    assert len(res["selected_hub_indices"]) == 2
    assert len(res["demand_allocations"]) == 3


def test_area_weighted_kmeans():
    from planx.suitability import area_weighted_kmeans

    pts = [(0.0, 0.0), (1.0, 1.0), (10.0, 10.0), (11.0, 11.0)]
    wts = [100.0, 150.0, 200.0, 250.0]

    res = area_weighted_kmeans(pts, wts, k=2)
    assert "labels" in res
    assert "centers" in res
    assert len(res["labels"]) == 4
    assert len(res["centers"]) == 2


def test_label_components_and_rank_sites():
    from planx.suitability import label_components, rank_sites

    mask = np.array(
        [
            [True, True, False, False],
            [True, False, False, False],
            [False, False, True, True],
            [False, False, True, True],
        ]
    )
    values = np.array(
        [
            [80.0, 90.0, 10.0, 10.0],
            [70.0, 10.0, 10.0, 10.0],
            [10.0, 10.0, 95.0, 100.0],
            [10.0, 10.0, 90.0, 95.0],
        ]
    )

    labels, count = label_components(mask)
    assert count == 2
    assert labels.shape == (4, 4)

    sites = rank_sites(labels, count, values, cell_area_m2=100.0, min_area_ha=0.0, top_n=2)
    assert len(sites) == 2
    assert sites[0]["mean"] >= sites[1]["mean"]


def test_macroform_generators():
    from planx.suitability import (
        gen_arti,
        gen_avlu,
        gen_C,
        gen_dikdortgen,
        gen_E,
        gen_H,
        gen_L,
        gen_T,
        gen_U,
    )

    rect = gen_dikdortgen(20.0, 30.0)
    assert rect["type_name"] == "dikdortgen"
    assert len(rect["coordinates"]) == 5

    l_shape = gen_L(30.0, 30.0)
    assert l_shape["type_name"] == "L"
    assert len(l_shape["coordinates"]) == 7

    u_shape = gen_U(40.0, 40.0)
    assert u_shape["type_name"] == "U"
    assert len(u_shape["coordinates"]) == 9

    t_shape = gen_T(30.0, 30.0)
    assert t_shape["type_name"] == "T"

    h_shape = gen_H(30.0, 30.0)
    assert h_shape["type_name"] == "H"

    avlu_shape = gen_avlu(40.0, 40.0)
    assert avlu_shape["type_name"] == "avlu"
    assert len(avlu_shape["courtyard_coordinates"]) == 5

    c_shape = gen_C(30.0, 30.0)
    assert c_shape["type_name"] == "C"

    e_shape = gen_E(30.0, 40.0)
    assert e_shape["type_name"] == "E"

    arti_shape = gen_arti(30.0, 30.0)
    assert arti_shape["type_name"] == "arti"
