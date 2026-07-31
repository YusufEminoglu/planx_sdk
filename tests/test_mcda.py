import numpy as np
import pytest

from planx.suitability.mcda import fuzzy_topsis_method, fuzzy_vikor_method


def test_fuzzy_topsis_method_basic():
    # 2 alternatives, 2 criteria
    l_mat = np.array([[1.0, 2.0], [3.0, 1.0]])
    m_mat = np.array([[2.0, 3.0], [4.0, 2.0]])
    u_mat = np.array([[3.0, 4.0], [5.0, 3.0]])

    wl = np.array([0.2, 0.3])
    wm = np.array([0.3, 0.4])
    wu = np.array([0.4, 0.5])

    # Both benefit
    ctype = np.array([1, 1])

    res = fuzzy_topsis_method(l_mat, m_mat, u_mat, wl, wm, wu, ctype)

    assert "closeness_coefficients" in res
    assert "ranking" in res
    assert res["closeness_coefficients"].shape == (2,)
    assert res["ranking"].shape == (2,)
    assert res["weighted_matrix_l"].shape == (2, 2)

    # Basic shape validation
    assert res["distance_positive"].shape == (2,)
    assert res["distance_negative"].shape == (2,)


def test_fuzzy_topsis_method_with_cost():
    l_mat = np.array([[1.0, 2.0], [3.0, 1.0]])
    m_mat = np.array([[2.0, 3.0], [4.0, 2.0]])
    u_mat = np.array([[3.0, 4.0], [5.0, 3.0]])

    wl = np.array([0.2, 0.3])
    wm = np.array([0.3, 0.4])
    wu = np.array([0.4, 0.5])

    # 1 benefit, 1 cost
    ctype = np.array([1, -1])

    res = fuzzy_topsis_method(l_mat, m_mat, u_mat, wl, wm, wu, ctype)

    assert len(res["closeness_coefficients"]) == 2


def test_fuzzy_topsis_method_validation_shapes():
    l_mat = np.array([[1.0, 2.0], [3.0, 1.0]])
    m_mat = np.array([[2.0, 3.0]])  # wrong shape
    u_mat = np.array([[3.0, 4.0], [5.0, 3.0]])

    wl = np.array([0.2, 0.3])
    wm = np.array([0.3, 0.4])
    wu = np.array([0.4, 0.5])
    ctype = np.array([1, 1])

    with pytest.raises(ValueError, match="same shape"):
        fuzzy_topsis_method(l_mat, m_mat, u_mat, wl, wm, wu, ctype)


def test_fuzzy_topsis_method_validation_weights_len():
    l_mat = np.array([[1.0, 2.0], [3.0, 1.0]])
    m_mat = np.array([[2.0, 3.0], [4.0, 2.0]])
    u_mat = np.array([[3.0, 4.0], [5.0, 3.0]])

    wl = np.array([0.2])  # wrong len
    wm = np.array([0.3, 0.4])
    wu = np.array([0.4, 0.5])
    ctype = np.array([1, 1])

    with pytest.raises(ValueError, match="length equal to the number of criteria"):
        fuzzy_topsis_method(l_mat, m_mat, u_mat, wl, wm, wu, ctype)


def test_fuzzy_topsis_method_validation_cost_zero():
    l_mat = np.array([[0.0, 2.0], [3.0, 1.0]])  # 0.0 in cost criteria
    m_mat = np.array([[2.0, 3.0], [4.0, 2.0]])
    u_mat = np.array([[3.0, 4.0], [5.0, 3.0]])

    wl = np.array([0.2, 0.3])
    wm = np.array([0.3, 0.4])
    wu = np.array([0.4, 0.5])
    ctype = np.array([-1, 1])

    with pytest.raises(ValueError, match="cannot have zero values"):
        fuzzy_topsis_method(l_mat, m_mat, u_mat, wl, wm, wu, ctype)


def test_fuzzy_topsis_method_validation_tfn_constraints():
    l_mat = np.array([[3.0, 2.0], [3.0, 1.0]])
    m_mat = np.array([[2.0, 3.0], [4.0, 2.0]])  # l > m for first element
    u_mat = np.array([[3.0, 4.0], [5.0, 3.0]])

    wl = np.array([0.2, 0.3])
    wm = np.array([0.3, 0.4])
    wu = np.array([0.4, 0.5])
    ctype = np.array([1, 1])

    with pytest.raises(ValueError, match="satisfy l <= m <= u"):
        fuzzy_topsis_method(l_mat, m_mat, u_mat, wl, wm, wu, ctype)


def test_fuzzy_topsis_method_validation_weight_tfn():
    l_mat = np.array([[1.0, 2.0], [3.0, 1.0]])
    m_mat = np.array([[2.0, 3.0], [4.0, 2.0]])
    u_mat = np.array([[3.0, 4.0], [5.0, 3.0]])

    wl = np.array([0.4, 0.3])  # l > m
    wm = np.array([0.3, 0.4])
    wu = np.array([0.4, 0.5])
    ctype = np.array([1, 1])

    with pytest.raises(ValueError, match="satisfy l <= m <= u"):
        fuzzy_topsis_method(l_mat, m_mat, u_mat, wl, wm, wu, ctype)


def test_fuzzy_topsis_method_validation_negative_weight():
    l_mat = np.array([[1.0, 2.0], [3.0, 1.0]])
    m_mat = np.array([[2.0, 3.0], [4.0, 2.0]])
    u_mat = np.array([[3.0, 4.0], [5.0, 3.0]])

    wl = np.array([-0.1, 0.3])  # negative
    wm = np.array([0.3, 0.4])
    wu = np.array([0.4, 0.5])
    ctype = np.array([1, 1])

    with pytest.raises(ValueError, match="non-negative"):
        fuzzy_topsis_method(l_mat, m_mat, u_mat, wl, wm, wu, ctype)


def test_fuzzy_topsis_method_validation_ctype():
    l_mat = np.array([[1.0, 2.0], [3.0, 1.0]])
    m_mat = np.array([[2.0, 3.0], [4.0, 2.0]])
    u_mat = np.array([[3.0, 4.0], [5.0, 3.0]])

    wl = np.array([0.2, 0.3])
    wm = np.array([0.3, 0.4])
    wu = np.array([0.4, 0.5])
    ctype = np.array([1, 0])  # invalid ctype

    with pytest.raises(ValueError, match="contain only 1 \\(benefit\\) or -1 \\(cost\\)"):
        fuzzy_topsis_method(l_mat, m_mat, u_mat, wl, wm, wu, ctype)


def test_fuzzy_vikor_method_basic():
    l_mat = np.array([[1.0, 2.0], [3.0, 1.0]])
    m_mat = np.array([[2.0, 3.0], [4.0, 2.0]])
    u_mat = np.array([[3.0, 4.0], [5.0, 3.0]])

    wl = np.array([0.2, 0.3])
    wm = np.array([0.3, 0.4])
    wu = np.array([0.4, 0.5])

    ctype = np.array([1, 1])

    res = fuzzy_vikor_method(l_mat, m_mat, u_mat, wl, wm, wu, ctype)

    assert "Q" in res
    assert "S" in res
    assert "R" in res
    assert "ranking" in res
    assert "compromise_set" in res
    assert res["Q"].shape == (2,)
    assert res["ranking"].shape == (2,)
    assert isinstance(res["compromise_set"], list)


def test_fuzzy_vikor_method_with_cost():
    l_mat = np.array([[1.0, 2.0], [3.0, 1.0]])
    m_mat = np.array([[2.0, 3.0], [4.0, 2.0]])
    u_mat = np.array([[3.0, 4.0], [5.0, 3.0]])

    wl = np.array([0.2, 0.3])
    wm = np.array([0.3, 0.4])
    wu = np.array([0.4, 0.5])

    ctype = np.array([1, -1])

    res = fuzzy_vikor_method(l_mat, m_mat, u_mat, wl, wm, wu, ctype)

    assert len(res["Q"]) == 2


def test_fuzzy_vikor_method_validation_shapes():
    l_mat = np.array([[1.0, 2.0], [3.0, 1.0]])
    m_mat = np.array([[2.0, 3.0]])  # wrong shape
    u_mat = np.array([[3.0, 4.0], [5.0, 3.0]])

    wl = np.array([0.2, 0.3])
    wm = np.array([0.3, 0.4])
    wu = np.array([0.4, 0.5])
    ctype = np.array([1, 1])

    with pytest.raises(ValueError, match="same shape"):
        fuzzy_vikor_method(l_mat, m_mat, u_mat, wl, wm, wu, ctype)


def test_fuzzy_vikor_method_validation_weights_len():
    l_mat = np.array([[1.0, 2.0], [3.0, 1.0]])
    m_mat = np.array([[2.0, 3.0], [4.0, 2.0]])
    u_mat = np.array([[3.0, 4.0], [5.0, 3.0]])

    wl = np.array([0.2])  # wrong len
    wm = np.array([0.3, 0.4])
    wu = np.array([0.4, 0.5])
    ctype = np.array([1, 1])

    with pytest.raises(ValueError, match="length equal to the number of criteria"):
        fuzzy_vikor_method(l_mat, m_mat, u_mat, wl, wm, wu, ctype)


def test_fuzzy_vikor_method_validation_tfn_constraints():
    l_mat = np.array([[3.0, 2.0], [3.0, 1.0]])
    m_mat = np.array([[2.0, 3.0], [4.0, 2.0]])  # l > m for first element
    u_mat = np.array([[3.0, 4.0], [5.0, 3.0]])

    wl = np.array([0.2, 0.3])
    wm = np.array([0.3, 0.4])
    wu = np.array([0.4, 0.5])
    ctype = np.array([1, 1])

    with pytest.raises(ValueError, match="satisfy l <= m <= u"):
        fuzzy_vikor_method(l_mat, m_mat, u_mat, wl, wm, wu, ctype)


def test_fuzzy_vikor_method_validation_v_range():
    l_mat = np.array([[1.0, 2.0], [3.0, 1.0]])
    m_mat = np.array([[2.0, 3.0], [4.0, 2.0]])
    u_mat = np.array([[3.0, 4.0], [5.0, 3.0]])

    wl = np.array([0.2, 0.3])
    wm = np.array([0.3, 0.4])
    wu = np.array([0.4, 0.5])
    ctype = np.array([1, 1])

    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        fuzzy_vikor_method(l_mat, m_mat, u_mat, wl, wm, wu, ctype, v=1.5)


def test_spotis_method_normal():
    from planx.suitability import spotis_method

    mat = np.array([[10.0, 200.0], [20.0, 150.0], [15.0, 300.0]])
    weights = [0.6, 0.4]
    types = ["+", "-"]  # criterion 0 benefit, criterion 1 cost

    res = spotis_method(matrix=mat, weights=weights, types=types)

    assert "scores" in res
    assert "ranks" in res
    assert "ideal_solution" in res
    assert "bounds" in res

    assert len(res["scores"]) == 3
    assert len(res["ranks"]) == 3
    assert res["ideal_solution"][0] == 20.0  # max benefit
    assert res["ideal_solution"][1] == 150.0  # min cost
    assert np.min(res["ranks"]) == 1
    assert np.max(res["ranks"]) == 3


def test_spotis_method_explicit_bounds():
    from planx.suitability import spotis_method

    mat = np.array([[10.0, 200.0], [20.0, 150.0]])
    weights = [0.5, 0.5]
    types = [1, 0]
    bounds = np.array([[0.0, 50.0], [100.0, 500.0]])

    res = spotis_method(matrix=mat, weights=weights, types=types, bounds=bounds)

    assert res["ideal_solution"][0] == 50.0
    assert res["ideal_solution"][1] == 100.0
    assert res["scores"].shape == (2,)


def test_spotis_method_validation():
    from planx.suitability import spotis_method

    mat = np.array([[10.0, 200.0], [20.0, 150.0]])
    weights = [0.5, 0.5]
    types = [1, 0]

    with pytest.raises(ValueError, match="matrix must be a 2D array"):
        spotis_method(np.array([10.0, 200.0]), weights, types)

    with pytest.raises(ValueError, match="bounds must be a \\(N, 2\\) array"):
        spotis_method(mat, weights, types, bounds=np.array([[0.0, 50.0]]))


def test_ivif_topsis_method_normal():
    from planx.suitability import ivif_topsis_method

    # 3 alternatives, 2 criteria, 4 IVIF values [mu_L, mu_U, nu_L, nu_U]
    ivif_mat = np.array(
        [
            [[0.5, 0.7, 0.1, 0.2], [0.4, 0.6, 0.2, 0.3]],
            [[0.6, 0.8, 0.1, 0.15], [0.5, 0.7, 0.1, 0.2]],
            [[0.3, 0.5, 0.3, 0.4], [0.2, 0.4, 0.4, 0.5]],
        ]
    )
    weights = [0.6, 0.4]
    types = ["+", "+"]

    res = ivif_topsis_method(ivif_matrix=ivif_mat, weights=weights, types=types)

    assert "closeness_coefficients" in res
    assert "ranks" in res
    assert "distance_to_pis" in res
    assert "distance_to_nis" in res

    assert len(res["closeness_coefficients"]) == 3
    assert len(res["ranks"]) == 3
    assert np.min(res["ranks"]) == 1
    assert np.max(res["ranks"]) == 3
    assert np.all((res["closeness_coefficients"] >= 0.0) & (res["closeness_coefficients"] <= 1.0))


def test_ivif_topsis_method_validation():
    from planx.suitability import ivif_topsis_method

    ivif_mat = np.array(
        [
            [[0.5, 0.7, 0.1, 0.2], [0.4, 0.6, 0.2, 0.3]],
            [[0.6, 0.8, 0.1, 0.15], [0.5, 0.7, 0.1, 0.2]],
        ]
    )

    with pytest.raises(ValueError, match="ivif_matrix must be a 3D array"):
        ivif_topsis_method(np.ones((2, 2)), [0.5, 0.5], ["+", "+"])

    with pytest.raises(ValueError, match="IVIF membership bounds"):
        invalid_mat = np.copy(ivif_mat)
        invalid_mat[0, 0, 0] = 0.8
        invalid_mat[0, 0, 1] = 0.5  # mu_L > mu_U
        ivif_topsis_method(invalid_mat, [0.5, 0.5], ["+", "+"])


def test_neutrosophic_waspas_method():
    from planx.suitability import neutrosophic_waspas_method

    dm = np.array(
        [
            [80.0, 90.0, 70.0],
            [60.0, 75.0, 85.0],
            [95.0, 60.0, 80.0],
        ]
    )
    w = np.array([0.4, 0.3, 0.3])

    res = neutrosophic_waspas_method(dm, w, lambda_param=0.5)

    assert "waspas_scores" in res
    assert "rankings" in res
    assert len(res["waspas_scores"]) == 3
    assert len(res["rankings"]) == 3


def test_if_vikor_method():
    from planx.suitability import if_vikor_method

    dm = np.array(
        [
            [8.0, 7.0, 6.0],
            [5.0, 9.0, 8.0],
            [7.0, 6.0, 9.0],
        ]
    )
    w = np.array([0.5, 0.3, 0.2])

    res = if_vikor_method(dm, w, v_preference=0.5)

    assert "s_scores" in res
    assert "r_scores" in res
    assert "q_scores" in res
    assert "rankings" in res
    assert len(res["q_scores"]) == 3
    assert len(res["rankings"]) == 3


def test_rough_topsis_method():
    from planx.suitability import rough_topsis_method

    lower = np.array(
        [
            [5.0, 6.0, 7.0],
            [4.0, 8.0, 5.0],
            [6.0, 5.0, 8.0],
        ]
    )
    upper = np.array(
        [
            [7.0, 8.0, 9.0],
            [6.0, 9.0, 7.0],
            [8.0, 7.0, 9.0],
        ]
    )
    w = np.array([0.4, 0.3, 0.3])

    res = rough_topsis_method(lower, upper, w)

    assert "closeness_coefficients" in res
    assert "rankings" in res
    assert len(res["closeness_coefficients"]) == 3
    assert len(res["rankings"]) == 3


def test_fuzzy_copras_method():
    from planx.suitability import fuzzy_copras_method

    fuzzy_mat = np.array(
        [
            [[0.2, 0.4, 0.6], [0.5, 0.7, 0.9]],
            [[0.4, 0.6, 0.8], [0.3, 0.5, 0.7]],
            [[0.6, 0.8, 1.0], [0.2, 0.4, 0.6]],
        ]
    )
    w = np.array([0.5, 0.5])
    types = ["+", "-"]

    res = fuzzy_copras_method(fuzzy_mat, w, types)

    assert "utility_degrees" in res
    assert "rankings" in res
    assert len(res["utility_degrees"]) == 3
    assert len(res["rankings"]) == 3


def test_picture_fuzzy_topsis():
    from planx.suitability import picture_fuzzy_topsis

    pf_mat = np.array(
        [
            [[0.7, 0.1, 0.1], [0.6, 0.2, 0.1]],
            [[0.5, 0.2, 0.2], [0.8, 0.1, 0.05]],
            [[0.4, 0.3, 0.2], [0.5, 0.3, 0.15]],
        ]
    )
    w = np.array([0.5, 0.5])

    res = picture_fuzzy_topsis(pf_mat, w)

    assert "closeness_coefficients" in res
    assert "rankings" in res
    assert len(res["closeness_coefficients"]) == 3
    assert len(res["rankings"]) == 3


def test_hesitant_fuzzy_dematel():
    from planx.suitability import hesitant_fuzzy_dematel

    m1 = np.array(
        [
            [0.0, 0.4, 0.2],
            [0.3, 0.0, 0.5],
            [0.1, 0.6, 0.0],
        ]
    )
    m2 = np.array(
        [
            [0.0, 0.5, 0.3],
            [0.2, 0.0, 0.4],
            [0.2, 0.7, 0.0],
        ]
    )

    res = hesitant_fuzzy_dematel([m1, m2])

    assert "total_relation_matrix" in res
    assert "prominence_d_plus_r" in res
    assert "relation_d_minus_r" in res
    assert "causal_classification" in res
    assert res["total_relation_matrix"].shape == (3, 3)
    assert len(res["causal_classification"]) == 3


def test_spherical_fuzzy_topsis():
    from planx.suitability import spherical_fuzzy_topsis

    sf_mat = np.array(
        [
            [[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]],
            [[0.5, 0.4, 0.2], [0.8, 0.1, 0.1]],
            [[0.4, 0.5, 0.2], [0.5, 0.4, 0.2]],
        ]
    )
    w = np.array([0.5, 0.5])

    res = spherical_fuzzy_topsis(sf_mat, w)

    assert "closeness_coefficients" in res
    assert "rankings" in res
    assert len(res["closeness_coefficients"]) == 3
    assert len(res["rankings"]) == 3
