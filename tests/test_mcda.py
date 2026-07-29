import numpy as np
import pytest

from planx.suitability.mcda import fuzzy_topsis_method


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
