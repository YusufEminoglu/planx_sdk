"""Tests for the land-use allocation engine and Pareto helpers."""

import numpy as np
import pytest

from planx.engine import allocate


def test_parse_targets_normalises_names_and_separators():
    targets = allocate.parse_targets("Residential=50000; green=30000.0")
    assert targets == [("residential", 50000.0), ("green", 30000.0)]


@pytest.mark.parametrize("text", ["", "residential", "res=abc", "res=-1", "=10"])
def test_parse_targets_rejects_invalid_entries(text):
    with pytest.raises(ValueError):
        allocate.parse_targets(text)


def test_allocate_land_use_performs_capacity_preserving_swap():
    suit = np.array([[10.0, 8.0], [9.0, 1.0]])
    result = allocate.allocate_land_use(suit, [10.0, 10.0], [10.0, 10.0])
    assert result["assign"].tolist() == [1, 0]
    assert result["objective"] == pytest.approx(170.0)
    assert result["swaps"] >= 1


def test_allocate_land_use_respects_locked_parcels():
    suit = np.array([[9.0, 1.0], [8.0, 2.0], [2.0, 8.0], [1.0, 9.0]])
    result = allocate.allocate_land_use(
        suit,
        np.full(4, 100.0),
        [200.0, 100.0],
        locked=np.array([-1, -1, 1, -1]),
    )
    assert result["assign"].tolist() == [0, 0, 1, -1]


@pytest.mark.parametrize(
    "suit, area, targets",
    [
        (np.zeros(3), [1.0], [1.0, 1.0]),
        (np.zeros((2, 2)), [1.0], [1.0, 1.0]),
        (np.zeros((2, 2)), [1.0, 1.0], [1.0]),
    ],
)
def test_allocate_land_use_rejects_mismatched_shapes(suit, area, targets):
    with pytest.raises(ValueError):
        allocate.allocate_land_use(suit, area, targets)


def test_allocate_multi_penalises_adjacent_uses():
    suit = np.array([[5.0, 5.0], [5.0, 6.0], [5.0, 5.0]])
    edges = [(0, 1, 10.0), (1, 2, 10.0)]
    result = allocate.allocate_multi(
        suit,
        np.ones(3),
        [2.0, 1.0],
        edges,
        np.array([[0.0, -1.0], [-1.0, 0.0]]),
    )
    assert result["assign"][1] == 0
    assert list(result["assign"]).count(1) == 1
    assert result["objective"] == pytest.approx(5.0)


def test_same_use_boundary_counts_only_matching_pairs():
    edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
    assert allocate._same_use_boundary([0, 0, 1, 1], edges) == pytest.approx(2.0)
    assert allocate._same_use_boundary([0, 1, 0, 1], edges) == 0.0


def test_check_connectivity_detects_disconnected_uses():
    adj = [[(1, 1.0)], [(0, 1.0)], [(3, 1.0)], [(2, 1.0)]]
    assert allocate.check_connectivity(0, np.array([0, 0]), [[(1, 1.0)], [(0, 1.0)]])
    assert not allocate.check_connectivity(0, np.array([0, -1, 0, -1]), adj)


def test_pareto_front_traces_suitability_compactness_tradeoff():
    suit = np.array([[10.0, 0.0], [0.0, 10.0], [10.0, 0.0], [0.0, 10.0]])
    edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
    result = allocate.pareto_front(
        suit,
        np.ones(4),
        [2.0, 2.0],
        edges,
        weights=[0.0, 5.0, 12.0, 20.0],
    )
    assert result["suit"][0] == pytest.approx(40.0)
    assert result["compact"][0] == pytest.approx(0.0)
    assert result["suit"][-1] == pytest.approx(20.0)
    assert result["compact"][-1] == pytest.approx(2.0)
    assert result["on_front"][0] and result["on_front"][-1]
    assert len(result["assign"]) == 4


def test_knee_index_needs_three_front_points():
    assert allocate._knee_index(np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.ones(2, bool)) == -1


def test_allocate_contiguous_keeps_each_use_connected():
    suit = np.zeros((16, 2))
    suit[0, 0] = suit[15, 0] = 10.0
    for p in range(16):
        if p not in (0, 15):
            suit[p, 1] = 5.0
            suit[p, 0] = 1.0
        else:
            suit[p, 1] = 1.0
    edges = []
    for row in range(4):
        for col in range(4):
            p = row * 4 + col
            if col < 3:
                edges.append((p, p + 1, 1.0))
            if row < 3:
                edges.append((p, p + 4, 1.0))
    result = allocate.allocate_contiguous(suit, np.full(16, 10.0), [20.0, 140.0], edges)
    adj = [[] for _ in range(16)]
    for i, j, length in edges:
        adj[i].append((j, length))
        adj[j].append((i, length))
    assert allocate.check_connectivity(0, result["assign"], adj)
    assert allocate.check_connectivity(1, result["assign"], adj)
    assert result["allocated"][0] == pytest.approx(20.0)
    assert result["allocated"][1] == pytest.approx(140.0)


def test_allocate_contiguous_warns_and_falls_back_when_infeasible():
    suit = np.array([[10.0, 1.0], [1.0, 10.0], [1.0, 1.0]])
    area = np.full(3, 10.0)
    edges = [(0, 1, 1.0), (1, 2, 1.0)]
    messages = []
    result = allocate.allocate_contiguous(
        suit,
        area,
        [30.0, 10.0],
        edges,
        log_warning_fn=messages.append,
    )
    assert messages and "infeasible" in messages[0].lower()
    assert result["assign"].shape == (3,)

    with pytest.warns(UserWarning, match="infeasible"):
        allocate.allocate_contiguous(suit, area, [30.0, 10.0], edges)
