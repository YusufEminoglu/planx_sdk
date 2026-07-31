# -*- coding: utf-8 -*-
"""Unit tests for planx.cellular_automata submodule."""

import numpy as np

from planx.cellular_automata import (
    markov_transition_probability_matrix,
    sleuth_cellular_automata_growth,
)


def test_sleuth_cellular_automata_growth():
    urban = np.zeros((10, 10), dtype=bool)
    urban[4, 4] = True
    slope = np.zeros((10, 10), dtype=np.float64)
    dist_road = np.ones((10, 10), dtype=np.float64) * 100.0
    excl = np.zeros((10, 10), dtype=bool)

    res = sleuth_cellular_automata_growth(urban, slope, dist_road, excl, steps=3)
    assert "simulated_urban_grid" in res
    assert "urban_cells_history" in res
    assert len(res["urban_cells_history"]) == 4


def test_markov_transition_probability_matrix():
    r1 = np.array([[0, 0, 1], [1, 2, 2]])
    r2 = np.array([[0, 1, 1], [1, 2, 3]])

    res = markov_transition_probability_matrix(r1, r2, num_classes=4)
    assert "transition_probability_matrix" in res
    assert res["transition_probability_matrix"].shape == (4, 4)
