# -*- coding: utf-8 -*-
"""Unit tests for planx.urban_morphology submodule."""

import numpy as np

from planx.urban_morphology import (
    block_porosity_and_grain_index,
    fractal_dimension_box_counting,
    spacemate_density_matrix,
)


def test_spacemate_density_matrix():
    fsi = np.array([2.5, 0.8, 1.5])
    gsi = np.array([0.45, 0.25, 0.35])

    res = spacemate_density_matrix(fsi, gsi)
    assert "osr" in res
    assert "typology_class" in res
    assert len(res["osr"]) == 3


def test_fractal_dimension_box_counting():
    grid = np.zeros((16, 16), dtype=bool)
    grid[4:12, 4:12] = True

    res = fractal_dimension_box_counting(grid)
    assert "fractal_dimension" in res
    assert res["fractal_dimension"] >= 0.0


def test_block_porosity_and_grain_index():
    b_areas = np.array([200.0, 300.0, 150.0])
    block_area = 2000.0
    block_perim = 180.0

    res = block_porosity_and_grain_index(b_areas, block_area, block_perim)
    assert "porosity_ratio" in res
    assert 0.0 <= res["porosity_ratio"] <= 1.0
