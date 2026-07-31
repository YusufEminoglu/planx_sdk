# -*- coding: utf-8 -*-
"""Unit tests for planx.urban_physics submodule."""

import numpy as np

from planx.urban_physics import (
    frontal_area_index_canopy,
    surface_albedo_cooling_potential,
)


def test_frontal_area_index_canopy():
    heights = np.array([12.0, 18.0, 24.0])
    widths = np.array([10.0, 15.0, 20.0])
    lot_area = 5000.0

    res = frontal_area_index_canopy(heights, widths, lot_area)
    assert "frontal_area_index_lambda_f" in res
    assert res["frontal_area_index_lambda_f"] > 0.0


def test_surface_albedo_cooling_potential():
    albedo = np.array([0.1, 0.15, 0.2])

    res = surface_albedo_cooling_potential(albedo, target_albedo=0.45)
    assert "mean_temperature_drop_c" in res
    assert res["mean_temperature_drop_c"] > 0.0
