# -*- coding: utf-8 -*-
"""Unit tests for planx.climate submodule."""

import numpy as np

from planx.climate import (
    carbon_sequestration_urban_canopy,
    stormwater_green_roof_retention_capacity,
)


def test_carbon_sequestration_urban_canopy():
    dbh = np.array([15.0, 25.0, 35.0, 45.0])
    res = carbon_sequestration_urban_canopy(dbh, canopy_cover_ha=2.5)
    assert "annual_co2_sequestration_tonnes" in res
    assert res["annual_co2_sequestration_tonnes"] > 0.0


def test_stormwater_green_roof_retention_capacity():
    res = stormwater_green_roof_retention_capacity(roof_area_m2=1000.0, rainfall_depth_mm=50.0)
    assert "retained_runoff_volume_m3" in res
    assert res["retained_runoff_volume_m3"] > 0.0
