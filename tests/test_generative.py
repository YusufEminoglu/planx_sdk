# -*- coding: utf-8 -*-
"""Unit tests for planx.generative submodule."""

from planx.generative import (
    recursive_parcel_subdivision,
    solar_envelope_buildable_volume,
)


def test_recursive_parcel_subdivision():
    res = recursive_parcel_subdivision(100.0, 100.0, min_parcel_area_m2=500.0)
    assert "parcels" in res
    assert res["parcel_count"] > 1


def test_solar_envelope_buildable_volume():
    res = solar_envelope_buildable_volume(20.0, 30.0, sun_altitude_angle_deg=45.0)
    assert "buildable_volume_m3" in res
    assert res["buildable_volume_m3"] > 0.0
