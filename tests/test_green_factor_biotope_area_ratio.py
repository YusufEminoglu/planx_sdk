# -*- coding: utf-8 -*-
"""
Tests for Berlin Biotope Area Factor (BAF) and Malmö Green Factor Engine.
"""

from __future__ import annotations

import math
import pytest

from planx.resilience.green_factor_biotope_area_ratio import (
    BiotopeAreaReport,
    BiotopeSurfaceElement,
    SurfaceEcologicalType,
    calculate_biotope_area_factor,
)


def test_basic_baf_calculation():
    """Verify BAF calculation on a standard 1,000 m2 parcel."""
    surfaces = [
        BiotopeSurfaceElement(SurfaceEcologicalType.SEALED_IMPERVIOUS, 400.0, "Building footprint"),
        BiotopeSurfaceElement(SurfaceEcologicalType.SEMI_PERVIOUS_PAVER, 200.0, "Permeable driveway"),
        BiotopeSurfaceElement(SurfaceEcologicalType.EXTENSIVE_GREEN_ROOF, 200.0, "Sedum roof"),
        BiotopeSurfaceElement(SurfaceEcologicalType.VEGETATION_SOIL_CONNECTED, 200.0, "Garden lawn and trees"),
    ]

    report = calculate_biotope_area_factor(
        parcel_area_m2=1000.0,
        surfaces=surfaces,
        target_baf_threshold=0.35,
    )

    # Weights: Sealed=0.0, Paver=0.3, ExtGreenRoof=0.5, SoilVeg=1.0
    # Expected effective green area:
    # 400*0.0 + 200*0.3 + 200*0.5 + 200*1.0 = 0 + 60 + 100 + 200 = 360 m2
    assert pytest.approx(report.total_effective_green_area_m2, rel=1e-5) == 360.0
    assert pytest.approx(report.biotope_area_factor, rel=1e-5) == 0.36
    assert report.is_compliant is True
    assert report.target_threshold == 0.35

    # Runoff coefficients: Sealed=0.90, Paver=0.50, ExtGreenRoof=0.40, SoilVeg=0.15
    # Total runoff: 400*0.90 + 200*0.50 + 200*0.40 + 200*0.15 = 360 + 100 + 80 + 30 = 570
    # C_comp = 570 / 1000 = 0.57
    assert pytest.approx(report.composite_runoff_coefficient, rel=1e-5) == 0.57

    # Runoff reduction vs baseline 0.90: (0.90 - 0.57) / 0.90 * 100 = 36.6667%
    assert pytest.approx(report.runoff_reduction_percentage, rel=1e-3) == 36.667

    # Shannon index > 0 because multiple surface types exist
    assert report.biodiversity_shannon_index > 0.0

    # Evapotranspiration cooling score should be positive
    assert 0.0 < report.evapotranspiration_cooling_score <= 10.0


def test_baf_non_compliant_threshold():
    """Verify compliance flag turns False when BAF is below statutory threshold."""
    surfaces = [
        BiotopeSurfaceElement(SurfaceEcologicalType.SEALED_IMPERVIOUS, 900.0),
        BiotopeSurfaceElement(SurfaceEcologicalType.VEGETATION_SOIL_CONNECTED, 100.0),
    ]

    report = calculate_biotope_area_factor(
        parcel_area_m2=1000.0,
        surfaces=surfaces,
        target_baf_threshold=0.60,
    )

    # 100 * 1.0 / 1000 = 0.10 < 0.60
    assert pytest.approx(report.biotope_area_factor, rel=1e-5) == 0.10
    assert report.is_compliant is False


def test_custom_weights_and_coefficients():
    """Verify user-defined custom ecological weights and runoff coefficients."""
    elem = BiotopeSurfaceElement(
        surface_type=SurfaceEcologicalType.EXTENSIVE_GREEN_ROOF,
        area_m2=150.0,
        custom_weight=0.75,
        custom_runoff_coeff=0.20,
    )

    assert elem.weight == 0.75
    assert elem.runoff_coefficient == 0.20
    assert pytest.approx(elem.effective_ecological_area_m2, rel=1e-5) == 112.5

    report = calculate_biotope_area_factor(
        parcel_area_m2=300.0,
        surfaces=[elem, BiotopeSurfaceElement(SurfaceEcologicalType.SEALED_IMPERVIOUS, 150.0)],
    )
    # Total effective = 112.5 + 0.0 = 112.5
    assert pytest.approx(report.biotope_area_factor, rel=1e-5) == 112.5 / 300.0


def test_dict_input_support():
    """Verify function accepts dictionaries as surface inputs."""
    dict_surfaces = [
        {"surface_type": "sealed_impervious", "area_m2": 500.0},
        {"surface_type": "water_retention_pond", "area_m2": 200.0},
        {"surface_type": "intensive_green_roof", "area_m2": 300.0},
    ]

    report = calculate_biotope_area_factor(
        parcel_area_m2=1000.0,
        surfaces=dict_surfaces,
        target_baf_threshold=0.40,
    )

    # Pond w=1.0 (200m2), Intensive w=0.7 (210m2) -> total = 410m2 -> BAF = 0.41
    assert pytest.approx(report.biotope_area_factor, rel=1e-5) == 0.41
    assert report.is_compliant is True

    # Test dictionary export
    d = report.to_dict()
    assert d["biotope_area_factor"] == 0.41
    assert d["is_compliant"] is True
    assert "surface_breakdown" in d


def test_validation_errors():
    """Verify input validation handles invalid parcel area and empty surfaces."""
    with pytest.raises(ValueError, match="parcel_area_m2 must be positive"):
        calculate_biotope_area_factor(0.0, [BiotopeSurfaceElement(SurfaceEcologicalType.SEALED_IMPERVIOUS, 100.0)])

    with pytest.raises(ValueError, match="parcel_area_m2 must be positive"):
        calculate_biotope_area_factor(-500.0, [BiotopeSurfaceElement(SurfaceEcologicalType.SEALED_IMPERVIOUS, 100.0)])

    with pytest.raises(ValueError, match="surfaces sequence cannot be empty"):
        calculate_biotope_area_factor(1000.0, [])
