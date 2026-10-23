"""Tests for the social vulnerability and energy-poverty indices."""

import numpy as np
import pytest

from planx.resilience.social import social_vulnerability_index, urban_energy_vulnerability_index


def test_social_vulnerability_index_scores_and_classes():
    scores, classes = social_vulnerability_index(
        {
            "elderly": np.array([0.0, 50.0, 100.0]),
            "low_income": np.array([0.0, 50.0, 100.0]),
        },
        {"elderly": 1.0, "low_income": 1.0},
    )
    assert scores.tolist() == pytest.approx([0.0, 50.0, 100.0])
    assert classes == ["Low", "Moderate", "Very High"]


def test_social_vulnerability_index_handles_constant_and_zero_weight():
    scores, classes = social_vulnerability_index(
        {"constant": np.array([5.0, 5.0, 5.0]), "ignored": np.array([1.0, 2.0, 3.0])},
        {"constant": 1.0, "ignored": 0.0},
    )
    assert scores.tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert classes == ["Low", "Low", "Low"]


def test_social_vulnerability_index_rejects_invalid_input():
    with pytest.raises(ValueError):
        social_vulnerability_index({}, {})
    with pytest.raises(ValueError):
        social_vulnerability_index({"a": np.ones(2), "b": np.ones(3)}, {})


def test_urban_energy_vulnerability_index_summarises_risk_zones():
    result = urban_energy_vulnerability_index(
        median_income=np.array([1000.0, 5000.0, 9000.0]),
        building_energy_efficiency_score=np.array([10.0, 50.0, 90.0]),
        climate_exposure_score=np.array([100.0, 50.0, 0.0]),
        vulnerable_demographics_ratio=np.array([1.0, 0.5, 0.0]),
    )
    index = result["energy_vulnerability_index"]
    assert index.shape == (3,)
    assert np.all((index >= 0.0) & (index <= 100.0))
    assert (
        result["high_risk_zones_count"]
        + result["moderate_risk_zones_count"]
        + result["low_risk_zones_count"]
        == 3
    )
    assert 0.0 <= result["overall_energy_poverty_gini"] <= 1.0


def test_urban_energy_vulnerability_index_handles_empty_input():
    empty = np.array([])
    result = urban_energy_vulnerability_index(empty, empty, empty, empty)
    assert result["high_risk_zones_count"] == 0
    assert result["overall_energy_poverty_gini"] == 0.0


@pytest.mark.parametrize(
    "income, efficiency, exposure, demographics",
    [
        (np.array([0.0]), np.array([50.0]), np.array([50.0]), np.array([0.5])),
        (np.array([100.0]), np.array([101.0]), np.array([50.0]), np.array([0.5])),
        (np.array([100.0]), np.array([50.0]), np.array([-1.0]), np.array([0.5])),
        (np.array([100.0]), np.array([50.0]), np.array([50.0]), np.array([1.5])),
        (np.array([100.0, 200.0]), np.array([50.0]), np.array([50.0]), np.array([0.5])),
    ],
)
def test_urban_energy_vulnerability_index_rejects_invalid_input(
    income, efficiency, exposure, demographics
):
    with pytest.raises(ValueError):
        urban_energy_vulnerability_index(income, efficiency, exposure, demographics)
