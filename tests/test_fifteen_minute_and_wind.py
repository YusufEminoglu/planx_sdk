# -*- coding: utf-8 -*-
"""Unit tests for 15-Minute City and Wind Corridor simulation engines in PlanX."""

from __future__ import annotations

import unittest
from planx.spatial.fifteen_minute_city import (
    FifteenMinuteCityResult,
    evaluate_15min_city,
)
from planx.urban_physics.wind_corridor import (
    WindCorridorResult,
    simulate_wind_corridors,
)


class TestFifteenMinuteAndWind(unittest.TestCase):
    def test_evaluate_15min_city_multi_dimension(self) -> None:
        zones = [
            {"id": "Z1", "x": 0.0, "y": 0.0},
            {"id": "Z2", "x": 500.0, "y": 500.0},
            {"id": "Z3", "x": 3000.0, "y": 3000.0},  # Remote zone
        ]
        amenities = [
            {"id": "A1", "x": 50.0, "y": 50.0, "category": "supplying"},
            {"id": "A2", "x": 100.0, "y": 100.0, "category": "caring"},
            {"id": "A3", "x": 120.0, "y": 80.0, "category": "learning"},
            {"id": "A4", "x": 80.0, "y": 150.0, "category": "enjoying"},
            {"id": "A5", "x": 200.0, "y": 200.0, "category": "working"},
            {"id": "A6", "x": 150.0, "y": 100.0, "category": "living"},
        ]

        result = evaluate_15min_city(zones, amenities, walking_speed_kmh=4.8)

        self.assertIsInstance(result, FifteenMinuteCityResult)
        self.assertEqual(result.total_zones, 3)
        self.assertGreaterEqual(result.mean_composite_score, 0.0)
        self.assertLessEqual(result.mean_composite_score, 100.0)

        # Z1 should have higher score than remote Z3
        self.assertGreater(result.zone_scores[0].composite_15m_score, result.zone_scores[2].composite_15m_score)
        self.assertIn("Z3", result.deficit_zone_ids)

        d = result.to_dict()
        self.assertIn("mean_composite_score", d)
        self.assertIn("equity_gini", d)

    def test_simulate_wind_corridors(self) -> None:
        buildings = [
            {"footprint": [(10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)], "height": 25.0},
            {"footprint": [(60.0, 10.0), (80.0, 10.0), (80.0, 30.0), (60.0, 30.0)], "height": 18.0},
        ]
        bbox = (0.0, 0.0, 100.0, 100.0)

        res = simulate_wind_corridors(
            buildings=buildings,
            domain_bbox=bbox,
            wind_direction_degrees=45.0,
            reference_wind_speed_ms=6.0,
            grid_resolution=20.0,
        )

        self.assertIsInstance(res, WindCorridorResult)
        self.assertEqual(res.domain_rows, 5)
        self.assertEqual(res.domain_cols, 5)
        self.assertEqual(len(res.cells), 25)
        self.assertGreater(res.mean_velocity_ratio, 0.0)

        d = res.to_dict()
        self.assertIn("mean_velocity_ratio", d)
        self.assertIn("corridor_cells_count", d)
