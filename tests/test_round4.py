# -*- coding: utf-8 -*-
"""Unit tests for PlanX Round 4 features (UHI Mitigation & School Catchment)."""

from __future__ import annotations

import unittest

from planx.spatial.school_catchment import (
    SchoolCatchmentResult,
    optimize_school_catchments,
)
from planx.urban_physics.uhi_mitigation import (
    UHIMitigationResult,
    simulate_uhi_temperature_reduction,
)


class TestPlanXRound4(unittest.TestCase):
    def test_uhi_mitigation_simulation(self) -> None:
        res = simulate_uhi_temperature_reduction(
            urban_area_m2=100000.0,
            roof_area_m2=40000.0,
            pavement_area_m2=40000.0,
            cool_roof_fraction=0.60,
            tree_canopy_fraction=0.30,
            ambient_air_temp_c=36.0,
        )

        self.assertIsInstance(res, UHIMitigationResult)
        self.assertGreater(res.surface_temp_reduction_c, 0.0)
        self.assertGreater(res.air_temp_reduction_c, 0.0)
        self.assertLess(res.mitigated_air_temp_c, res.baseline_air_temp_c)
        self.assertGreater(res.annual_energy_savings_kwh, 0.0)

        d = res.to_dict()
        self.assertIn("surface_temp_reduction_c", d)
        self.assertIn("air_temp_reduction_c", d)

    def test_school_catchment_optimizer(self) -> None:
        schools = [
            {"id": "SCH_North", "capacity": 600, "position": (0.0, 500.0)},
            {"id": "SCH_South", "capacity": 600, "position": (0.0, -500.0)},
        ]
        zones = [
            {"id": "Z1", "students": 150, "position": (0.0, 400.0)},
            {"id": "Z2", "students": 200, "position": (0.0, 300.0)},
            {"id": "Z3", "students": 180, "position": (0.0, -400.0)},
            {"id": "Z4", "students": 220, "position": (0.0, -300.0)},
        ]

        result = optimize_school_catchments(schools, zones)
        self.assertIsInstance(result, SchoolCatchmentResult)
        self.assertEqual(result.total_schools, 2)
        self.assertEqual(result.total_students, 750)
        self.assertEqual(result.total_assigned_students, 750)
        self.assertGreater(result.mean_student_walk_distance_m, 0.0)
        self.assertEqual(result.overcapacity_schools_count, 0)
        self.assertGreaterEqual(result.gini_accessibility_index, 0.0)

        d = result.to_dict()
        self.assertIn("total_schools", d)
        self.assertIn("gini_equity", d)
