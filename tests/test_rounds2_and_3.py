# -*- coding: utf-8 -*-
"""Unit tests for PlanX Rounds 2 and 3 features.

Covers Sponge City, Parcel Gen, Daylight SVF and TOD Optimizer.
"""

from __future__ import annotations

import unittest

from planx.climate.sponge_city import (
    SpongeCityResult,
    simulate_sponge_city_runoff,
)
from planx.generative.parcel_generator import (
    ParcelLayoutResult,
    generate_optimized_parcels,
)
from planx.suitability.tod_optimizer import (
    TODTypologyResult,
    evaluate_tod_potential,
)
from planx.urban_physics.daylight_svf import (
    DaylightAutonomyResult,
    compute_3d_sky_view_factor,
)


class TestPlanXRounds2And3(unittest.TestCase):
    def test_sponge_city_runoff_simulator(self) -> None:
        landuse = {"impervious": 30000.0, "green": 20000.0}
        lids = [
            {"type": "bioswale", "area_m2": 500.0, "storage_depth_mm": 200.0},
            {"type": "permeable_pavement", "area_m2": 1500.0, "storage_depth_mm": 100.0},
        ]
        res = simulate_sponge_city_runoff(landuse, lids, design_rainfall_depth_mm=70.0)

        self.assertIsInstance(res, SpongeCityResult)
        self.assertEqual(res.total_catchment_area_m2, 50000.0)
        self.assertGreater(res.runoff_reduction_percent, 0.0)
        self.assertLess(res.lid_mitigated_runoff_volume_m3, res.baseline_runoff_volume_m3)

        d = res.to_dict()
        self.assertIn("runoff_reduction_percent", d)

    def test_generative_parcel_layout_optimizer(self) -> None:
        block = [(0.0, 0.0), (120.0, 0.0), (120.0, 40.0), (0.0, 40.0)]
        layout = generate_optimized_parcels(block, target_parcel_area_m2=400.0, min_frontage_m=12.0)

        self.assertIsInstance(layout, ParcelLayoutResult)
        self.assertGreater(layout.total_parcels, 1)
        self.assertGreater(layout.efficiency_net_to_gross_ratio, 0.5)

        d = layout.to_dict()
        self.assertIn("total_parcels", d)

    def test_continuous_daylight_svf(self) -> None:
        buildings = [
            {"x": 20.0, "y": 20.0, "height": 25.0},
            {"x": 60.0, "y": 20.0, "height": 30.0},
        ]
        bbox = (0.0, 0.0, 80.0, 80.0)
        svf_res = compute_3d_sky_view_factor(buildings, domain_bbox=bbox, grid_resolution=20.0)

        self.assertIsInstance(svf_res, DaylightAutonomyResult)
        self.assertGreater(svf_res.mean_sky_view_factor, 0.0)
        self.assertLessEqual(svf_res.mean_sky_view_factor, 1.0)
        self.assertGreater(svf_res.daylight_autonomy_percent, 0.0)

    def test_tod_typology_optimizer(self) -> None:
        stations = [
            {
                "id": "ST_01",
                "peak_trips_per_hour": 28.0,
                "landuse_shares": {"res": 0.4, "com": 0.4, "civ": 0.2},
                "intersection_density_km2": 110.0,
            },
            {
                "id": "ST_02",
                "peak_trips_per_hour": 6.0,
                "landuse_shares": {"res": 0.9, "com": 0.1},
                "intersection_density_km2": 30.0,
            },
        ]
        res = evaluate_tod_potential(stations)

        self.assertIsInstance(res, TODTypologyResult)
        self.assertEqual(res.total_stations_evaluated, 2)
        self.assertGreater(
            res.station_scores[0].composite_tod_index, res.station_scores[1].composite_tod_index
        )
        self.assertGreater(
            res.station_scores[0].recommended_far_density,
            res.station_scores[1].recommended_far_density,
        )
