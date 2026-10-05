# -*- coding: utf-8 -*-
"""Unit tests for PlanX Round 9 features (Solid Waste Allocation & 2D Urban Flash Flood)."""

from __future__ import annotations

import unittest

from planx.infrastructure.solid_waste_route_allocator import (
    CollectionZoneDemand,
    WasteRouteResult,
    optimize_waste_collection_routes,
)
from planx.resilience.urban_flood_inundation_2d import (
    FloodInundationResult,
    SurfaceDepressionCell,
    simulate_urban_flash_flood,
)


class TestPlanXRound9(unittest.TestCase):
    def test_solid_waste_route_allocator(self) -> None:
        zones = [
            CollectionZoneDemand(
                "Zone_Kadıköy", bin_count=45, waste_volume_m3=8.5, coordinates_xy=(2.0, 3.0)
            ),
            CollectionZoneDemand(
                "Zone_Üsküdar", bin_count=60, waste_volume_m3=12.0, coordinates_xy=(4.0, 6.0)
            ),
            CollectionZoneDemand(
                "Zone_Ataşehir", bin_count=35, waste_volume_m3=7.0, coordinates_xy=(7.0, 5.0)
            ),
        ]

        res = optimize_waste_collection_routes(zones, compactor_truck_capacity_m3=16.0)

        self.assertIsInstance(res, WasteRouteResult)
        self.assertEqual(res.total_waste_collected_m3, 27.5)
        self.assertGreater(res.total_travel_distance_km, 0.0)
        self.assertGreater(res.fuel_consumption_liters, 0.0)
        self.assertGreaterEqual(res.landfill_trips_count, 2)

        d = res.to_dict()
        self.assertIn("waste_m3", d)
        self.assertIn("fuel_liters", d)

    def test_2d_urban_flash_flood(self) -> None:
        cells = [
            SurfaceDepressionCell(
                "Sink_1",
                ground_elevation_m=10.0,
                impervious_ratio=0.85,
                drainage_inlet_capacity_m3s=0.001,
                surface_area_m2=200.0,
            ),
            SurfaceDepressionCell(
                "Ridge_2",
                ground_elevation_m=14.0,
                impervious_ratio=0.60,
                drainage_inlet_capacity_m3s=0.02,
                surface_area_m2=200.0,
            ),
        ]

        res = simulate_urban_flash_flood(cells, storm_rainfall_mm=90.0, storm_duration_hours=1.5)

        self.assertIsInstance(res, FloodInundationResult)
        self.assertGreater(res.peak_inundation_depth_m, 0.0)
        self.assertGreater(res.ponded_water_volume_m3, 0.0)
        self.assertGreaterEqual(res.submerged_cells_count, 1)

        d = res.to_dict()
        self.assertIn("peak_depth_m", d)
        self.assertIn("ponded_volume_m3", d)
