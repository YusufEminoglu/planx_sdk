# -*- coding: utf-8 -*-
"""Unit tests for PlanX Round 8 features (District Heating/Cooling & Transit Signal Priority)."""

from __future__ import annotations

import unittest

from planx.infrastructure.district_cooling_heating import (
    BuildingThermalDemandProfile,
    DHCNetworkResult,
    simulate_dhc_thermal_network,
)
from planx.mobility.transit_signal_priority_tsp import (
    SignalizedJunction,
    TSPPerformanceResult,
    simulate_transit_signal_priority,
)


class TestPlanXRound8(unittest.TestCase):
    def test_district_cooling_heating(self) -> None:
        buildings = [
            BuildingThermalDemandProfile(
                "Hospital",
                peak_heating_demand_kw=350.0,
                peak_cooling_demand_kw=250.0,
                pipe_distance_from_plant_m=300.0,
            ),
            BuildingThermalDemandProfile(
                "University",
                peak_heating_demand_kw=500.0,
                peak_cooling_demand_kw=400.0,
                pipe_distance_from_plant_m=600.0,
            ),
        ]

        res = simulate_dhc_thermal_network(
            buildings, supply_water_temp_c=65.0, return_water_temp_c=35.0
        )

        self.assertIsInstance(res, DHCNetworkResult)
        self.assertGreater(res.total_thermal_delivered_mwh_yr, 0.0)
        self.assertGreater(res.annual_co2_avoided_tonnes, 0.0)
        self.assertGreater(res.pumping_electrical_energy_kwh_yr, 0.0)
        self.assertGreater(res.overall_system_cop, 3.0)

        d = res.to_dict()
        self.assertIn("thermal_delivered_mwh_yr", d)
        self.assertIn("co2_avoided_tonnes", d)

    def test_transit_signal_priority(self) -> None:
        juncs = [
            SignalizedJunction(
                "J1",
                distance_along_corridor_m=400.0,
                cycle_time_seconds=90.0,
                green_split_seconds=35.0,
            ),
            SignalizedJunction(
                "J2",
                distance_along_corridor_m=900.0,
                cycle_time_seconds=90.0,
                green_split_seconds=35.0,
            ),
            SignalizedJunction(
                "J3",
                distance_along_corridor_m=1500.0,
                cycle_time_seconds=90.0,
                green_split_seconds=35.0,
            ),
        ]

        tsp_res = simulate_transit_signal_priority(
            juncs, bus_cruising_speed_kmh=45.0, dwell_time_per_stop_seconds=15.0
        )

        self.assertIsInstance(tsp_res, TSPPerformanceResult)
        self.assertEqual(tsp_res.total_corridor_length_m, 1500.0)
        self.assertLessEqual(
            tsp_res.tsp_optimized_travel_time_seconds, tsp_res.baseline_travel_time_seconds
        )
        self.assertGreaterEqual(tsp_res.bus_delay_reduction_pct, 0.0)
        self.assertGreater(tsp_res.corridor_average_speed_kmh, 0.0)

        d = tsp_res.to_dict()
        self.assertIn("baseline_time_s", d)
        self.assertIn("tsp_time_s", d)
        self.assertIn("delay_reduction_pct", d)
