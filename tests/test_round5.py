# -*- coding: utf-8 -*-
"""Unit tests for PlanX Round 5 features (Solar Microgrid & Silver Aging Mobility)."""

from __future__ import annotations

import unittest

from planx.demographics.aging_mobility_index import (
    AgingMobilityResult,
    SeniorZoneProfile,
    calculate_silver_mobility_score,
)
from planx.energy.solar_microgrid import (
    BatteryStorageProfile,
    SolarMicrogridResult,
    simulate_community_solar_microgrid,
)


class TestPlanXRound5(unittest.TestCase):
    def test_solar_microgrid_simulation(self) -> None:
        res = simulate_community_solar_microgrid(
            rooftop_area_m2=4000.0,
            solar_efficiency=0.20,
            daily_insolation_kwh_m2=5.0,
            total_daily_demand_kwh=3000.0,
            battery_profile=BatteryStorageProfile(capacity_kwh=300.0, initial_charge_kwh=150.0),
        )

        self.assertIsInstance(res, SolarMicrogridResult)
        self.assertGreater(res.total_pv_generation_kwh, 0.0)
        self.assertGreater(res.self_sufficiency_ratio_pct, 0.0)
        self.assertGreater(res.solar_self_consumption_pct, 0.0)
        self.assertEqual(len(res.hourly_state_of_charge), 24)

        d = res.to_dict()
        self.assertIn("self_sufficiency_ratio_pct", d)
        self.assertIn("pv_generation_kwh", d)

    def test_silver_mobility_score(self) -> None:
        zones = [
            SeniorZoneProfile(
                zone_id="Z_Central",
                senior_population_65plus=250,
                mean_sidewalk_width_m=2.4,
                curb_ramp_compliance_pct=95.0,
                resting_bench_density_per_km=10.0,
                max_walking_slope_pct=2.5,
                pedestrian_crossing_time_seconds=22.0,
                shade_coverage_pct=35.0,
            ),
            SeniorZoneProfile(
                zone_id="Z_Hilly",
                senior_population_65plus=120,
                mean_sidewalk_width_m=1.2,
                curb_ramp_compliance_pct=40.0,
                resting_bench_density_per_km=2.0,
                max_walking_slope_pct=9.0,
                pedestrian_crossing_time_seconds=14.0,
                shade_coverage_pct=15.0,
            ),
        ]

        result = calculate_silver_mobility_score(zones)
        self.assertIsInstance(result, AgingMobilityResult)
        self.assertEqual(result.zone_grades["Z_Central"], "A")
        self.assertIn(result.zone_grades["Z_Hilly"], ("D", "F"))
        self.assertEqual(result.steep_slopes_hazard_count, 1)

        d = result.to_dict()
        self.assertIn("overall_silver_index", d)
        self.assertIn("zone_grades", d)
