# -*- coding: utf-8 -*-
"""Unit tests for PlanX Round 6 features (EV Fleet Charging & Seismic PML Risk)."""

from __future__ import annotations

import unittest

from planx.mobility.ev_fleet_charging_optimizer import (
    ChargingDepotResult,
    EVBusProfile,
    optimize_ev_fleet_charging,
)
from planx.resilience.seismic_vulnerability_pml import (
    BuildingStructureAsset,
    SeismicPMLReport,
    calculate_seismic_pml_risk,
)


class TestPlanXRound6(unittest.TestCase):
    def test_ev_fleet_charging_optimizer(self) -> None:
        fleet = [
            EVBusProfile(bus_id="Bus_101", battery_capacity_kwh=300.0, arrival_soc_pct=20.0, target_departure_soc_pct=85.0),
            EVBusProfile(bus_id="Bus_102", battery_capacity_kwh=300.0, arrival_soc_pct=30.0, target_departure_soc_pct=90.0),
            EVBusProfile(bus_id="Bus_103", battery_capacity_kwh=350.0, arrival_soc_pct=15.0, target_departure_soc_pct=80.0),
        ]

        res = optimize_ev_fleet_charging(fleet, max_grid_transformer_capacity_kw=500.0)

        self.assertIsInstance(res, ChargingDepotResult)
        self.assertGreater(res.total_energy_delivered_kwh, 0.0)
        self.assertLessEqual(res.peak_grid_load_kw, 500.0)
        self.assertEqual(len(res.hourly_depot_load_profile_kw), 24)

        d = res.to_dict()
        self.assertIn("energy_delivered_kwh", d)
        self.assertIn("peak_grid_load_kw", d)

    def test_seismic_pml_risk_assessment(self) -> None:
        portfolio = [
            BuildingStructureAsset(
                building_id="School_A",
                structural_type="RC_FRAME",
                construction_year=1992,
                number_of_stories=4,
                replacement_value_usd=2000000.0,
                ground_pga_g=0.45,
                soil_class="ZC",
            ),
            BuildingStructureAsset(
                building_id="Hospital_B",
                structural_type="RC_SHEAR_WALL",
                construction_year=2019,
                number_of_stories=8,
                replacement_value_usd=15000000.0,
                ground_pga_g=0.45,
                soil_class="ZB",
            ),
        ]

        report = calculate_seismic_pml_risk(portfolio)

        self.assertIsInstance(report, SeismicPMLReport)
        self.assertEqual(report.total_portfolio_value_usd, 17000000.0)
        self.assertGreater(report.probable_maximum_loss_usd, 0.0)
        self.assertGreater(report.portfolio_pml_ratio_pct, 0.0)
        self.assertEqual(len(report.building_vulnerability_scores), 2)

        d = report.to_dict()
        self.assertIn("pml_loss_usd", d)
        self.assertIn("portfolio_value_usd", d)
