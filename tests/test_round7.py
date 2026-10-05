# -*- coding: utf-8 -*-
"""Unit tests for PlanX Round 7 features (Water Distribution Network & TIF Cashflows)."""

from __future__ import annotations

import unittest

from planx.economics.tax_increment_financing_tif import (
    PropertyTaxAssessment,
    TIFDistrictResult,
    simulate_tif_district_cashflows,
)
from planx.infrastructure.water_distribution_epanet import (
    PipeHydraulicSegment,
    WaterNetworkResult,
    simulate_water_pipe_network,
)


class TestPlanXRound7(unittest.TestCase):
    def test_water_distribution_network(self) -> None:
        pipes = [
            PipeHydraulicSegment(
                pipe_id="Main_P1",
                from_node="Tank",
                to_node="Junction1",
                length_m=200.0,
                diameter_mm=200.0,
                flow_rate_lps=40.0,
            ),
            PipeHydraulicSegment(
                pipe_id="Branch_P2",
                from_node="Junction1",
                to_node="NodeA",
                length_m=150.0,
                diameter_mm=150.0,
                flow_rate_lps=20.0,
            ),
        ]

        res = simulate_water_pipe_network(pipes, source_pressure_head_m=50.0)

        self.assertIsInstance(res, WaterNetworkResult)
        self.assertEqual(res.total_network_length_m, 350.0)
        self.assertGreater(res.max_flow_velocity_ms, 0.0)
        self.assertGreater(res.total_head_loss_m, 0.0)
        self.assertEqual(len(res.pipe_diagnostics), 2)

        d = res.to_dict()
        self.assertIn("network_length_m", d)
        self.assertIn("max_velocity_ms", d)

    def test_tif_district_cashflows(self) -> None:
        props = [
            PropertyTaxAssessment(
                property_id="Block_1",
                base_assessed_value_usd=2000000.0,
                current_assessed_value_usd=2000000.0,
                annual_appreciation_rate_pct=5.0,
            ),
            PropertyTaxAssessment(
                property_id="Block_2",
                base_assessed_value_usd=3000000.0,
                current_assessed_value_usd=3000000.0,
                annual_appreciation_rate_pct=4.0,
            ),
        ]

        tif_res = simulate_tif_district_cashflows(
            props, lifespan_years=15, annual_bond_debt_service_usd=80000.0
        )

        self.assertIsInstance(tif_res, TIFDistrictResult)
        self.assertGreater(tif_res.total_base_tax_revenue_usd, 0.0)
        self.assertGreater(tif_res.total_incremental_tax_captured_usd, 0.0)
        self.assertGreater(tif_res.net_present_value_usd, 0.0)
        self.assertGreater(tif_res.bond_debt_service_coverage_ratio, 0.0)
        self.assertEqual(len(tif_res.annual_cashflows), 15)

        d = tif_res.to_dict()
        self.assertIn("base_revenue_usd", d)
        self.assertIn("incremental_captured_usd", d)
        self.assertIn("dscr", d)
