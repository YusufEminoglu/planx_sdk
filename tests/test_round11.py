# -*- coding: utf-8 -*-
"""Unit tests for PlanX Round 11 features (EV Charging Queue & HAZUS Seismic Vulnerability)."""

from __future__ import annotations

import unittest

from planx.mobility import (
    EVChargingQueueResult,
    EVStationQueueConfig,
    simulate_ev_charging_station_queues,
)
from planx.resilience import (
    BuildingFragilityCurve,
    SeismicDamageResult,
    evaluate_seismic_building_damage_hazus,
)


class TestPlanXRound11(unittest.TestCase):
    def test_ev_charging_queue_markov(self) -> None:
        cfg = EVStationQueueConfig(
            num_fast_chargers_c=4,
            max_queue_capacity_k=10,
            mean_arrival_rate_evs_per_hour_lambda=10.0,
            mean_charging_time_minutes_mu=20.0,
        )

        res = simulate_ev_charging_station_queues(cfg, charger_rated_power_kw=150.0)

        self.assertIsInstance(res, EVChargingQueueResult)
        self.assertEqual(res.num_chargers, 4)
        self.assertGreater(res.charger_utilization_factor_rho, 0.0)
        self.assertGreaterEqual(res.average_waiting_time_minutes_wq, 0.0)
        self.assertGreater(res.grid_peak_power_demand_kw, 0.0)

        d = res.to_dict()
        self.assertIn("chargers", d)
        self.assertIn("utilization_rho", d)
        self.assertIn("waiting_time_min", d)

    def test_seismic_building_damage_hazus(self) -> None:
        curve = BuildingFragilityCurve(building_class="C2M", median_spectral_accel_moderate_g=0.40)
        res = evaluate_seismic_building_damage_hazus(peak_ground_spectral_accel_sa_g=0.60, building_occupants_count=300, fragility=curve)

        self.assertIsInstance(res, SeismicDamageResult)
        self.assertGreaterEqual(res.peak_spectral_acceleration_sa_g, 0.0)
        self.assertGreater(res.prob_moderate_damage + res.prob_extensive_damage + res.prob_complete_collapse, 0.0)
        self.assertIn(res.building_damage_state_modal, ["NONE", "SLIGHT", "MODERATE", "EXTENSIVE", "COMPLETE"])
        self.assertGreaterEqual(res.post_earthquake_shelter_demand_persons, 0)

        d = res.to_dict()
        self.assertIn("sa_g", d)
        self.assertIn("modal_damage", d)
        self.assertIn("shelter_demand_persons", d)
