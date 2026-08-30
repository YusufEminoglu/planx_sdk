# -*- coding: utf-8 -*-
"""Unit tests for PlanX Round 10 features (Betterment Levy & Biodiversity Corridors)."""

from __future__ import annotations

import unittest

from planx.economics.land_value_capture_betterment import (
    BettermentLevyResult,
    ZoningUpliftZone,
    calculate_betterment_levy_assessment,
)
from planx.environment.urban_biodiversity_corridor import (
    BiodiversityCorridorResult,
    EcologicalHabitatPatch,
    evaluate_biodiversity_corridor_connectivity,
)


class TestPlanXRound10(unittest.TestCase):
    def test_land_value_capture_betterment_levy(self) -> None:
        zones = [
            ZoningUpliftZone("Zone_A", parcel_area_m2=2500.0, baseline_land_value_per_m2=800.0, distance_to_new_transit_hub_m=200.0, rezoned_far_increase=2.0),
            ZoningUpliftZone("Zone_B", parcel_area_m2=4000.0, baseline_land_value_per_m2=600.0, distance_to_new_transit_hub_m=800.0, rezoned_far_increase=1.0),
        ]

        res = calculate_betterment_levy_assessment(zones, municipal_betterment_tax_rate_pct=35.0)

        self.assertIsInstance(res, BettermentLevyResult)
        self.assertEqual(res.total_parcels_assessed, 2)
        self.assertGreater(res.total_unearned_land_value_uplift_usd, 0.0)
        self.assertGreater(res.total_municipal_betterment_revenue_usd, 0.0)
        self.assertGreater(res.average_value_uplift_pct, 10.0)

        d = res.to_dict()
        self.assertIn("parcels_assessed", d)
        self.assertIn("municipal_revenue_usd", d)

    def test_urban_biodiversity_corridors(self) -> None:
        patches = [
            EcologicalHabitatPatch("Forest_Park", area_hectares=15.0, center_coordinates=(100.0, 100.0), habitat_quality_score=0.95),
            EcologicalHabitatPatch("River_Wetland", area_hectares=8.0, center_coordinates=(350.0, 200.0), habitat_quality_score=0.90),
            EcologicalHabitatPatch("Botanical_Garden", area_hectares=5.0, center_coordinates=(1500.0, 1500.0), habitat_quality_score=0.75),
        ]

        res = evaluate_biodiversity_corridor_connectivity(patches, max_species_dispersal_distance_m=800.0)

        self.assertIsInstance(res, BiodiversityCorridorResult)
        self.assertEqual(res.total_habitat_patches, 3)
        self.assertGreaterEqual(res.connected_network_components_count, 1)
        self.assertGreater(len(res.corridor_links), 0)

        d = res.to_dict()
        self.assertIn("patches_count", d)
        self.assertIn("connectivity_index", d)
