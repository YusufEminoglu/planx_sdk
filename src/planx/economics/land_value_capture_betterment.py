# -*- coding: utf-8 -*-
"""Municipal Land Value Capture & Infrastructure Betterment Levy Calculator for PlanX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class ZoningUpliftZone:
    zone_id: str
    parcel_area_m2: float
    baseline_land_value_per_m2: float
    distance_to_new_transit_hub_m: float
    rezoned_far_increase: float = 1.5  # e.g., FAR +1.5 uplift


@dataclass
class BettermentLevyResult:
    total_parcels_assessed: int
    total_unearned_land_value_uplift_usd: float
    total_municipal_betterment_revenue_usd: float
    average_value_uplift_pct: float
    zone_assessments: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result to a JSON-ready dictionary."""
        return {
            "parcels_assessed": self.total_parcels_assessed,
            "total_uplift_usd": round(self.total_unearned_land_value_uplift_usd, 2),
            "municipal_revenue_usd": round(self.total_municipal_betterment_revenue_usd, 2),
            "avg_uplift_pct": round(self.average_value_uplift_pct, 1),
        }


def calculate_betterment_levy_assessment(
    zones: Sequence[ZoningUpliftZone | dict[str, Any]],
    municipal_betterment_tax_rate_pct: float = 40.0,  # 40% value capture rate
    transit_proximity_elasticity: float = 0.25,  # 25% price bump per 500m proximity
) -> BettermentLevyResult:
    """Calculate municipal unearned increment land value capture and developer
    betterment levies following transit rezoning.
    """
    records: list[ZoningUpliftZone] = []
    for z in zones:
        if isinstance(z, ZoningUpliftZone):
            records.append(z)
        else:
            records.append(
                ZoningUpliftZone(
                    zone_id=str(z.get("zone_id", z.get("id", "Z1"))),
                    parcel_area_m2=float(z.get("parcel_area_m2", z.get("area_m2", 1000.0))),
                    baseline_land_value_per_m2=float(
                        z.get("baseline_land_value_per_m2", z.get("base_val", 500.0))
                    ),
                    distance_to_new_transit_hub_m=float(
                        z.get("distance_to_new_transit_hub_m", z.get("dist_m", 300.0))
                    ),
                    rezoned_far_increase=float(z.get("rezoned_far_increase", 1.5)),
                )
            )

    tot_uplift = 0.0
    tot_revenue = 0.0
    pcts: list[float] = []
    details: list[dict[str, Any]] = []

    for r in records:
        # Distance decay factor for transit uplift: max 1.0 at 0m, decaying out to 1500m
        dist_factor = max(0.05, 1.0 - (r.distance_to_new_transit_hub_m / 1500.0))
        # Value uplift formula: FAR increase * base value * transit elasticity * distance factor
        far_multiplier = 1.0 + 0.30 * r.rezoned_far_increase
        transit_multiplier = 1.0 + transit_proximity_elasticity * dist_factor

        post_val_m2 = r.baseline_land_value_per_m2 * far_multiplier * transit_multiplier
        delta_val_m2 = post_val_m2 - r.baseline_land_value_per_m2
        parcel_uplift = delta_val_m2 * r.parcel_area_m2

        levy_due = parcel_uplift * (municipal_betterment_tax_rate_pct / 100.0)
        uplift_pct = (delta_val_m2 / max(1.0, r.baseline_land_value_per_m2)) * 100.0

        tot_uplift += parcel_uplift
        tot_revenue += levy_due
        pcts.append(uplift_pct)

        details.append(
            {
                "zone_id": r.zone_id,
                "post_val_m2": round(post_val_m2, 2),
                "parcel_uplift_usd": round(parcel_uplift, 2),
                "levy_due_usd": round(levy_due, 2),
                "uplift_pct": round(uplift_pct, 1),
            }
        )

    avg_pct = sum(pcts) / max(1, len(pcts))

    return BettermentLevyResult(
        total_parcels_assessed=len(records),
        total_unearned_land_value_uplift_usd=tot_uplift,
        total_municipal_betterment_revenue_usd=tot_revenue,
        average_value_uplift_pct=avg_pct,
        zone_assessments=details,
    )
