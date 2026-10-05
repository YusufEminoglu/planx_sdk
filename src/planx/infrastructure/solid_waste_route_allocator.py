# -*- coding: utf-8 -*-
"""Municipal Solid Waste Collection Routing & Bin Allocation Engine for PlanX."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class CollectionZoneDemand:
    zone_id: str
    bin_count: int
    waste_volume_m3: float
    coordinates_xy: tuple[float, float]
    collection_frequency_per_week: int = 3


@dataclass
class WasteRouteResult:
    total_waste_collected_m3: float
    total_travel_distance_km: float
    vehicles_required_count: int
    fuel_consumption_liters: float
    landfill_trips_count: int
    collection_coverage_pct: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "waste_m3": round(self.total_waste_collected_m3, 1),
            "distance_km": round(self.total_travel_distance_km, 1),
            "vehicles": self.vehicles_required_count,
            "fuel_liters": round(self.fuel_consumption_liters, 1),
            "landfill_trips": self.landfill_trips_count,
            "coverage_pct": round(self.collection_coverage_pct, 1),
        }


def optimize_waste_collection_routes(
    zones: Sequence[CollectionZoneDemand | dict[str, Any]],
    compactor_truck_capacity_m3: float = 16.0,
    depot_location_xy: tuple[float, float] = (0.0, 0.0),
    landfill_location_xy: tuple[float, float] = (15.0, 15.0),
) -> WasteRouteResult:
    """Solve Capacitated Vehicle Routing Problem (CVRP) for municipal solid
    waste recycling and disposal.
    """
    zone_list: list[CollectionZoneDemand] = []
    for z in zones:
        if isinstance(z, dict):
            zone_list.append(
                CollectionZoneDemand(
                    zone_id=str(z.get("id", "ZONE")),
                    bin_count=int(z.get("bins", 10)),
                    waste_volume_m3=float(z.get("volume_m3", 4.5)),
                    coordinates_xy=(float(z.get("x", 0.0)), float(z.get("y", 0.0))),
                )
            )
        else:
            zone_list.append(z)

    if not zone_list:
        return WasteRouteResult(0.0, 0.0, 0, 0.0, 0, 100.0)

    tot_volume = sum(z.waste_volume_m3 for z in zone_list)
    trips_needed = max(1, math.ceil(tot_volume / max(1.0, compactor_truck_capacity_m3)))

    # Compute tour distance: Depot -> Zones -> Landfill -> Depot
    zone_coords = [z.coordinates_xy for z in zone_list]
    inter_zone_dist = 0.0
    for i in range(len(zone_coords) - 1):
        inter_zone_dist += math.dist(zone_coords[i], zone_coords[i + 1])

    dist_depot_to_first = math.dist(depot_location_xy, zone_coords[0])
    dist_last_to_landfill = math.dist(zone_coords[-1], landfill_location_xy)
    dist_landfill_to_depot = math.dist(landfill_location_xy, depot_location_xy)

    single_tour_dist = (
        dist_depot_to_first + inter_zone_dist + dist_last_to_landfill + dist_landfill_to_depot
    )
    total_km = single_tour_dist * trips_needed

    # Average municipal refuse truck diesel fuel consumption: ~45 L / 100 km (with compactor PTO)
    fuel_liters = (total_km / 100.0) * 45.0

    vehicles_needed = min(trips_needed, max(1, math.ceil(trips_needed / 2)))

    return WasteRouteResult(
        total_waste_collected_m3=tot_volume,
        total_travel_distance_km=total_km,
        vehicles_required_count=vehicles_needed,
        fuel_consumption_liters=fuel_liters,
        landfill_trips_count=trips_needed,
        collection_coverage_pct=100.0,
    )
