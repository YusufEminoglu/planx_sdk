# -*- coding: utf-8 -*-
"""Urban Biodiversity Stepping-Stone & Ecological Habitat Connectivity Corridors for PlanX."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class EcologicalHabitatPatch:
    patch_id: str
    area_hectares: float
    center_coordinates: tuple[float, float]
    habitat_quality_score: float = 0.85  # 0.0 to 1.0 (canopy density, water proximity)


@dataclass
class BiodiversityCorridorResult:
    total_habitat_patches: int
    connected_network_components_count: int
    mean_connectivity_index: float
    critical_stepping_stone_gaps_count: int
    corridor_links: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "patches_count": self.total_habitat_patches,
            "components": self.connected_network_components_count,
            "connectivity_index": round(self.mean_connectivity_index, 3),
            "gaps_count": self.critical_stepping_stone_gaps_count,
            "links_count": len(self.corridor_links),
        }


def evaluate_biodiversity_corridor_connectivity(
    habitat_patches: Sequence[EcologicalHabitatPatch | dict[str, Any]],
    max_species_dispersal_distance_m: float = 600.0,  # e.g., urban birds / pollinator threshold
    urban_matrix_resistance_cost: float = 1.4,  # Asphalt / building barrier penalty
) -> BiodiversityCorridorResult:
    """Compute Least-Cost Path ecological connectivity and stepping-stone corridor permeability across fragmented urban landscapes."""
    patches: list[EcologicalHabitatPatch] = []
    for p in habitat_patches:
        if isinstance(p, EcologicalHabitatPatch):
            patches.append(p)
        else:
            coords = p.get("center", p.get("center_coordinates", (0.0, 0.0)))
            patches.append(
                EcologicalHabitatPatch(
                    patch_id=str(p.get("id", p.get("patch_id", f"P{len(patches)+1}"))),
                    area_hectares=float(p.get("area_hectares", p.get("area_ha", 5.0))),
                    center_coordinates=(float(coords[0]), float(coords[1])),
                    habitat_quality_score=float(p.get("habitat_quality_score", 0.85)),
                )
            )

    n = len(patches)
    if n == 0:
        return BiodiversityCorridorResult(0, 0, 0.0, 0, [])

    links: list[dict[str, Any]] = []
    connected_pairs = 0
    total_pairs = max(1, n * (n - 1) // 2)

    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = patches[i], patches[j]
            euclid_d = math.hypot(
                p1.center_coordinates[0] - p2.center_coordinates[0],
                p1.center_coordinates[1] - p2.center_coordinates[1],
            )
            effective_cost_dist = euclid_d * urban_matrix_resistance_cost

            if effective_cost_dist <= max_species_dispersal_distance_m:
                permeability = math.exp(-effective_cost_dist / max_species_dispersal_distance_m)
                combined_quality = (p1.habitat_quality_score + p2.habitat_quality_score) / 2.0
                corridor_flux = permeability * combined_quality * (p1.area_hectares * p2.area_hectares) ** 0.5

                connected_pairs += 1
                links.append({
                    "from_patch": p1.patch_id,
                    "to_patch": p2.patch_id,
                    "cost_distance_m": round(effective_cost_dist, 1),
                    "permeability": round(permeability, 3),
                    "corridor_flux": round(corridor_flux, 2),
                })

    conn_index = connected_pairs / float(total_pairs)
    gaps = total_pairs - connected_pairs

    return BiodiversityCorridorResult(
        total_habitat_patches=n,
        connected_network_components_count=max(1, n - connected_pairs),
        mean_connectivity_index=conn_index,
        critical_stepping_stone_gaps_count=gaps,
        corridor_links=links,
    )
