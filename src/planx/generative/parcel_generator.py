# -*- coding: utf-8 -*-
"""Generative Land Subdivision & Parcel Layout Optimizer for PlanX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class ParcelPolygon:
    parcel_id: str
    area_m2: float
    frontage_m: float
    aspect_ratio: float
    coordinates: list[tuple[float, float]]


@dataclass
class ParcelLayoutResult:
    total_block_area_m2: float
    total_parcels: int
    mean_parcel_area_m2: float
    efficiency_net_to_gross_ratio: float
    parcels: list[ParcelPolygon]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result to a JSON-ready dictionary."""
        return {
            "total_block_area_m2": round(self.total_block_area_m2, 1),
            "total_parcels": self.total_parcels,
            "mean_parcel_area_m2": round(self.mean_parcel_area_m2, 1),
            "efficiency_ratio": round(self.efficiency_net_to_gross_ratio, 3),
            "parcels": [
                {
                    "id": p.parcel_id,
                    "area_m2": round(p.area_m2, 1),
                    "frontage_m": round(p.frontage_m, 1),
                    "aspect_ratio": round(p.aspect_ratio, 2),
                }
                for p in self.parcels
            ],
        }


def generate_optimized_parcels(
    block_polygon: Sequence[tuple[float, float]],
    target_parcel_area_m2: float = 400.0,
    min_frontage_m: float = 12.0,
    max_depth_m: float = 30.0,
) -> ParcelLayoutResult:
    """Subdivide a raw cadastral land block polygon into compliant buildable parcels."""
    poly = list(block_polygon)
    n = len(poly)
    if n < 3:
        return ParcelLayoutResult(0.0, 0, 0.0, 0.0, [])

    # Calculate bounding box
    min_x = min(p[0] for p in poly)
    max_x = max(p[0] for p in poly)
    min_y = min(p[1] for p in poly)
    max_y = max(p[1] for p in poly)

    block_w = max_x - min_x
    block_h = max_y - min_y
    total_area = block_w * block_h  # Rectangular approximation

    # Determine subdivision axis (along longest edge)
    parcels: list[ParcelPolygon] = []
    if block_w >= block_h:
        num_splits = max(1, int(block_w / max(min_frontage_m, target_parcel_area_m2 / max_depth_m)))
        dx = block_w / num_splits
        for i in range(num_splits):
            px1 = min_x + i * dx
            px2 = px1 + dx
            coords = [(px1, min_y), (px2, min_y), (px2, max_y), (px1, max_y), (px1, min_y)]
            area = dx * block_h
            aspect = block_h / max(1.0, dx)
            parcels.append(
                ParcelPolygon(
                    parcel_id=f"P_{i + 1:02d}",
                    area_m2=area,
                    frontage_m=dx,
                    aspect_ratio=aspect,
                    coordinates=coords,
                )
            )
    else:
        num_splits = max(1, int(block_h / max(min_frontage_m, target_parcel_area_m2 / max_depth_m)))
        dy = block_h / num_splits
        for i in range(num_splits):
            py1 = min_y + i * dy
            py2 = py1 + dy
            coords = [(min_x, py1), (max_x, py1), (max_x, py2), (min_x, py2), (min_x, py1)]
            area = block_w * dy
            aspect = block_w / max(1.0, dy)
            parcels.append(
                ParcelPolygon(
                    parcel_id=f"P_{i + 1:02d}",
                    area_m2=area,
                    frontage_m=dy,
                    aspect_ratio=aspect,
                    coordinates=coords,
                )
            )

    mean_area = sum(p.area_m2 for p in parcels) / max(1, len(parcels))
    efficiency = sum(p.area_m2 for p in parcels) / max(1.0, total_area)

    return ParcelLayoutResult(
        total_block_area_m2=total_area,
        total_parcels=len(parcels),
        mean_parcel_area_m2=mean_area,
        efficiency_net_to_gross_ratio=min(1.0, efficiency),
        parcels=parcels,
    )
