# -*- coding: utf-8 -*-
"""Continuous 3D Sky View Factor (SVF) & Daylight Autonomy Field Analyzer for PlanX."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass
class DaylightAutonomyResult:
    mean_sky_view_factor: float
    min_sky_view_factor: float
    max_sky_view_factor: float
    daylight_autonomy_percent: float  # Percentage of space exceeding 300 lux threshold
    grid_svf_matrix: list[list[float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_svf": round(self.mean_sky_view_factor, 3),
            "min_svf": round(self.min_sky_view_factor, 3),
            "max_svf": round(self.max_sky_view_factor, 3),
            "daylight_autonomy_percent": round(self.daylight_autonomy_percent, 2),
            "grid_rows": len(self.grid_svf_matrix),
            "grid_cols": len(self.grid_svf_matrix[0]) if self.grid_svf_matrix else 0,
        }


def compute_3d_sky_view_factor(
    buildings: list[dict[str, Any]],
    domain_bbox: tuple[float, float, float, float],
    grid_resolution: float = 25.0,
    num_ray_azimuths: int = 16,
) -> DaylightAutonomyResult:
    r"""Calculate continuous 3D Sky View Factor (SVF = (1/2pi) \int \cos^2(\beta(\phi)) d\phi) across urban canyons."""
    min_x, min_y, max_x, max_y = domain_bbox
    cols = max(1, int((max_x - min_x) / grid_resolution))
    rows = max(1, int((max_y - min_y) / grid_resolution))

    svf_grid: list[list[float]] = []
    all_svf: list[float] = []

    azimuths = [i * (2.0 * math.pi / num_ray_azimuths) for i in range(num_ray_azimuths)]

    for r in range(rows):
        row_vals = []
        cy = min_y + (r + 0.5) * grid_resolution
        for c in range(cols):
            cx = min_x + (c + 0.5) * grid_resolution

            # Cast rays in all azimuth directions to find maximum obstacle elevation angle beta
            ray_svf_sum = 0.0
            for az in azimuths:
                dx = math.cos(az)
                dy = math.sin(az)
                max_tan_beta = 0.0

                for b in buildings:
                    bx = b.get("x", 0.0)
                    by = b.get("y", 0.0)
                    bh = b.get("height", 15.0)

                    dist = math.hypot(bx - cx, by - cy)
                    if dist > 0.1 and dist < 200.0:
                        # Angle to building
                        b_dir = math.atan2(by - cy, bx - cx)
                        diff = abs(b_dir - az)
                        while diff > math.pi:
                            diff -= 2 * math.pi
                        if abs(diff) < math.pi / 4.0:
                            tan_beta = bh / dist
                            max_tan_beta = max(max_tan_beta, tan_beta)

                beta = math.atan(max_tan_beta)
                # Unobstructed sky fraction along this ray
                ray_svf_sum += math.cos(beta) ** 2

            svf_val = ray_svf_sum / num_ray_azimuths
            svf_val = max(0.05, min(1.0, svf_val))
            row_vals.append(round(svf_val, 3))
            all_svf.append(svf_val)

        svf_grid.append(row_vals)

    mean_s = float(np.mean(all_svf)) if all_svf else 0.5
    min_s = float(np.min(all_svf)) if all_svf else 0.0
    max_s = float(np.max(all_svf)) if all_svf else 1.0

    # Daylight Autonomy: fraction of cells where SVF >= 0.40
    da_pct = (sum(1 for s in all_svf if s >= 0.40) / max(1, len(all_svf))) * 100.0

    return DaylightAutonomyResult(
        mean_sky_view_factor=mean_s,
        min_sky_view_factor=min_s,
        max_sky_view_factor=max_s,
        daylight_autonomy_percent=da_pct,
        grid_svf_matrix=svf_grid,
    )
