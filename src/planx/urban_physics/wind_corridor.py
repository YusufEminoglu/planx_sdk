# -*- coding: utf-8 -*-
"""Urban Wind Ventilation Corridor & Roughness Aerodynamics Simulator for PlanX."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class VentilationGridCell:
    """Aerodynamic metrics for a single spatial cell in the wind domain."""

    grid_x: int
    grid_y: int
    center_x: float
    center_y: float
    building_density_lambda_p: float  # Plan area index
    frontal_area_index_lambda_f: float  # Frontal area index
    roughness_length_z0: float  # Aerodynamic roughness (meters)
    displacement_height_zd: float  # Zero-plane displacement height (meters)
    wind_velocity_ratio: float  # U_ped / U_ref (0.0 to 1.0)
    is_ventilation_corridor: bool  # True if high flow path
    is_stagnation_zone: bool  # True if blocked wind trap


@dataclass
class WindCorridorResult:
    """Master simulation report of urban wind corridors."""

    wind_direction_degrees: float
    reference_wind_speed_ms: float
    grid_resolution_m: float
    domain_rows: int
    domain_cols: int
    mean_velocity_ratio: float
    corridor_cells_count: int
    stagnation_cells_count: int
    cells: list[VentilationGridCell]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result to a JSON-ready dictionary."""
        return {
            "wind_direction_degrees": self.wind_direction_degrees,
            "reference_wind_speed_ms": self.reference_wind_speed_ms,
            "grid_resolution_m": self.grid_resolution_m,
            "domain_dimensions": f"{self.domain_cols}x{self.domain_rows}",
            "mean_velocity_ratio": round(self.mean_velocity_ratio, 3),
            "corridor_cells_count": self.corridor_cells_count,
            "stagnation_cells_count": self.stagnation_cells_count,
        }


def _polygon_area(poly: Sequence[tuple[float, float]]) -> float:
    """Shoelace formula for 2D polygon area."""
    n = len(poly)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1]
    return abs(area) / 2.0


def _polygon_projected_width(poly: Sequence[tuple[float, float]], wind_angle_rad: float) -> float:
    """Project building polygon orthogonal to wind direction."""
    if not poly:
        return 0.0
    # Normal vector to wind: angle + pi/2
    nx = math.cos(wind_angle_rad + math.pi / 2.0)
    ny = math.sin(wind_angle_rad + math.pi / 2.0)
    projections = [p[0] * nx + p[1] * ny for p in poly]
    return max(projections) - min(projections)


def simulate_wind_corridors(
    buildings: list[dict[str, Any]],
    domain_bbox: tuple[float, float, float, float],
    wind_direction_degrees: float = 45.0,
    reference_wind_speed_ms: float = 5.0,
    grid_resolution: float = 20.0,
) -> WindCorridorResult:
    """Simulate urban wind ventilation corridors and aerodynamic roughness.

    Uses Grimmond & Oke / Macdonald morphometric aerodynamic models:
    - Zero-plane displacement height: z_d = H * (1 + alpha^(-lambda_p) * (lambda_p - 1))
    - Roughness length: z_0 = H * (1 - z_d / H)
      * exp(-(0.5 * beta * C_d / kappa^2 * (1 - z_d / H) * lambda_f)^(-0.5))
    - Velocity ratio U / U_ref based on log wind profile at pedestrian height (1.5m).

    Args:
        buildings: List of dicts with 'footprint' (list of (x,y)), 'height' (meters).
        domain_bbox: (min_x, min_y, max_x, max_y) domain bounds in meters.
        wind_direction_degrees: Angle from which wind blows (0 = North, 90 = East).
        reference_wind_speed_ms: Freestream wind speed at 10m height (m/s).
        grid_resolution: Grid cell size in meters (default: 20m).

    Returns:
        WindCorridorResult containing cell-by-cell wind velocity ratios and
        corridor classifications.
    """
    min_x, min_y, max_x, max_y = domain_bbox
    cols = max(1, int((max_x - min_x) / grid_resolution))
    rows = max(1, int((max_y - min_y) / grid_resolution))
    cell_area = grid_resolution * grid_resolution

    wind_rad = math.radians(wind_direction_degrees)
    kappa = 0.40  # von Karman constant
    c_d = 1.20  # Building drag coefficient
    alpha_const = 4.43
    beta_const = 1.0

    cells_grid: list[VentilationGridCell] = []
    vel_ratios: list[float] = []
    corridors_count = 0
    stagnation_count = 0

    for r in range(rows):
        for c in range(cols):
            cx = min_x + (c + 0.5) * grid_resolution
            cy = min_y + (r + 0.5) * grid_resolution
            c_min_x = min_x + c * grid_resolution
            c_max_x = c_min_x + grid_resolution
            c_min_y = min_y + r * grid_resolution
            c_max_y = c_min_y + grid_resolution

            # Find buildings overlapping this grid cell
            cell_bld_area = 0.0
            cell_frontal_area = 0.0
            heights = []

            for b in buildings:
                fp = b.get("footprint", [])
                h = float(b.get("height", 10.0))
                if not fp:
                    continue
                # Centroid check for cell membership
                bx = sum(p[0] for p in fp) / len(fp)
                by = sum(p[1] for p in fp) / len(fp)

                if c_min_x <= bx < c_max_x and c_min_y <= by < c_max_y:
                    b_area = _polygon_area(fp)
                    proj_w = _polygon_projected_width(fp, wind_rad)
                    cell_bld_area += b_area
                    cell_frontal_area += proj_w * h
                    heights.append(h)

            lambda_p = min(0.95, cell_bld_area / cell_area)
            lambda_f = min(1.50, cell_frontal_area / cell_area)
            mean_h = float(np.mean(heights)) if heights else 0.0

            if mean_h > 0 and lambda_p > 0.01:
                # Macdonald aerodynamic formula
                zd = mean_h * (1.0 + (alpha_const ** (-lambda_p)) * (lambda_p - 1.0))
                zd = max(0.0, min(mean_h * 0.90, zd))

                term = (0.5 * beta_const * c_d / (kappa**2)) * (1.0 - zd / mean_h) * lambda_f
                if term > 0:
                    z0 = mean_h * (1.0 - zd / mean_h) * math.exp(-(term ** (-0.5)))
                else:
                    z0 = 0.05
            else:
                zd = 0.0
                z0 = 0.03  # Open ground / grass roughness

            # Log-law pedestrian wind speed ratio at z = 1.5m
            # U(z) = U_ref * ln((z - zd)/z0) / ln((10 - zd)/z0)
            z_ped = 1.5
            if z_ped > zd and z_ped > z0:
                num = math.log((z_ped - zd) / max(1e-4, z0))
                den = math.log(max(1.1, (10.0 - zd) / max(1e-4, z0)))
                vel_ratio = max(0.05, min(1.0, num / den))
            else:
                # Inside building canopy: reduced exponentially by canopy drag
                vel_ratio = max(0.02, math.exp(-2.5 * lambda_p))

            # Classification
            is_corridor = (vel_ratio >= 0.65) and (lambda_p < 0.20)
            is_stagnation = (vel_ratio <= 0.20) and (lambda_p > 0.40)

            if is_corridor:
                corridors_count += 1
            if is_stagnation:
                stagnation_count += 1

            vel_ratios.append(vel_ratio)

            cells_grid.append(
                VentilationGridCell(
                    grid_x=c,
                    grid_y=r,
                    center_x=cx,
                    center_y=cy,
                    building_density_lambda_p=round(lambda_p, 3),
                    frontal_area_index_lambda_f=round(lambda_f, 3),
                    roughness_length_z0=round(z0, 3),
                    displacement_height_zd=round(zd, 3),
                    wind_velocity_ratio=round(vel_ratio, 3),
                    is_ventilation_corridor=is_corridor,
                    is_stagnation_zone=is_stagnation,
                )
            )

    mean_vr = float(np.mean(vel_ratios)) if vel_ratios else 0.50

    return WindCorridorResult(
        wind_direction_degrees=wind_direction_degrees,
        reference_wind_speed_ms=reference_wind_speed_ms,
        grid_resolution_m=grid_resolution,
        domain_rows=rows,
        domain_cols=cols,
        mean_velocity_ratio=mean_vr,
        corridor_cells_count=corridors_count,
        stagnation_cells_count=stagnation_count,
        cells=cells_grid,
    )
