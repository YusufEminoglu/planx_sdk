# -*- coding: utf-8 -*-
"""Wildfire risk and Wildland-Urban Interface (WUI) exposure models."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _calculate_terrain_factors(dem: np.ndarray, cell_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Calculates slope and aspect in degrees using Horn's method."""
    padded = np.pad(dem, pad_width=1, mode="edge")

    # 3x3 window components
    z1 = padded[:-2, :-2]
    z2 = padded[:-2, 1:-1]
    z3 = padded[:-2, 2:]
    z4 = padded[1:-1, :-2]
    z6 = padded[1:-1, 2:]
    z7 = padded[2:, :-2]
    z8 = padded[2:, 1:-1]
    z9 = padded[2:, 2:]

    # Gradients
    dx = ((z3 + 2 * z6 + z9) - (z1 + 2 * z4 + z7)) / (8.0 * cell_size)
    dy = ((z7 + 2 * z8 + z9) - (z1 + 2 * z2 + z3)) / (8.0 * cell_size)

    slope_radians = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_radians)

    # Aspect calculation (clockwise from North [0, 360])
    aspect_rad = np.arctan2(dy, -dx)
    aspect_deg = (270.0 + np.degrees(aspect_rad)) % 360.0

    # Set aspect to NaN for flat cells (slope < 1.0 degree)
    aspect_deg[slope_deg < 1.0] = np.nan

    return slope_deg, aspect_deg


def wildfire_risk_index(
    dem: np.ndarray,
    cell_size: float,
    vegetation_density: np.ndarray,
    hemisphere: str = "northern",
    slope_weight: float = 0.35,
    aspect_weight: float = 0.20,
    veg_weight: float = 0.45,
) -> tuple[np.ndarray, list[list[str]]]:
    """Calculates wildfire risk index and risk classes across a grid.

    Combines terrain slope (steeper slope increases spread speed), aspect (exposure to
    solar drying based on hemisphere), and vegetation/fuel density into a combined
    exposure index [0, 100].

    Args:
        dem: 2D NumPy array containing elevation values. NaNs represent no-data.
        cell_size: Size of each grid cell in map units. Must be > 0.
        vegetation_density: 2D NumPy array of same shape as dem containing fuel /
            vegetation cover share [0.0, 1.0].
        hemisphere: 'northern' or 'southern'. Determines aspect risk scoring.
        slope_weight: Weight for the terrain slope factor.
        aspect_weight: Weight for the terrain aspect factor.
        veg_weight: Weight for the vegetation/fuel factor.

    Returns:
        Tuple of:
          - scores: 2D NumPy array containing wildfire risk scores [0, 100].
          - risk_classes: List of lists of strings matching dem.shape containing risk category
            ('Low', 'Moderate', 'High', 'Very High').
    """
    dem_arr = np.asarray(dem, dtype=np.float64)
    shape = dem_arr.shape
    veg_arr = np.asarray(vegetation_density, dtype=np.float64)

    if dem_arr.ndim != 2:
        raise ValueError("DEM must be a 2D array")
    if veg_arr.shape != shape:
        raise ValueError("vegetation_density shape must match dem shape")
    if cell_size <= 0:
        raise ValueError("cell_size must be greater than 0")

    valid = np.isfinite(dem_arr)
    if not np.any(valid):
        return np.zeros_like(dem_arr), [["Low" for _ in range(shape[1])] for _ in range(shape[0])]

    # Calculate terrain factors
    slope_deg, aspect_deg = _calculate_terrain_factors(dem_arr, cell_size)

    # 1. Slope Score: linearly scale [0, 30] degrees to [0, 100]
    slope_score = np.clip(slope_deg / 30.0 * 100.0, 0.0, 100.0)

    # 2. Aspect Score: solar exposure dryness factor
    if hemisphere.lower() == "northern":
        # South-facing (180 deg) is max risk (100), North-facing (0/360) is min risk (0)
        aspect_score = (1.0 - np.cos(np.radians(aspect_deg))) * 50.0
    else:
        # Southern hemisphere: North-facing is max risk (100), South is min (0)
        aspect_score = (1.0 + np.cos(np.radians(aspect_deg))) * 50.0

    # Flat areas with undefined aspect get a neutral/low aspect score
    aspect_score[np.isnan(aspect_score)] = 0.0

    # 3. Vegetation Score: scale [0.0, 1.0] to [0, 100]
    veg_score = np.clip(veg_arr * 100.0, 0.0, 100.0)

    # Weighted Linear Combination
    weight_sum = slope_weight + aspect_weight + veg_weight
    if weight_sum <= 0.0:
        weight_sum = 1.0

    scores = (
        slope_score * slope_weight + aspect_score * aspect_weight + veg_score * veg_weight
    ) / weight_sum

    # Preserve NaNs from the original DEM
    scores[~valid] = np.nan

    # Classify scores
    risk_classes = []
    for r in range(shape[0]):
        row_classes = []
        for c in range(shape[1]):
            val = scores[r, c]
            if not np.isfinite(val):
                row_classes.append("Low")
            elif val >= 75.0:
                row_classes.append("Very High")
            elif val >= 55.0:
                row_classes.append("High")
            elif val >= 35.0:
                row_classes.append("Moderate")
            else:
                row_classes.append("Low")
        row_classes = list(row_classes)
        risk_classes.append(row_classes)

    return scores, risk_classes


def wildfire_evacuation_encroachment(
    fire_origin: tuple[int, int],
    wind_vector: tuple[float, float],
    slope_grid: np.ndarray,
    fuel_grid: np.ndarray,
    cell_size: float = 30.0,
    time_steps: int = 10,
) -> dict:
    """Simulates Rothermel wildfire front expansion velocity and safe evacuation buffer.

    Args:
        fire_origin: Grid cell coordinate (row, col) of fire ignition.
        wind_vector: Wind velocity vector (u, v) in m/s.
        slope_grid: 2D array of terrain slope angles in degrees.
        fuel_grid: 2D array of fuel load density [0.0, 1.0].
        cell_size: Grid cell spatial resolution float (m).
        time_steps: Number of simulation propagation minutes.

    Returns:
        Dict containing simulation results:
          - burn_arrival_time: 2D NumPy array of min arrival time for each cell (inf for unburned).
          - flame_encroachment_mask: 2D boolean NumPy array of burned cells at time_steps.
          - safe_evacuation_buffer_m: 1D array of safe evacuation distance (m) per time step.
    """
    slope = np.asarray(slope_grid, dtype=np.float64)
    fuel = np.asarray(fuel_grid, dtype=np.float64)
    rows, cols = slope.shape

    if fuel.shape != (rows, cols):
        raise ValueError("fuel_grid shape must match slope_grid shape.")

    import heapq

    arrival_time = np.full((rows, cols), np.inf, dtype=np.float64)
    r_orig, c_orig = fire_origin

    if not (0 <= r_orig < rows and 0 <= c_orig < cols):
        raise ValueError("fire_origin is outside grid boundaries.")

    arrival_time[r_orig, c_orig] = 0.0

    pq = [(0.0, r_orig, c_orig)]

    wu, wv = wind_vector
    wind_speed = math.sqrt(wu**2 + wv**2)

    dr = [-1, -1, -1, 0, 0, 1, 1, 1]
    dc = [-1, 0, 1, -1, 1, -1, 0, 1]
    dists = [math.sqrt(2.0), 1.0, math.sqrt(2.0), 1.0, 1.0, math.sqrt(2.0), 1.0, math.sqrt(2.0)]

    while pq:
        t_curr, r, c = heapq.heappop(pq)
        if t_curr > arrival_time[r, c]:
            continue

        fuel_val = fuel[r, c]
        if fuel_val <= 0.0:
            continue

        slope_deg = slope[r, c]
        slope_factor = 0.02 * math.tan(math.radians(slope_deg))
        base_ros = 2.0 * (0.5 + fuel_val) * (1.0 + 0.05 * wind_speed + slope_factor)

        for i in range(8):
            nr, nc = r + dr[i], c + dc[i]
            if 0 <= nr < rows and 0 <= nc < cols:
                d_m = dists[i] * cell_size
                t_travel = d_m / max(0.1, base_ros)
                t_next = t_curr + t_travel / 60.0  # minutes

                if t_next < arrival_time[nr, nc]:
                    arrival_time[nr, nc] = t_next
                    heapq.heappush(pq, (t_next, nr, nc))

    mask = arrival_time <= float(time_steps)

    # Dynamic safe evacuation buffer (m)
    buffer_m = np.zeros(time_steps, dtype=np.float64)
    for step in range(1, time_steps + 1):
        burned_count = np.sum(arrival_time <= step)
        effective_radius = math.sqrt(burned_count * (cell_size**2) / math.pi)
        buffer_m[step - 1] = effective_radius + 100.0  # 100m safety margin

    return {
        "burn_arrival_time": arrival_time,
        "flame_encroachment_mask": mask,
        "safe_evacuation_buffer_m": buffer_m,
    }


def wildfire_evacuation_front_buffer(
    ignition_coords: np.ndarray,
    wind_speed_kmh: float,
    wind_direction_deg: float,
    terrain_slope_deg: np.ndarray,
    time_elapsed_hours: float,
    buffer_safety_factor: float = 1.5,
) -> dict[str, Any]:
    """Simulates dynamic wildfire front expansion and safety buffer zones for evacuation corridors.

    Args:
        ignition_coords: (2,) or (I, 2) ignition point coordinates.
        wind_speed_kmh: Wind velocity in km/h (>= 0).
        wind_direction_deg: Compass direction wind is blowing TO in degrees (0=North,
            90=East, 180=South, 270=West).
        terrain_slope_deg: Slope in degrees at ignition sites.
        time_elapsed_hours: Hours of fire growth (> 0).
        buffer_safety_factor: Multiplier for safety perimeter (default 1.5).

    Returns:
        Dict with keys:
        - forward_rate_of_spread_m_min: float
        - forward_distance_m: float
        - flank_distance_m: float
        - safety_buffer_distance_m: float
        - fire_ellipse_axes: dict with keys 'semi_major_m', 'semi_minor_m'
    """
    ig_arr = np.asarray(ignition_coords, dtype=np.float64)
    if ig_arr.ndim not in (1, 2) or ig_arr.shape[-1] != 2:
        raise ValueError("ignition_coords must be of shape (2,) or (I, 2).")
    slope_arr = np.asarray(terrain_slope_deg, dtype=np.float64)

    if wind_speed_kmh < 0:
        raise ValueError("wind_speed_kmh must be non-negative.")
    if time_elapsed_hours <= 0:
        raise ValueError("time_elapsed_hours must be greater than 0.")
    if not (0 <= wind_direction_deg < 360):
        raise ValueError("wind_direction_deg must be in [0, 360).")
    if buffer_safety_factor <= 0:
        raise ValueError("buffer_safety_factor must be positive.")

    # Convert slope to float (taking max slope if an array is passed)
    slope_val = float(np.max(slope_arr))

    # Base ROS
    r_0 = 0.5 + 0.1 * wind_speed_kmh
    # Slope multiplier
    phi_s = 1.0 + 0.05 * math.tan(math.radians(slope_val))
    # Forward ROS
    r_forward = r_0 * phi_s * (1.0 + 0.02 * wind_speed_kmh)

    r_flank = r_forward * 0.4
    r_back = r_forward * 0.1

    d_forward = r_forward * 60.0 * time_elapsed_hours
    d_flank = r_flank * 60.0 * time_elapsed_hours
    d_back = r_back * 60.0 * time_elapsed_hours
    d_buffer = d_forward * buffer_safety_factor

    return {
        "forward_rate_of_spread_m_min": float(r_forward),
        "forward_distance_m": float(d_forward),
        "flank_distance_m": float(d_flank),
        "safety_buffer_distance_m": float(d_buffer),
        "fire_ellipse_axes": {
            "semi_major_m": float((d_forward + d_back) / 2.0),
            "semi_minor_m": float(d_flank),
        },
    }
