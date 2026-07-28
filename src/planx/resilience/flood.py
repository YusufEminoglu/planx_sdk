# -*- coding: utf-8 -*-
"""Flood risk and pluvial vulnerability models."""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np
import scipy.ndimage


def pluvial_flood_susceptibility(
    dem: np.ndarray,
    cell_size: float,
    neighborhood_radius: float = 150.0,
    drainage_dists: Optional[np.ndarray] = None,
    elevation_weight: float = 0.45,
    slope_weight: float = 0.30,
    drainage_weight: float = 0.25,
) -> tuple[np.ndarray, list[list[str]]]:
    """Calculates pluvial flood susceptibility scores across a DEM grid.

    This is a screening model that identifies low-lying, flat, and drainage-proximate
    areas using NumPy and SciPy image processing filters.

    Args:
        dem: 2D NumPy array containing elevation values. NaNs represent no-data.
        cell_size: Size of each grid cell in map units.
        neighborhood_radius: Radius in map units to calculate local relief.
        drainage_dists: Optional 2D NumPy array of the same shape as dem containing
            Euclidean distance to the closest drainage network.
        elevation_weight: Weight for relative low-elevation (0-1).
        slope_weight: Weight for flat / low-slope areas (0-1).
        drainage_weight: Weight for drainage proximity (0-1).

    Returns:
        Tuple of:
          - scores: 2D NumPy array containing the flood susceptibility score [0, 100].
          - risk_classes: List of lists of strings matching dem.shape containing risk category
            ('Low', 'Moderate', 'High', 'Very High').
    """
    dem_arr = np.asarray(dem, dtype=np.float64)
    shape = dem_arr.shape

    if dem_arr.ndim != 2:
        raise ValueError("DEM must be a 2D array")

    # Mask of valid pixels
    valid = np.isfinite(dem_arr)
    if not np.any(valid):
        return np.zeros_like(dem_arr), [["Low" for _ in range(shape[1])] for _ in range(shape[0])]

    # 1. Relative low elevation
    min_elev = np.min(dem_arr[valid])
    max_elev = np.max(dem_arr[valid])
    elev_range = float(max_elev - min_elev)
    if elev_range <= 0:
        elev_range = 1.0

    rel_low = np.clip((1.0 - (dem_arr - min_elev) / elev_range) * 100.0, 0.0, 100.0)

    # 2. Slope proxy (local relief max - min in neighborhood)
    radius_pixels = max(1, int(round(neighborhood_radius / cell_size)))
    filter_size = 2 * radius_pixels + 1

    # To handle NaNs correctly in filters, we can temporarily fill them with inf/neg-inf
    dem_temp_max = dem_arr.copy()
    dem_temp_max[~valid] = -np.inf
    dem_temp_min = dem_arr.copy()
    dem_temp_min[~valid] = np.inf

    max_dem = scipy.ndimage.maximum_filter(dem_temp_max, size=filter_size)
    min_dem = scipy.ndimage.minimum_filter(dem_temp_min, size=filter_size)

    # Calculate relief
    relief = max_dem - min_dem
    relief[~np.isfinite(relief)] = 0.0

    # Flatness/slope proxy: 100 - (relief / radius) * 100
    slope_proxy = np.clip(100.0 - (relief / neighborhood_radius) * 100.0, 0.0, 100.0)

    # 3. Proximity to drainage
    if drainage_dists is not None:
        d_dists = np.asarray(drainage_dists, dtype=np.float64)
        if d_dists.shape != shape:
            raise ValueError("drainage_dists shape must match dem shape")
        drain_score = np.clip(100.0 - (d_dists / (4.0 * neighborhood_radius)) * 100.0, 0.0, 100.0)
    else:
        drain_score = np.full(shape, 50.0, dtype=np.float64)
        drainage_weight = 0.0  # Exclude from weighting if not provided

    # 4. Weighted linear combination
    weight_sum = elevation_weight + slope_weight + drainage_weight
    if weight_sum <= 0:
        weight_sum = 1.0

    scores = (
        rel_low * elevation_weight + slope_proxy * slope_weight + drain_score * drainage_weight
    ) / weight_sum

    scores = np.clip(scores, 0.0, 100.0)
    scores[~valid] = np.nan

    # 5. Risk classification
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
        risk_classes.append(row_classes)

    return scores, risk_classes


def coastal_flood_inundation(
    dem: np.ndarray,
    water_level: float,
    sea_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates coastal flood inundation using a hydrologically connected bathtub model.

    Identifies cells that are flooded by a given water level rise. A cell is flooded if
    its elevation is less than or equal to the water level AND it is connected to a
    sea/water source cell via other flooded cells (8-connectivity).

    Args:
        dem: 2D NumPy array containing elevation values. NaNs represent no-data.
        water_level: Target water level/elevation for flooding (e.g. 2.0 meters).
        sea_mask: Optional 2D boolean array of the same shape as dem marking sea/water
            source cells. If omitted, all boundary cells with elevation <= 0 are used.

    Returns:
        Tuple of:
          - flooded: 2D boolean NumPy array where True indicates flooded cells.
          - water_depth: 2D NumPy array containing water depth (water_level - dem)
            for flooded cells, and 0.0 elsewhere (NaNs preserved).
    """
    dem_arr = np.asarray(dem, dtype=np.float64)
    shape = dem_arr.shape

    if dem_arr.ndim != 2:
        raise ValueError("DEM must be a 2D array")

    valid = np.isfinite(dem_arr)

    # 1. Identify potentially flooded cells (below or equal to water level)
    potential = (dem_arr <= water_level) & valid

    # 2. Determine sea mask/water source seeds
    if sea_mask is not None:
        seeds = np.asarray(sea_mask, dtype=bool)
        if seeds.shape != shape:
            raise ValueError("sea_mask shape must match dem shape")
    else:
        # Create default sea mask: boundary cells with elevation <= 0
        seeds = np.zeros(shape, dtype=bool)
        # Check boundary rows and columns
        if shape[0] > 0 and shape[1] > 0:
            seeds[0, :] = dem_arr[0, :] <= 0.0
            seeds[-1, :] = dem_arr[-1, :] <= 0.0
            seeds[:, 0] = dem_arr[:, 0] <= 0.0
            seeds[:, -1] = dem_arr[:, -1] <= 0.0
        seeds &= valid

    # Only keep seeds that are actually below the water level
    seeds = seeds & potential

    if not np.any(seeds):
        # No water source is flooded/active
        return np.zeros(shape, dtype=bool), np.zeros(shape, dtype=np.float64)

    # 3. Connectivity analysis using 8-connectivity
    structure = np.ones((3, 3), dtype=bool)
    labeled_array, num_features = scipy.ndimage.label(potential, structure=structure)

    # Extract labels at seed locations
    seed_labels = labeled_array[seeds]
    unique_seed_labels = np.unique(seed_labels)
    unique_seed_labels = unique_seed_labels[unique_seed_labels > 0]

    # Mask flooded cells
    flooded = np.isin(labeled_array, unique_seed_labels)

    # 4. Calculate water depth: water_level - dem
    water_depth = np.zeros(shape, dtype=np.float64)
    water_depth[flooded] = water_level - dem_arr[flooded]
    water_depth[~valid] = np.nan

    return flooded, water_depth


def socio_economic_flood_risk(
    hazard_depth: np.ndarray,
    building_exposure: np.ndarray,
    social_vulnerability: np.ndarray,
    method: str = "multiplicative",
    w_hazard: float = 0.4,
    w_exposure: float = 0.3,
    w_vulnerability: float = 0.3,
) -> tuple[np.ndarray, list[list[str]]]:
    """Calculates the composite Socio-Economic Flood Risk Index across a grid.

    Combines flood hazard depth (or susceptibility), building exposure, and
    social vulnerability into a unified risk score [0, 100] and risk classes.

    Args:
        hazard_depth: 2D NumPy array of shape (R, C) containing flood depth (meters)
            or susceptibility scores.
        building_exposure: 2D NumPy array of shape (R, C) containing building footprints
            or asset value density.
        social_vulnerability: 2D NumPy array of shape (R, C) containing Social
            Vulnerability Index (SVI) values.
        method: Risk calculation method: 'multiplicative' (H * E * V) or
            'additive' (weighted linear combination).
        w_hazard: Weight for hazard in additive method.
        w_exposure: Weight for exposure in additive method.
        w_vulnerability: Weight for vulnerability in additive method.

    Returns:
        Tuple of:
          - risk_scores: 2D NumPy array of shape (R, C) containing risk scores [0, 100].
          - risk_classes: List of lists of risk class category strings
            ('Low', 'Moderate', 'High', 'Very High').
    """
    h = np.asarray(hazard_depth, dtype=np.float64)
    e = np.asarray(building_exposure, dtype=np.float64)
    v = np.asarray(social_vulnerability, dtype=np.float64)

    shape = h.shape
    if e.shape != shape or v.shape != shape:
        raise ValueError("All input arrays must have the same shape")

    valid = np.isfinite(h) & np.isfinite(e) & np.isfinite(v)

    if not np.any(valid):
        return np.zeros_like(h), [["Low" for _ in range(shape[1])] for _ in range(shape[0])]

    # Helper function to normalize arrays to [0.0, 100.0]
    def min_max_normalize(arr: np.ndarray) -> np.ndarray:
        val_subset = arr[valid]
        mn = np.min(val_subset)
        mx = np.max(val_subset)
        rng = mx - mn
        if rng <= 0.0:
            rng = 1.0
        return np.clip((arr - mn) / rng * 100.0, 0.0, 100.0)

    # Normalize inputs
    h_norm = min_max_normalize(h)
    e_norm = min_max_normalize(e)
    v_norm = min_max_normalize(v)

    method_lower = method.lower()
    if method_lower == "multiplicative":
        # H * E * V / 10000 -> scales to [0, 100] since each is [0, 100]
        scores = (h_norm * e_norm * v_norm) / 10000.0
    elif method_lower == "additive":
        w_sum = w_hazard + w_exposure + w_vulnerability
        if w_sum <= 0.0:
            w_sum = 1.0
        scores = (h_norm * w_hazard + e_norm * w_exposure + v_norm * w_vulnerability) / w_sum
    else:
        raise ValueError(f"Unknown risk calculation method: {method}")

    scores = np.clip(scores, 0.0, 100.0)
    scores[~valid] = np.nan

    # Risk classification
    risk_classes = []
    for r in range(shape[0]):
        row_classes = []
        for c in range(shape[1]):
            val = scores[r, c]
            if not np.isfinite(val):
                row_classes.append("Low")
            elif val >= 75.0:
                row_classes.append("Very High")
            elif val >= 50.0:
                row_classes.append("High")
            elif val >= 25.0:
                row_classes.append("Moderate")
            else:
                row_classes.append("Low")
        row_classes = list(row_classes)
        risk_classes.append(row_classes)

    return scores, risk_classes


def coastal_surge_inundation(
    dem: np.ndarray,
    surge_height: float,
    sea_mask: np.ndarray,
    cell_size: float = 10.0,
    wave_runup: float = 0.0,
    distance_decay: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates hydrologically connected coastal storm surge inundation.

    Args:
        dem: 2D NumPy elevation array (m).
        surge_height: Storm surge water surface elevation (m).
        sea_mask: 2D boolean NumPy array where True indicates sea boundary cells.
        cell_size: Grid cell size in meters (default 10.0).
        wave_runup: Additional wave runup height (m).
        distance_decay: Optional decay rate per kilometer from sea coast (default 0.0).

    Returns:
        A tuple of:
          - inundation_depth: 2D NumPy float array of flooded water depth (m).
          - flooded_mask: 2D NumPy boolean array indicating flooded cells.
    """
    dem_arr = np.asarray(dem, dtype=np.float64)
    sea_arr = np.asarray(sea_mask, dtype=bool)

    if dem_arr.ndim != 2:
        raise ValueError("dem must be a 2D array")
    if sea_arr.shape != dem_arr.shape:
        raise ValueError("sea_mask shape must match dem shape")

    rows, cols = dem_arr.shape
    eff_surge = surge_height + wave_runup

    flooded = np.zeros((rows, cols), dtype=bool)
    depth = np.zeros((rows, cols), dtype=np.float64)

    queue: deque[tuple[int, int, float]] = deque()

    sea_indices = np.argwhere(sea_arr)
    for r, c in sea_indices:
        flooded[r, c] = True
        depth[r, c] = max(0.0, eff_surge - dem_arr[r, c])
        queue.append((int(r), int(c), 0.0))

    neighbors = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    while queue:
        r, c, dist_km = queue.popleft()

        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if not flooded[nr, nc] and np.isfinite(dem_arr[nr, nc]):
                    step_km = (math.sqrt(dr**2 + dc**2) * cell_size) / 1000.0
                    next_dist = dist_km + step_km
                    surge_decayed = max(0.0, eff_surge - next_dist * distance_decay)
                    if dem_arr[nr, nc] <= surge_decayed:
                        flooded[nr, nc] = True
                        depth[nr, nc] = surge_decayed - dem_arr[nr, nc]
                        queue.append((nr, nc, next_dist))

    return depth, flooded


def urban_stormwater_peak_runoff(
    catchment_area_ha: float,
    land_use_ratios: dict[str, float],
    rainfall_intensity_mm_hr: float,
) -> dict:
    """Calculates peak stormwater runoff discharge Q (m^3/s) using Rational Method.

    Args:
        catchment_area_ha: Total catchment area in hectares float.
        land_use_ratios: Dict mapping land use types ("roofs", "pavement") to fractions.
        rainfall_intensity_mm_hr: Design storm rainfall intensity in mm/hr float.

    Returns:
        Dict containing stormwater discharge results:
          - composite_runoff_coefficient: Composite runoff coefficient C in [0.05, 0.95].
          - peak_discharge_m3_s: Peak discharge rate Q in m^3/s.
          - total_runoff_volume_m3: Total runoff volume assuming 1-hour storm (m^3).
    """
    if catchment_area_ha <= 0 or rainfall_intensity_mm_hr <= 0:
        raise ValueError("catchment_area_ha and rainfall_intensity_mm_hr must be positive.")

    c_table = {
        "roofs": 0.90,
        "pavement": 0.85,
        "commercial": 0.80,
        "residential": 0.50,
        "lawns": 0.20,
        "forest": 0.10,
        "parks": 0.15,
    }

    c_weighted = 0.0
    weight_sum = 0.0
    for ltype, ratio in land_use_ratios.items():
        c_val = c_table.get(ltype.lower(), 0.50)
        c_weighted += c_val * ratio
        weight_sum += ratio

    if weight_sum > 0:
        c_comp = float(c_weighted / weight_sum)
    else:
        c_comp = 0.50

    q_peak = 0.00278 * c_comp * float(rainfall_intensity_mm_hr) * float(catchment_area_ha)
    vol_m3 = q_peak * 3600.0

    return {
        "composite_runoff_coefficient": c_comp,
        "peak_discharge_m3_s": q_peak,
        "total_runoff_volume_m3": vol_m3,
    }
