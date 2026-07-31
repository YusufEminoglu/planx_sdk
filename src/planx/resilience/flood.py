# -*- coding: utf-8 -*-
"""Flood risk and pluvial vulnerability models."""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Optional

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


def detention_basin_sizing(
    catchment_area_ha: float,
    cn_pre: float,
    cn_post: float,
    design_storm_mm: float,
    storm_duration_hr: float = 6.0,
) -> dict:
    """Sizes urban stormwater detention basins using the SCS Curve Number method.

    Computes the required detention storage volume from the difference between
    pre- and post-development runoff depths for a given design storm event.

    Args:
        catchment_area_ha: Catchment area in hectares.
        cn_pre: Pre-development SCS Curve Number (1-100).
        cn_post: Post-development SCS Curve Number (1-100).
        design_storm_mm: Design storm rainfall depth in millimeters.
        storm_duration_hr: Storm event duration in hours (default 6.0).

    Returns:
        Dict containing detention basin sizing results:
          - runoff_pre_mm: Float pre-development runoff depth (mm).
          - runoff_post_mm: Float post-development runoff depth (mm).
          - detention_depth_mm: Float required detention depth (mm).
          - detention_volume_m3: Float required storage volume (m^3).
          - peak_inflow_m3_s: Float estimated peak inflow rate (m^3/s).
    """
    if catchment_area_ha <= 0:
        raise ValueError("catchment_area_ha must be positive.")
    if not (1.0 <= cn_pre <= 100.0):
        raise ValueError("cn_pre must be between 1 and 100.")
    if not (1.0 <= cn_post <= 100.0):
        raise ValueError("cn_post must be between 1 and 100.")
    if design_storm_mm <= 0:
        raise ValueError("design_storm_mm must be positive.")

    P = float(design_storm_mm)

    # SCS runoff equation: Q = (P - 0.2*S)^2 / (P + 0.8*S) when P > 0.2*S
    def _scs_runoff(cn: float) -> float:
        S = 25400.0 / max(cn, 1.0) - 254.0  # potential maximum retention (mm)
        Ia = 0.2 * S  # initial abstraction
        if P <= Ia:
            return 0.0
        return (P - Ia) ** 2 / (P + 0.8 * S)

    q_pre = _scs_runoff(cn_pre)
    q_post = _scs_runoff(cn_post)
    detention_mm = max(0.0, q_post - q_pre)

    # Volume = depth * area (convert ha to m^2, mm to m)
    area_m2 = catchment_area_ha * 10000.0
    volume_m3 = detention_mm * 0.001 * area_m2

    # Triangular hydrograph peak estimate: Qp = 0.208 * A * Q / Tp
    # Tp ≈ 0.6 * Tc, with Tc ≈ storm_duration_hr for simplicity
    tp_hr = max(0.6 * storm_duration_hr, 1e-6)
    area_km2 = catchment_area_ha / 100.0
    peak_inflow = 0.208 * area_km2 * q_post / tp_hr

    return {
        "runoff_pre_mm": round(q_pre, 4),
        "runoff_post_mm": round(q_post, 4),
        "detention_depth_mm": round(detention_mm, 4),
        "detention_volume_m3": round(volume_m3, 2),
        "peak_inflow_m3_s": round(peak_inflow, 6),
    }


def scs_unit_hydrograph(
    watershed_area_km2: float,
    curve_number: float,
    rainfall_mm: float,
    storm_duration_hr: float,
    time_of_concentration_hr: float,
    dt_minutes: float = 5.0,
    peak_rate_factor: float = 484.0,
) -> dict:
    """Calculates the SCS Unit Hydrograph for storm runoff routing.

    Args:
        watershed_area_km2: Watershed area in square kilometers.
        curve_number: SCS Curve Number (30 to 100).
        rainfall_mm: Total rainfall depth in millimeters.
        storm_duration_hr: Duration of the storm event in hours.
        time_of_concentration_hr: Time of concentration in hours.
        dt_minutes: Time step for the hydrograph in minutes (default 5.0).
        peak_rate_factor: Peak rate factor (default 484.0).

    Returns:
        Dict containing:
          - time_minutes: 1D NumPy array of time steps in minutes.
          - discharge_m3s: 1D NumPy array of discharge at each time step in m^3/s.
          - peak_discharge_m3s: Float peak discharge in m^3/s.
          - time_to_peak_hr: Float time to peak in hours.
          - total_runoff_mm: Float total runoff depth in mm.
          - total_volume_m3: Float total runoff volume in m^3.
          - lag_time_hr: Float lag time in hours.
          - retention_mm: Float potential maximum retention in mm.
    """
    if watershed_area_km2 <= 0:
        raise ValueError("watershed_area_km2 must be positive.")
    if not (30.0 <= curve_number <= 100.0):
        raise ValueError("curve_number must be between 30 and 100.")
    if rainfall_mm < 0:
        raise ValueError("rainfall_mm must be non-negative.")
    if storm_duration_hr <= 0:
        raise ValueError("storm_duration_hr must be positive.")
    if time_of_concentration_hr <= 0:
        raise ValueError("time_of_concentration_hr must be positive.")
    if dt_minutes <= 0:
        raise ValueError("dt_minutes must be positive.")
    if peak_rate_factor <= 0:
        raise ValueError("peak_rate_factor must be positive.")

    # 2. Potential Maximum Retention
    s_retention = (25400.0 / curve_number) - 254.0

    # 3. Excess Rainfall (SCS CN Method)
    ia = 0.2 * s_retention
    if rainfall_mm <= ia:
        q_total = 0.0
    else:
        q_total = ((rainfall_mm - ia) ** 2) / (rainfall_mm - ia + s_retention)

    # 4. Time Parameters
    t_lag = 0.6 * time_of_concentration_hr
    dt_hr = dt_minutes / 60.0
    tp = (dt_hr / 2.0) + t_lag

    # 5. Peak Discharge
    qp = (peak_rate_factor / 484.0) * 0.208 * watershed_area_km2 * q_total / tp

    # 6. SCS Dimensionless Unit Hydrograph Ordinates
    t_ratio = np.array(
        [
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
            1.6,
            1.7,
            1.8,
            1.9,
            2.0,
            2.2,
            2.4,
            2.6,
            2.8,
            3.0,
            3.2,
            3.4,
            3.6,
            3.8,
            4.0,
            4.5,
            5.0,
        ],
        dtype=np.float64,
    )

    q_ratio = np.array(
        [
            0.000,
            0.030,
            0.100,
            0.190,
            0.310,
            0.470,
            0.660,
            0.820,
            0.930,
            0.990,
            1.000,
            0.990,
            0.930,
            0.860,
            0.780,
            0.680,
            0.560,
            0.460,
            0.390,
            0.330,
            0.280,
            0.207,
            0.147,
            0.107,
            0.077,
            0.055,
            0.040,
            0.029,
            0.021,
            0.015,
            0.011,
            0.005,
            0.000,
        ],
        dtype=np.float64,
    )

    # 7. Hydrograph Generation
    t_max_min = 5.0 * tp * 60.0
    time_minutes = np.arange(0.0, t_max_min + dt_minutes * 0.5, dt_minutes)

    t_over_tp = (time_minutes / 60.0) / tp

    q_interp = np.interp(t_over_tp, t_ratio, q_ratio, left=0.0, right=0.0)
    discharge_m3s = q_interp * qp

    total_volume_m3 = q_total * watershed_area_km2 * 1000.0

    return {
        "time_minutes": time_minutes,
        "discharge_m3s": discharge_m3s,
        "peak_discharge_m3s": float(qp),
        "time_to_peak_hr": float(tp),
        "total_runoff_mm": float(q_total),
        "total_volume_m3": float(total_volume_m3),
        "lag_time_hr": float(t_lag),
        "retention_mm": float(s_retention),
    }


def stormwater_retention_basin_design(
    drainage_area_ha: float,
    impervious_ratio: float,
    rainfall_depth_mm: float,
    soil_infiltration_rate_mmh: float,
    max_allowable_drain_hours: float = 48.0,
    basin_safety_factor: float = 1.2,
) -> dict[str, Any]:
    """Engineers green infrastructure stormwater retention & infiltration basin
    volumes, surface area, and drain-down time.

    Args:
        drainage_area_ha: Drainage basin area in hectares (> 0).
        impervious_ratio: Ratio of impervious surface [0, 1].
        rainfall_depth_mm: Design storm rainfall depth in mm (> 0).
        soil_infiltration_rate_mmh: Saturated hydraulic conductivity K_sat in mm/hour (> 0).
        max_allowable_drain_hours: Max allowed emptying time (default 48.0 h).
        basin_safety_factor: Safety multiplier for design storage (default 1.2).

    Returns:
        Dict with keys:
        - `runoff_volume_m3`: float
        - `design_storage_volume_m3`: float
        - `max_basin_depth_m`: float
        - `min_basin_surface_area_m2`: float
        - `actual_draindown_hours`: float
        - `runoff_coefficient`: float
        - `is_drain_time_compliant`: bool
    """
    if drainage_area_ha <= 0:
        raise ValueError("drainage_area_ha must be positive.")
    if not (0.0 <= impervious_ratio <= 1.0):
        raise ValueError("impervious_ratio must be between 0 and 1.")
    if rainfall_depth_mm <= 0:
        raise ValueError("rainfall_depth_mm must be positive.")
    if soil_infiltration_rate_mmh <= 0:
        raise ValueError("soil_infiltration_rate_mmh must be positive.")

    a_m2 = float(drainage_area_ha * 10000.0)
    c = 0.05 + 0.9 * float(impervious_ratio)
    r_mm = c * float(rainfall_depth_mm)
    v_raw_m3 = (r_mm / 1000.0) * a_m2
    v_design_m3 = v_raw_m3 * float(basin_safety_factor)

    d_max_m = (float(soil_infiltration_rate_mmh) / 1000.0) * float(max_allowable_drain_hours)

    a_basin_m2 = v_design_m3 / d_max_m if d_max_m > 0 else float("inf")

    t_drain_hours = (d_max_m * 1000.0) / float(soil_infiltration_rate_mmh)

    return {
        "runoff_volume_m3": float(v_raw_m3),
        "design_storage_volume_m3": float(v_design_m3),
        "max_basin_depth_m": float(d_max_m),
        "min_basin_surface_area_m2": float(a_basin_m2),
        "actual_draindown_hours": float(t_drain_hours),
        "runoff_coefficient": float(c),
        "is_drain_time_compliant": bool(t_drain_hours <= max_allowable_drain_hours),
    }


def coastal_storm_surge_inundation_engine(
    dem_grid: np.ndarray,
    coastal_mask: np.ndarray,
    storm_surge_m: float,
    sea_level_rise_m: float = 0.0,
    manning_grid: Optional[np.ndarray] = None,
    cell_size_m: float = 10.0,
) -> dict[str, Any]:
    """Simulates coastal storm surge and sea level rise hydrologic connectivity inundation.

    Args:
        dem_grid: 2D NumPy array of Digital Elevation Model heights (meters).
        coastal_mask: 2D boolean NumPy array marking coastline seed cells.
        storm_surge_m: Storm surge elevation in meters (>= 0).
        sea_level_rise_m: Sea level rise projection in meters (>= 0).
        manning_grid: Optional 2D array of Manning roughness friction values.
        cell_size_m: Grid cell spatial size in meters (> 0).

    Returns:
        Dict containing:
          - 'inundation_depth': 2D float array of flood water depths in meters.
          - 'inundated_area_m2': Total flooded ground area in m^2.
          - 'max_depth_m': Peak water depth in meters.
          - 'mean_depth_m': Mean water depth across flooded cells in meters.
          - 'volume_m3': Total inundated water volume in m^3.
          - 'connectivity_mask': 2D boolean array of hydrologically connected flooded cells.
          - 'hazard_classification_counts': Dict with cell counts for low (<0.5m), medium (0.5-1.5m), high (>1.5m).
    """
    dem = np.asarray(dem_grid, dtype=np.float64)
    c_mask = np.asarray(coastal_mask, dtype=bool)

    if dem.ndim != 2:
        raise ValueError("dem_grid must be a 2D array.")
    if c_mask.shape != dem.shape:
        raise ValueError("coastal_mask shape must match dem_grid shape.")
    if storm_surge_m < 0:
        raise ValueError("storm_surge_m must be non-negative.")
    if sea_level_rise_m < 0:
        raise ValueError("sea_level_rise_m must be non-negative.")
    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be positive.")

    total_wl = float(storm_surge_m + sea_level_rise_m)

    rows, cols = dem.shape
    inundation_depth = np.zeros((rows, cols), dtype=np.float64)
    connected = np.zeros((rows, cols), dtype=bool)

    from collections import deque
    queue = deque()

    for r in range(rows):
        for c in range(cols):
            if c_mask[r, c] and dem[r, c] < total_wl:
                connected[r, c] = True
                inundation_depth[r, c] = total_wl - dem[r, c]
                queue.append((r, c))

    neighbors_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while queue:
        r, c = queue.popleft()
        curr_wl = dem[r, c] + inundation_depth[r, c]

        for dr, dc in neighbors_dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                headloss = 0.0
                if manning_grid is not None:
                    dist = cell_size_m * (1.41421356 if dr != 0 and dc != 0 else 1.0)
                    n_val = float(manning_grid[nr, nc])
                    headloss = max(0.0, n_val * 0.01 * dist)

                avail_wl = curr_wl - headloss
                if avail_wl > dem[nr, nc]:
                    depth = avail_wl - dem[nr, nc]
                    if not connected[nr, nc] or depth > inundation_depth[nr, nc]:
                        connected[nr, nc] = True
                        inundation_depth[nr, nc] = depth
                        queue.append((nr, nc))

    flooded_mask = connected & (inundation_depth > 0.0)
    flooded_depths = inundation_depth[flooded_mask]

    cell_area = cell_size_m * cell_size_m
    inundated_area = float(np.sum(flooded_mask) * cell_area)
    max_depth = float(np.max(flooded_depths)) if len(flooded_depths) > 0 else 0.0
    mean_depth = float(np.mean(flooded_depths)) if len(flooded_depths) > 0 else 0.0
    volume = float(np.sum(flooded_depths) * cell_area)

    low_cnt = int(np.sum(flooded_mask & (inundation_depth < 0.5)))
    med_cnt = int(np.sum(flooded_mask & (inundation_depth >= 0.5) & (inundation_depth <= 1.5)))
    high_cnt = int(np.sum(flooded_mask & (inundation_depth > 1.5)))

    return {
        "inundation_depth": inundation_depth,
        "inundated_area_m2": inundated_area,
        "max_depth_m": max_depth,
        "mean_depth_m": mean_depth,
        "volume_m3": volume,
        "connectivity_mask": connected,
        "hazard_classification_counts": {
            "low": low_cnt,
            "medium": med_cnt,
            "high": high_cnt,
        },
    }

