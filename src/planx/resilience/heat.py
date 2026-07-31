# -*- coding: utf-8 -*-
"""Urban heat comfort and microclimate vulnerability models."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def urban_heat_comfort_risk(
    impervious_share: np.ndarray,
    building_share: np.ndarray,
    green_share: np.ndarray,
    cooling_dists: np.ndarray,
    vuln_counts: np.ndarray,
    cooling_distance: float = 400.0,
    w_imperv: float = 0.30,
    w_green: float = 0.30,
    w_build: float = 0.25,
    w_vuln: float = 0.15,
) -> tuple[np.ndarray, list[list[str]]]:
    """Calculates urban heat comfort risk scores across a grid.

    Combines hardscape share, building density, green space deficit, cooling distance,
    and vulnerable asset counts into a normalized 0-100 score.

    Args:
        impervious_share: 2D NumPy array of shape (R, C) containing hardscape area share [0, 1].
        building_share: 2D NumPy array of shape (R, C) containing building footprint share [0, 1].
        green_share: 2D NumPy array of shape (R, C) containing green area share [0, 1].
        cooling_dists: 2D NumPy array of shape (R, C) containing distance to nearest green/water.
        vuln_counts: 2D NumPy array of shape (R, C) containing number of vulnerable assets in cell.
        cooling_distance: Maximum walking distance to cooling area (threshold for normalization).
        w_imperv: Weight for impervious surfaces.
        w_green: Weight for green deficit & cooling distance.
        w_build: Weight for building density.
        w_vuln: Weight for vulnerable asset count.

    Returns:
        Tuple of:
          - scores: 2D NumPy array of shape (R, C) containing heat risk scores [0, 100].
          - risk_classes: List of lists of risk category strings matching the shape
            ('Low', 'Moderate', 'High', 'Very High').
    """
    imp = np.asarray(impervious_share, dtype=np.float64)
    bld = np.asarray(building_share, dtype=np.float64)
    grn = np.asarray(green_share, dtype=np.float64)
    dst = np.asarray(cooling_dists, dtype=np.float64)
    vuln = np.asarray(vuln_counts, dtype=np.float64)

    shape = imp.shape
    if bld.shape != shape or grn.shape != shape or dst.shape != shape or vuln.shape != shape:
        raise ValueError("All input arrays must have the same shape")

    # 1. Impervious score [0, 100]
    imp_score = imp * 100.0

    # 2. Building score [0, 100]
    bld_score = bld * 100.0

    # 3. Green deficit [0, 100]
    green_deficit = (1.0 - grn) * 100.0

    # 4. Cooling distance score [0, 100]
    cooling_score = np.clip((dst / cooling_distance) * 100.0, 0.0, 100.0)

    # Combined green deficit & cooling distance score
    green_score = (green_deficit + cooling_score) * 0.5

    # 5. Vulnerability score
    vuln_score = np.clip(vuln * 20.0, 0.0, 100.0)

    # Weighted sum
    weight_sum = w_imperv + w_green + w_build + w_vuln
    if weight_sum <= 0:
        weight_sum = 1.0

    scores = (
        imp_score * w_imperv + bld_score * w_build + green_score * w_green + vuln_score * w_vuln
    ) / weight_sum

    scores = np.clip(scores, 0.0, 100.0)

    # Risk classes
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


def urban_heat_island_intensity(
    albedo: np.ndarray,
    ndvi: np.ndarray,
    building_height: np.ndarray,
    building_footprint: np.ndarray,
    wind_speed: Optional[np.ndarray] = None,
    base_intensity: float = 2.0,
) -> np.ndarray:
    """Estimates Urban Heat Island (UHI) intensity (in degrees Celsius) across a grid.

    Calculates UHI intensity using a proxy model based on surface albedo,
    Normalized Difference Vegetation Index (NDVI) proxy, building heights,
    building footprints, and wind speed.

    Args:
        albedo: 2D NumPy array of shape (R, C) containing surface albedo values [0.0, 1.0].
        ndvi: 2D NumPy array of shape (R, C) containing NDVI values [-1.0, 1.0].
        building_height: 2D NumPy array of shape (R, C) containing average building height (meters).
        building_footprint: 2D NumPy array of shape (R, C) containing building
            footprint share [0.0, 1.0].
        wind_speed: Optional 2D NumPy array of shape (R, C) containing wind speed (m/s).
            If omitted, a constant wind speed of 1.5 m/s is assumed.
        base_intensity: Baseline temperature offset in degrees Celsius.

    Returns:
        2D NumPy array of shape (R, C) containing estimated UHI intensity
        offset (in degrees Celsius).
    """
    alb = np.asarray(albedo, dtype=np.float64)
    veg = np.asarray(ndvi, dtype=np.float64)
    bh = np.asarray(building_height, dtype=np.float64)
    bf = np.asarray(building_footprint, dtype=np.float64)

    shape = alb.shape
    if veg.shape != shape or bh.shape != shape or bf.shape != shape:
        raise ValueError("All input arrays must have the same shape")

    if wind_speed is None:
        wind = np.full(shape, 1.5, dtype=np.float64)
    else:
        wind = np.asarray(wind_speed, dtype=np.float64)
        if wind.shape != shape:
            raise ValueError("wind_speed array must have the same shape as other inputs")

    # 1. Albedo factor
    alb_contrib = 3.0 * (1.0 - alb)

    # 2. Vegetation cooling factor
    veg_index = np.clip((veg + 1.0) / 2.0, 0.0, 1.0)
    veg_contrib = 4.0 * (1.0 - veg_index) - 2.0 * veg_index

    # 3. Urban canyon / building volume factor
    vol_index = bh * bf
    vol_contrib = 4.0 * np.clip(vol_index / 30.0, 0.0, 1.0)

    # 4. Wind mitigation factor
    wind_cooling = 1.0 * np.log1p(np.clip(wind, 0.0, None))
    wind_cooling = np.clip(wind_cooling, 0.0, 2.0)

    # Calculate final UHI intensity offset (in C)
    uhi_intensity = base_intensity + alb_contrib + veg_contrib + vol_contrib - wind_cooling

    # UHI cannot be negative
    uhi_intensity = np.maximum(uhi_intensity, 0.0)

    return uhi_intensity


def optimize_canopy_placement(
    indptr: np.ndarray,
    adj: np.ndarray,
    edge_lengths: np.ndarray,
    n: int,
    pedestrian_flow: np.ndarray,
    existing_canopy: np.ndarray,
    heat_index: np.ndarray,
    num_trees: int,
) -> np.ndarray:
    """Identifies the optimal street segments to plant trees for maximum heat mitigation.

    Prioritizes segments with high pedestrian traffic (choice centrality flow) and high
    heat stress that lack adequate existing canopy cover.

    Priority Score formula:
        Priority = Flow * Heat * (1.0 - Existing_Canopy)

    Args:
        indptr: CSR indptr array of shape (n + 1,).
        adj: CSR adj array of shape (E,).
        edge_lengths: CSR edge weights array of shape (E,).
        n: Number of nodes in the network.
        pedestrian_flow: 1D array of shape (E,) representing pedestrian choice centrality flow.
        existing_canopy: 1D array of shape (E,) representing existing canopy [0.0, 1.0].
        heat_index: 1D array of shape (E,) representing heat/PET index values.
        num_trees: Total budget of segments to plant new trees on.

    Returns:
        1D NumPy array of sorted edge indices representing the optimal planting sites
        (highest impact first).
    """
    flow = np.asarray(pedestrian_flow, dtype=np.float64)
    canopy = np.asarray(existing_canopy, dtype=np.float64)
    heat = np.asarray(heat_index, dtype=np.float64)

    num_edges = len(adj)
    if len(flow) != num_edges or len(canopy) != num_edges or len(heat) != num_edges:
        raise ValueError("flow, existing_canopy, and heat_index arrays must match number of edges")

    if num_trees <= 0:
        return np.array([], dtype=np.int64)

    # Normalize heat to [0.0, 1.0] if not already normalized
    min_h, max_h = np.min(heat), np.max(heat)
    h_diff = max_h - min_h
    if h_diff > 0:
        h_norm = (heat - min_h) / h_diff
    else:
        h_norm = np.zeros_like(heat)

    # Compute priority score
    priority = flow * h_norm * (1.0 - canopy)

    # Sort descending
    sorted_indices = np.argsort(priority)[::-1]

    # Return top N
    return sorted_indices[:num_trees]


def calculate_grid_sky_view_factor(
    height_grid: np.ndarray,
    resolution: float = 1.0,
    max_radius: float = 100.0,
    num_directions: int = 8,
) -> np.ndarray:
    """Calculates the Sky View Factor (SVF) for an urban height grid (DSM).

    SVF measures the proportion of sky visible from the ground at each cell, ranging from
    0.0 (completely obstructed canyon) to 1.0 (completely open sky).

    Calculates SVF using a high-performance 2D grid shifting proxy along radial directions.

    Args:
        height_grid: 2D NumPy array of shape (R, C) representing building/surface heights.
        resolution: Grid cell resolution in meters (default 1.0m).
        max_radius: Maximum radius to search for obstacles in meters (default 100.0m).
        num_directions: Number of radial directions to check (default 8).

    Returns:
        2D NumPy array of shape (R, C) containing SVF values [0.0, 1.0].
    """
    heights = np.asarray(height_grid, dtype=np.float64)
    if heights.ndim != 2:
        raise ValueError("height_grid must be a 2D array")
    if resolution <= 0.0:
        raise ValueError("resolution must be greater than 0.0")
    if max_radius <= 0.0:
        raise ValueError("max_radius must be greater than 0.0")
    if num_directions <= 0:
        raise ValueError("num_directions must be greater than 0")

    rows, cols = heights.shape
    svf = np.zeros_like(heights)

    max_pixels = max(1, int(round(max_radius / resolution)))

    # Shift helper
    def _shift_grid(grid: np.ndarray, oy: int, ox: int) -> np.ndarray:
        shifted = np.zeros_like(grid)
        if oy >= 0:
            t_ystart, t_yend = oy, rows
            s_ystart, s_yend = 0, rows - oy
        else:
            t_ystart, t_yend = 0, rows + oy
            s_ystart, s_yend = -oy, rows

        if ox >= 0:
            t_xstart, t_xend = ox, cols
            s_xstart, s_xend = 0, cols - ox
        else:
            t_xstart, t_xend = 0, cols + ox
            s_xstart, s_xend = -ox, cols

        if t_ystart < t_yend and t_xstart < t_xend:
            shifted[t_ystart:t_yend, t_xstart:t_xend] = grid[s_ystart:s_yend, s_xstart:s_xend]
        return shifted

    # Iterate over all directions
    for k in range(num_directions):
        theta = 2.0 * np.pi * k / num_directions
        max_angles_k = np.zeros_like(heights)

        # Collect unique pixel steps to minimize grid shifting operations
        seen = set()
        unique_steps = []
        for p in range(1, max_pixels + 1):
            ox = int(round(p * np.sin(theta)))
            oy = int(round(p * np.cos(theta)))
            if ox == 0 and oy == 0:
                continue
            if (oy, ox) not in seen:
                seen.add((oy, ox))
                unique_steps.append((oy, ox))

        for oy, ox in unique_steps:
            d = np.hypot(ox, oy) * resolution
            shifted = _shift_grid(heights, oy, ox)
            diff = shifted - heights
            # Calculate elevation angle
            angles = np.arctan(np.maximum(0.0, diff) / d)
            max_angles_k = np.maximum(max_angles_k, angles)

        # Contribution of this direction to SVF: cos(alpha_k)^2
        svf += np.cos(max_angles_k) ** 2

    svf /= num_directions
    return np.clip(svf, 0.0, 1.0)


def classify_local_climate_zones(
    building_share: np.ndarray,
    impervious_share: np.ndarray,
    building_height: np.ndarray,
) -> np.ndarray:
    """Classifies grid cells into Local Climate Zone (LCZ) built categories (Stewart & Oke, 2012).

    Uses building footprint share (BSF), impervious surface share (ISF), and average
    building height to classify each cell into a built LCZ type (1 to 10) or pervious (11).

    Built Categories:
        - 1: Compact high-rise (BSF > 0.4, Height > 25m)
        - 2: Compact mid-rise (BSF > 0.4, Height 10m-25m)
        - 3: Compact low-rise (BSF > 0.4, Height 3m-10m)
        - 4: Open high-rise (BSF 0.2-0.4, Height > 25m)
        - 5: Open mid-rise (BSF 0.2-0.4, Height 10m-25m)
        - 6: Open low-rise (BSF 0.2-0.4, Height 3m-10m)
        - 7: Lightweight low-rise (BSF > 0.5, Height 3m-10m, with low ISF)
        - 8: Large low-rise (BSF 0.3-0.5, Height 3m-15m, with high ISF)
        - 9: Sparsely built (BSF 0.05-0.2, Height 3m-15m)
        - 10: Heavy industry / large industrial blocks (BSF > 0.2, Height 5m-15m, high ISF)
        - 11: Pervious / natural surfaces (BSF < 0.05)

    Args:
        building_share: 2D NumPy array of shape (R, C) containing building footprint
            share [0.0, 1.0].
        impervious_share: 2D NumPy array of shape (R, C) containing impervious
            area share [0.0, 1.0].
        building_height: 2D NumPy array of shape (R, C) containing average building
            heights (meters).

    Returns:
        2D NumPy array of shape (R, C) containing LCZ class labels (integers 1 to 11).
    """
    bsf = np.asarray(building_share, dtype=np.float64)
    isf = np.asarray(impervious_share, dtype=np.float64)
    h = np.asarray(building_height, dtype=np.float64)

    shape = bsf.shape
    if isf.shape != shape or h.shape != shape:
        raise ValueError("All input arrays must have the same shape")

    # Initialize all as 11 (pervious / other)
    lcz = np.full(shape, 11, dtype=np.int32)

    # Built masks:
    pervious_mask = bsf < 0.05
    is_built = ~pervious_mask

    # Compact masks (BSF > 0.4)
    is_compact = is_built & (bsf >= 0.4)
    # Open masks (BSF 0.2 to 0.4)
    is_open = is_built & (bsf >= 0.2) & (bsf < 0.4)
    # Sparsely built (BSF 0.05 to 0.2)
    is_sparse = is_built & (bsf >= 0.05) & (bsf < 0.2)

    # Height thresholds
    is_high = h > 25.0
    is_mid = (h >= 10.0) & (h <= 25.0)
    is_low = h < 10.0

    # Assign LCZs based on hierarchy
    # Compact high-rise (1)
    lcz[is_compact & is_high] = 1
    # Compact mid-rise (2)
    lcz[is_compact & is_mid] = 2
    # Compact low-rise (3)
    lcz[is_compact & is_low] = 3

    # Open high-rise (4)
    lcz[is_open & is_high] = 4
    # Open mid-rise (5)
    lcz[is_open & is_mid] = 5
    # Open low-rise (6)
    lcz[is_open & is_low] = 6

    # Large low-rise (8)
    lcz[is_open & (h >= 3.0) & (h <= 15.0) & (isf >= 0.5)] = 8

    # Lightweight low-rise (7)
    lcz[is_compact & (h >= 3.0) & (h <= 10.0) & (isf < 0.3)] = 7

    # Heavy industry (10)
    lcz[is_open & (h >= 5.0) & (h <= 15.0) & (isf >= 0.7)] = 10

    # Sparsely built (9)
    lcz[is_sparse & (h >= 3.0) & (h <= 15.0)] = 9

    # Set any negative/invalid parameters to 11
    lcz[(bsf < 0.0) | (isf < 0.0) | (h < 0.0)] = 11

    return lcz


def calculate_solar_access(
    height_grid: np.ndarray,
    resolution: float,
    sun_altitudes: np.ndarray,
    sun_azimuths: np.ndarray,
    max_shadow_dist: float = 150.0,
) -> np.ndarray:
    """Calculates the Solar Access Index (0-100) across an urban height grid (DSM).

    Measures the percentage of time each cell receives direct sunlight across a series of
    sun positions (defined by altitude and azimuth angles).

    Args:
        height_grid: 2D NumPy array of shape (R, C) representing building/surface heights.
        resolution: Grid cell resolution in meters.
        sun_altitudes: 1D NumPy array of solar altitude angles in degrees
            (sun height above horizon).
        sun_azimuths: 1D NumPy array of solar azimuth angles in degrees
            (compass direction of sun).
        max_shadow_dist: Maximum distance in meters to trace shadows (default 150.0m).

    Returns:
        2D NumPy array of shape (R, C) containing solar access index values [0.0, 100.0].
    """
    heights = np.asarray(height_grid, dtype=np.float64)
    alts = np.asarray(sun_altitudes, dtype=np.float64)
    azis = np.asarray(sun_azimuths, dtype=np.float64)

    if heights.ndim != 2:
        raise ValueError("height_grid must be a 2D array")
    if resolution <= 0.0:
        raise ValueError("resolution must be greater than 0.0")
    if len(alts) != len(azis):
        raise ValueError("sun_altitudes and sun_azimuths must have identical length")

    rows, cols = heights.shape
    num_steps = len(alts)
    if num_steps == 0:
        return np.full(heights.shape, 100.0)

    # Shift helper
    def _shift_grid(grid: np.ndarray, oy: int, ox: int) -> np.ndarray:
        shifted = np.zeros_like(grid)
        if oy >= 0:
            t_ystart, t_yend = oy, rows
            s_ystart, s_yend = 0, rows - oy
        else:
            t_ystart, t_yend = 0, rows + oy
            s_ystart, s_yend = -oy, rows

        if ox >= 0:
            t_xstart, t_xend = ox, cols
            s_xstart, s_xend = 0, cols - ox
        else:
            t_xstart, t_xend = 0, cols + ox
            s_xstart, s_xend = -ox, cols

        if t_ystart < t_yend and t_xstart < t_xend:
            shifted[t_ystart:t_yend, t_xstart:t_xend] = grid[s_ystart:s_yend, s_xstart:s_xend]
        return shifted

    sunlit_sum = np.zeros_like(heights, dtype=np.float64)
    valid_steps = 0

    for step in range(num_steps):
        alt_deg = alts[step]
        azi_deg = azis[step]

        # If sun is below or at the horizon, it's completely shaded
        if alt_deg <= 0.0:
            continue

        valid_steps += 1

        alt_rad = np.radians(alt_deg)
        azi_rad = np.radians(azi_deg)

        # Shadow direction is opposite to the sun's azimuth
        # In grid row-down coordinates: dx = -sin(azi), dy = cos(azi)
        dx_shadow = -np.sin(azi_rad)
        dy_shadow = np.cos(azi_rad)

        # Initialize shadow height grid with the heights themselves
        shadow_height = heights.copy()

        # Find maximum propagation pixels based on maximum heights in the grid
        max_h = float(np.max(heights))
        tan_alt = np.tan(alt_rad)

        # Avoid division by zero
        max_d_proj = max_h / tan_alt if tan_alt > 0.0 else max_shadow_dist
        max_dist = min(max_shadow_dist, max_d_proj)
        max_pixels = max(1, int(round(max_dist / resolution)))

        seen = set()
        unique_steps = []
        for p in range(1, max_pixels + 1):
            ox = int(round(p * dx_shadow))
            oy = int(round(p * dy_shadow))
            if ox == 0 and oy == 0:
                continue
            if (oy, ox) not in seen:
                seen.add((oy, ox))
                unique_steps.append((oy, ox))

        for oy, ox in unique_steps:
            d = np.hypot(ox, oy) * resolution
            shifted = _shift_grid(heights, oy, ox)
            sh = shifted - d * tan_alt
            shadow_height = np.maximum(shadow_height, sh)

        # A cell is sunlit if its height is equal to or greater than the shadow height
        is_sunlit = heights >= (shadow_height - 1e-5)
        sunlit_sum += is_sunlit.astype(np.float64)

    if valid_steps == 0:
        return np.zeros_like(heights)

    return (sunlit_sum / valid_steps) * 100.0


def tree_canopy_microclimate_cooling(
    tree_coords: np.ndarray,
    canopy_radii: np.ndarray,
    lai: np.ndarray,
    grid_coords: np.ndarray,
    max_cooling_dist: float = 50.0,
) -> np.ndarray:
    """Calculates microclimate air temperature reduction Delta T_cool (°C) from tree canopies.

    Args:
        tree_coords: (T, 2) NumPy array of tree trunk coordinates.
        canopy_radii: (T,) NumPy array of tree canopy radii in meters.
        lai: (T,) Leaf Area Index array (LAI).
        grid_coords: (G, 2) NumPy array of spatial grid evaluation points.
        max_cooling_dist: Maximum cooling influence distance float in meters.

    Returns:
        1D NumPy array of shape (G,) containing temperature reduction Delta T_cool (°C).
    """
    trees = np.asarray(tree_coords, dtype=np.float64)
    radii = np.asarray(canopy_radii, dtype=np.float64)
    lai_arr = np.asarray(lai, dtype=np.float64)
    grid = np.asarray(grid_coords, dtype=np.float64)

    g_count = len(grid)
    t_count = len(trees)

    if g_count == 0 or t_count == 0:
        return np.zeros(g_count, dtype=np.float64)

    delta_t = np.zeros(g_count, dtype=np.float64)

    diffs = grid[:, None, :] - trees[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))

    for t_idx in range(t_count):
        r_t = radii[t_idx]
        lai_t = lai_arr[t_idx]
        d_t = dists[:, t_idx]

        in_canopy = d_t <= r_t
        out_canopy = (d_t > r_t) & (d_t <= (r_t + max_cooling_dist))

        t_cool_t = np.zeros(g_count, dtype=np.float64)
        t_cool_t[in_canopy] = 0.6 * lai_t

        decay_dist = d_t[out_canopy] - r_t
        t_cool_t[out_canopy] = (0.6 * lai_t) * np.exp(-3.0 * decay_dist / max_cooling_dist)

        delta_t = np.maximum(delta_t, t_cool_t)

    return delta_t


def calculate_building_solar_radiation(
    roof_areas: np.ndarray,
    svf: np.ndarray,
    solar_irradiance_kwh_m2: float = 1200.0,
    pv_efficiency: float = 0.18,
) -> dict:
    """Calculates 3D building rooftop solar radiation potential and photovoltaic energy output.

    Args:
        roof_areas: 1D array of building roof surface areas in m^2.
        svf: 1D array of local Sky View Factor (SVF) values in [0, 1].
        solar_irradiance_kwh_m2: Annual horizontal solar irradiance in kWh/m^2.
        pv_efficiency: Solar PV panel conversion efficiency float (default 0.18 for 18%).

    Returns:
        Dict containing solar potential statistics:
          - annual_radiation_kwh: 1D NumPy array of incident solar energy per building (kWh/yr).
          - annual_pv_generation_kwh: 1D array of estimated PV electricity generation (kWh/yr).
          - total_pv_generation_mwh: Float total portfolio PV generation (MWh/yr).
    """
    areas = np.asarray(roof_areas, dtype=np.float64)
    svf_arr = np.asarray(svf, dtype=np.float64)
    n = len(areas)

    if len(svf_arr) != n:
        raise ValueError("roof_areas and svf must have identical length.")

    radiation_kwh = areas * svf_arr * float(solar_irradiance_kwh_m2)
    pv_gen_kwh = radiation_kwh * float(pv_efficiency)
    total_mwh = float(np.sum(pv_gen_kwh) / 1000.0)

    return {
        "annual_radiation_kwh": radiation_kwh,
        "annual_pv_generation_kwh": pv_gen_kwh,
        "total_pv_generation_mwh": total_mwh,
    }


def urban_heat_vulnerability_index(
    uhi_intensity: np.ndarray,
    sensitivity_density: np.ndarray,
    canopy_cover_ratio: np.ndarray,
) -> dict:
    """Synthesizes the composite Urban Heat Vulnerability Index (HVI) across spatial units.

    Args:
        uhi_intensity: Array of Urban Heat Island temperature anomaly (°C).
        sensitivity_density: Array of vulnerable population density (elderly + infants / km^2).
        canopy_cover_ratio: Array of tree canopy cover fraction [0.0, 1.0].

    Returns:
        Dict containing HVI assessment:
          - hvi_score: Array of normalized HVI scores in [0, 100].
          - vulnerability_category: List of strings ("Low", "Moderate", "High", "Very High").
    """
    uhi = np.asarray(uhi_intensity, dtype=np.float64)
    sens = np.asarray(sensitivity_density, dtype=np.float64)
    canopy = np.asarray(canopy_cover_ratio, dtype=np.float64)

    if uhi.shape != sens.shape or uhi.shape != canopy.shape:
        raise ValueError("uhi_intensity, sensitivity_density, and canopy_cover_ratio must match.")

    def min_max(arr: np.ndarray) -> np.ndarray:
        min_v, max_v = np.min(arr), np.max(arr)
        if max_v > min_v:
            return (arr - min_v) / (max_v - min_v)
        return np.zeros_like(arr)

    e_norm = min_max(uhi)
    s_norm = min_max(sens)
    ac_norm = min_max(canopy)

    hvi_raw = 0.4 * e_norm + 0.4 * s_norm - 0.2 * ac_norm
    min_h, max_h = np.min(hvi_raw), np.max(hvi_raw)

    if max_h > min_h:
        hvi_score = (hvi_raw - min_h) / (max_h - min_h) * 100.0
    else:
        hvi_score = np.zeros_like(hvi_raw)

    cats = []
    for val in hvi_score.flat:
        if val >= 75.0:
            cats.append("Very High")
        elif val >= 50.0:
            cats.append("High")
        elif val >= 25.0:
            cats.append("Moderate")
        else:
            cats.append("Low")

    return {
        "hvi_score": hvi_score,
        "vulnerability_category": cats,
    }


def optimize_tree_canopy_greening(
    lst_temperatures: np.ndarray,
    air_quality_index: np.ndarray,
    pedestrian_density: np.ndarray,
    existing_canopy_ratio: np.ndarray,
    candidate_locations_coords: np.ndarray,
    budget_max_trees: int,
    cooling_radius: float = 100.0,
) -> dict[str, Any]:
    """Multi-objective greedy location-allocation solver for urban tree canopy optimization.

    Maximizes heat mitigation and air quality benefit.

    Args:
        lst_temperatures: 1D array of Land Surface Temp (°C) per candidate cell.
        air_quality_index: 1D array of Air pollution / AQI score per cell.
        pedestrian_density: 1D array of Pedestrian count / foot traffic per cell.
        existing_canopy_ratio: 1D array of Current tree canopy ratio [0, 1].
        candidate_locations_coords: 2D array of coordinates (N, 2) of candidate cells.
        budget_max_trees: Max number of trees B to plant.
        cooling_radius: Distance decay radius for microclimate cooling benefit.

    Returns:
        Dict with keys:
        - selected_indices: (B,) int array of selected cell indices
        - selected_coords: (B, 2) float array of coordinates
        - total_heat_reduction_score: float
        - priority_scores: (N,) float array of initial priority scores
        - post_greening_heat_mitigation: (N,) float array of estimated LST temp reduction (°C)
    """
    lst = np.asarray(lst_temperatures, dtype=np.float64)
    aqi = np.asarray(air_quality_index, dtype=np.float64)
    ped = np.asarray(pedestrian_density, dtype=np.float64)
    canopy = np.asarray(existing_canopy_ratio, dtype=np.float64)
    coords = np.asarray(candidate_locations_coords, dtype=np.float64)

    n_candidates = lst.shape[0]

    if (
        aqi.shape != (n_candidates,)
        or ped.shape != (n_candidates,)
        or canopy.shape != (n_candidates,)
        or coords.shape != (n_candidates, 2)
    ):
        raise ValueError("Input arrays must have matching shape corresponding to N candidates.")

    if np.any(canopy < 0) or np.any(canopy > 1):
        raise ValueError("existing_canopy_ratio must be between 0 and 1.")

    if budget_max_trees <= 0:
        raise ValueError("budget_max_trees must be greater than 0.")

    if cooling_radius <= 0:
        raise ValueError("cooling_radius must be positive.")

    def min_max(arr: np.ndarray) -> np.ndarray:
        min_v, max_v = np.min(arr), np.max(arr)
        if max_v > min_v:
            return (arr - min_v) / (max_v - min_v)
        return np.zeros_like(arr)

    lst_norm = min_max(lst)
    aqi_norm = min_max(aqi)
    ped_norm = min_max(ped)

    priority_scores = (lst_norm * 0.4 + aqi_norm * 0.3 + ped_norm * 0.3) * (1.0 - canopy)
    initial_priority_scores = priority_scores.copy()

    selected_indices = []
    total_heat_reduction_score = 0.0
    post_greening_heat_mitigation = np.zeros(n_candidates, dtype=np.float64)

    current_priority = priority_scores.copy()

    for _ in range(min(budget_max_trees, n_candidates)):
        if np.all(current_priority <= 0):
            break

        best_idx = int(np.argmax(current_priority))
        selected_indices.append(best_idx)

        # apply cooling effect to surrounding cells
        best_coord = coords[best_idx]
        dists = np.sqrt(np.sum((coords - best_coord) ** 2, axis=1))

        # Gaussian distance decay
        effect = np.exp(-(dists**2) / (2 * (cooling_radius / 3) ** 2))
        effect[dists > cooling_radius] = 0

        # update post_greening_heat_mitigation (approximate °C reduction)
        # Apply Gaussian distance-decay cooling reduction to surrounding cells
        # within cooling_radius.
        # Update priority scores of nearby cells.
        # Let's say max cooling per tree is 1.0 degree at center?
        cooling_effect = effect * 1.5  # arbitrary scaling for heat mitigation if not specified
        post_greening_heat_mitigation += cooling_effect
        total_heat_reduction_score += np.sum(cooling_effect)

        # reduce priority of nearby cells
        # The priority reduction can be proportional to the cooling effect
        # We can just reduce LST component or reduce priority directly
        reduction = effect * 0.2
        current_priority = np.maximum(0, current_priority - reduction)
        current_priority[best_idx] = 0  # Cannot plant here again

    sel_idx_arr = np.array(selected_indices, dtype=int)
    if len(sel_idx_arr) > 0:
        sel_coords = coords[sel_idx_arr]
    else:
        sel_coords = np.empty((0, 2), dtype=np.float64)

    return {
        "selected_indices": sel_idx_arr,
        "selected_coords": sel_coords,
        "total_heat_reduction_score": float(total_heat_reduction_score),
        "priority_scores": initial_priority_scores,
        "post_greening_heat_mitigation": post_greening_heat_mitigation,
    }


def surface_cool_island_simulator(
    albedo_grid: np.ndarray,
    target_albedo_grid: np.ndarray,
    solar_irradiance_wm2: float = 800.0,
    ambient_temp_c: float = 35.0,
    cell_size_m: float = 10.0,
    green_fraction_grid: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Simulates urban surface cooling and microclimate thermal comfort improvements.

    Evaluates Land Surface Temperature (LST) and PET thermal comfort mitigation resulting from
    surface albedo increases (cool roofs, cool pavements) and vegetation evapotranspiration.

    Args:
        albedo_grid: 2D NumPy array of baseline surface albedo [0.0, 1.0].
        target_albedo_grid: 2D NumPy array of modified target albedo [0.0, 1.0].
        solar_irradiance_wm2: Peak solar irradiance in W/m^2 (default 800.0).
        ambient_temp_c: Baseline ambient air temperature in deg C (default 35.0).
        cell_size_m: Spatial resolution in meters (default 10.0).
        green_fraction_grid: Optional 2D array of green vegetation cover fraction [0.0, 1.0].

    Returns:
        Dict containing:
          - 'lst_reduction_c': 2D float array of LST cooling in deg C (>= 0).
          - 'new_lst_grid': 2D float array of resulting LST in deg C.
          - 'net_radiation_change_wm2': 2D float array of net radiation reduction in W/m^2.
          - 'mean_cooling_c': Float mean LST reduction across modified cells.
          - 'max_cooling_c': Float peak LST reduction in deg C.
          - 'total_heat_mitigated_mwh': Float total energy reflection equivalent in MWh.
          - 'pet_comfort_improvement_c': 2D float array of estimated PET comfort
            improvement in deg C.
    """
    alb_base = np.asarray(albedo_grid, dtype=np.float64)
    alb_targ = np.asarray(target_albedo_grid, dtype=np.float64)

    if alb_base.ndim != 2:
        raise ValueError("albedo_grid must be a 2D array.")
    if alb_targ.shape != alb_base.shape:
        raise ValueError("target_albedo_grid shape must match albedo_grid shape.")
    if np.any((alb_base < 0.0) | (alb_base > 1.0)):
        raise ValueError("albedo_grid values must be between 0.0 and 1.0.")
    if np.any((alb_targ < 0.0) | (alb_targ > 1.0)):
        raise ValueError("target_albedo_grid values must be between 0.0 and 1.0.")
    if solar_irradiance_wm2 <= 0:
        raise ValueError("solar_irradiance_wm2 must be positive.")
    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be positive.")

    delta_albedo = np.maximum(0.0, alb_targ - alb_base)
    delta_rn = delta_albedo * solar_irradiance_wm2

    h_combined = 20.0
    lst_cooling_albedo = delta_rn / h_combined

    veg_cooling = np.zeros_like(alb_base)
    if green_fraction_grid is not None:
        g_frac = np.asarray(green_fraction_grid, dtype=np.float64)
        if g_frac.shape != alb_base.shape:
            raise ValueError("green_fraction_grid shape must match albedo_grid shape.")
        g_frac = np.clip(g_frac, 0.0, 1.0)
        veg_cooling = g_frac * 3.0

    total_lst_cooling = lst_cooling_albedo + veg_cooling

    base_lst = ambient_temp_c + (1.0 - alb_base) * 15.0
    new_lst = base_lst - total_lst_cooling

    pet_improvement = 0.6 * total_lst_cooling

    modified_mask = (delta_albedo > 0.0) | (veg_cooling > 0.0)
    mean_cool = float(np.mean(total_lst_cooling[modified_mask])) if np.any(modified_mask) else 0.0
    max_cool = float(np.max(total_lst_cooling)) if len(total_lst_cooling) > 0 else 0.0

    cell_area_m2 = cell_size_m * cell_size_m
    total_watts = np.sum(delta_rn) * cell_area_m2
    total_mwh = float((total_watts * 6.0) / 1e6)

    return {
        "lst_reduction_c": total_lst_cooling,
        "new_lst_grid": new_lst,
        "net_radiation_change_wm2": delta_rn,
        "mean_cooling_c": mean_cool,
        "max_cooling_c": max_cool,
        "total_heat_mitigated_mwh": total_mwh,
        "pet_comfort_improvement_c": pet_improvement,
    }


def wind_canopy_aerodynamic_drag_simulator(
    inflow_wind_speed_ms: float,
    tree_lai_grid: np.ndarray,
    building_frontal_density_grid: np.ndarray,
    tree_height_m: float = 10.0,
    drag_coefficient: float = 0.2,
) -> dict[str, Any]:
    """Simulates urban canopy aerodynamic drag, wind speed attenuation, and Lawson comfort classes.

    Evaluates momentum absorption from tree Leaf Area Index (LAI) and building frontal area density
    to model pedestrian-level (1.5m) wind velocity profiles and comfort classification.

    Args:
        inflow_wind_speed_ms: Open-terrain baseline wind speed in m/s at 10m height (> 0).
        tree_lai_grid: 2D NumPy array of Leaf Area Index (LAI) values [0.0, 10.0].
        building_frontal_density_grid: 2D NumPy array of building frontal area density
            lambda_f [0.0, 1.0].
        tree_height_m: Mean canopy height in meters (default 10.0).
        drag_coefficient: Canopy aerodynamic drag coefficient Cd (default 0.2).

    Returns:
        Dict containing:
          - 'pedestrian_wind_speed_ms': 2D float array of wind speed at 1.5m height in m/s.
          - 'attenuation_ratio': 2D float array of wind speed reduction ratios [0, 1].
          - 'comfort_category_grid': 2D int array of Lawson wind comfort categories (1 to 5).
          - 'mean_wind_speed_ms': Float mean pedestrian wind speed.
          - 'max_wind_speed_ms': Float peak pedestrian wind speed.
          - 'comfortable_area_ratio': Float fraction of grid cells with comfort category <= 3.
    """
    if inflow_wind_speed_ms <= 0:
        raise ValueError("inflow_wind_speed_ms must be positive.")
    if tree_height_m <= 0:
        raise ValueError("tree_height_m must be positive.")
    if drag_coefficient <= 0:
        raise ValueError("drag_coefficient must be positive.")

    lai = np.asarray(tree_lai_grid, dtype=np.float64)
    lambda_f = np.asarray(building_frontal_density_grid, dtype=np.float64)

    if lai.ndim != 2:
        raise ValueError("tree_lai_grid must be a 2D array.")
    if lambda_f.shape != lai.shape:
        raise ValueError("building_frontal_density_grid shape must match tree_lai_grid shape.")
    if np.any(lai < 0):
        raise ValueError("tree_lai_grid values must be non-negative.")
    if np.any((lambda_f < 0.0) | (lambda_f > 1.0)):
        raise ValueError("building_frontal_density_grid values must be between 0.0 and 1.0.")

    cd_eff = drag_coefficient * lai + 0.5 * lambda_f
    atten_factor = np.exp(-0.5 * cd_eff)

    u_attenuated = inflow_wind_speed_ms * atten_factor
    u_pedestrian = u_attenuated * 0.75

    attenuation_ratio = 1.0 - (u_pedestrian / inflow_wind_speed_ms)
    attenuation_ratio = np.clip(attenuation_ratio, 0.0, 1.0)

    categories = np.zeros_like(u_pedestrian, dtype=int)
    categories[u_pedestrian < 1.8] = 1
    categories[(u_pedestrian >= 1.8) & (u_pedestrian < 3.6)] = 2
    categories[(u_pedestrian >= 3.6) & (u_pedestrian < 5.3)] = 3
    categories[(u_pedestrian >= 5.3) & (u_pedestrian < 7.6)] = 4
    categories[u_pedestrian >= 7.6] = 5

    mean_u = float(np.mean(u_pedestrian))
    max_u = float(np.max(u_pedestrian))
    comfortable_ratio = float(np.sum(categories <= 3) / categories.size)

    return {
        "pedestrian_wind_speed_ms": u_pedestrian,
        "attenuation_ratio": attenuation_ratio,
        "comfort_category_grid": categories,
        "mean_wind_speed_ms": mean_u,
        "max_wind_speed_ms": max_u,
        "comfortable_area_ratio": comfortable_ratio,
    }


def heatwave_health_vulnerability_engine(
    temp_c_grid: np.ndarray,
    humidity_pct_grid: np.ndarray,
    vulnerable_pop_ratio_grid: np.ndarray,
    ac_coverage_ratio_grid: np.ndarray,
) -> dict[str, Any]:
    """Urban Heat Wave Health Vulnerability Engine.

    Calculates apparent heat index, incorporates vulnerable demographics and AC deficit,
    and estimates relative health risk scores and alert levels.

    Args:
        temp_c_grid: Ambient temperature grid in Celsius.
        humidity_pct_grid: Relative humidity percentage grid (0-100).
        vulnerable_pop_ratio_grid: Fraction of elderly/infant population per cell (0-1).
        ac_coverage_ratio_grid: Fraction of households with air conditioning per cell (0-1).

    Returns:
        Dict containing heat index grid, vulnerability score grid, mean heat index,
        and severe risk ratio.
    """
    tf = temp_c_grid * 1.8 + 32.0
    rh = humidity_pct_grid

    hi_f = (
        -42.379
        + 2.04901523 * tf
        + 10.14333127 * rh
        - 0.22475541 * tf * rh
        - 0.00683783 * (tf**2)
        - 0.05481717 * (rh**2)
        + 0.00122874 * (tf**2) * rh
        + 0.00085282 * tf * (rh**2)
        - 0.00000199 * (tf**2) * (rh**2)
    )
    hi_c = (hi_f - 32.0) / 1.8
    hi_c = np.where(temp_c_grid < 25.0, temp_c_grid, hi_c)

    ac_deficit = 1.0 - ac_coverage_ratio_grid
    vulnerability_score = (hi_c / 40.0) * (1.0 + vulnerable_pop_ratio_grid) * (1.0 + ac_deficit)

    severe_ratio = float(np.mean(vulnerability_score > 1.5))

    return {
        "heat_index_c_grid": hi_c,
        "vulnerability_score_grid": vulnerability_score,
        "mean_heat_index_c": float(np.mean(hi_c)),
        "max_heat_index_c": float(np.max(hi_c)),
        "severe_vulnerability_ratio": severe_ratio,
        "alert_level": "EXTREME"
        if severe_ratio > 0.3
        else ("WARNING" if severe_ratio > 0.1 else "ADVISORY"),
    }


def wui_ember_transport_simulator(
    fire_line_coords: np.ndarray,
    wind_speed_ms: float,
    wind_direction_deg: float,
    target_grid_coords: np.ndarray,
    canopy_height_m: float = 15.0,
) -> dict[str, Any]:
    """Wildfire Urban Interface (WUI) Ember Transport Simulator.

    Models firebrand spot ignition distribution and ember density over WUI residential zones.

    Args:
        fire_line_coords: Array of shape (M, 2) containing fire front coordinates.
        wind_speed_ms: Wind velocity in m/s (> 0).
        wind_direction_deg: Wind blowing direction in degrees (0-360).
        target_grid_coords: Array of shape (G, 2) containing grid point coordinates.
        canopy_height_m: Average tree canopy height (m).

    Returns:
        Dict containing ember density array, max spotting distance, and high-risk count.
    """
    if wind_speed_ms <= 0:
        raise ValueError("wind_speed_ms must be positive.")

    d_spot_max = 0.05 * (wind_speed_ms**1.2) * (canopy_height_m**0.8) * 10.0

    diffs = target_grid_coords[:, None, :] - fire_line_coords[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))

    ember_density = np.zeros(len(target_grid_coords), dtype=np.float64)
    for m in range(len(fire_line_coords)):
        d_m = dists[:, m]
        in_range = d_m <= d_spot_max
        ember_density[in_range] += np.exp(-3.0 * d_m[in_range] / max(d_spot_max, 1.0))

    return {
        "ember_density_grid": ember_density,
        "max_spotting_distance_m": float(d_spot_max),
        "high_ignition_risk_count": int(np.sum(ember_density > 0.5)),
    }


def green_infra_cooling_engine(
    park_coords: np.ndarray,
    park_areas_ha: np.ndarray,
    target_grid_coords: np.ndarray,
    max_cooling_dist_m: float = 500.0,
) -> dict[str, Any]:
    """Green Infrastructure Cooling Effect & Park Cool Island Simulator.

    Calculates Park Cool Island (PCI) temperature reduction decay over urban target areas.

    Args:
        park_coords: Array of shape (P, 2) for park centroids.
        park_areas_ha: Array of shape (P,) for park sizes in hectares.
        target_grid_coords: Array of shape (G, 2) for evaluation points.
        max_cooling_dist_m: Maximum temperature decay distance (m).

    Returns:
        Dict containing temperature reduction grid (C), mean cooling, and max cooling.
    """
    from scipy.spatial.distance import cdist

    p_count = len(park_coords)
    g_count = len(target_grid_coords)

    dists = cdist(target_grid_coords, park_coords)

    pci_max = 1.2 * np.log(park_areas_ha + 1.0)

    cooling_grid = np.zeros(g_count, dtype=np.float64)
    for p in range(p_count):
        d_p = dists[:, p]
        in_range = d_p <= max_cooling_dist_m
        c_p = pci_max[p] * np.exp(-3.0 * d_p[in_range] / max_cooling_dist_m)
        cooling_grid[in_range] = np.maximum(cooling_grid[in_range], c_p)

    return {
        "temperature_reduction_c_grid": cooling_grid,
        "mean_cooling_c": float(np.mean(cooling_grid)),
        "max_cooling_c": float(np.max(cooling_grid)),
        "cooled_area_ratio": float(np.mean(cooling_grid > 0.5)),
    }


def calculate_solar_radiation_surface(
    latitude_deg: float,
    day_of_year: int = 172,
    cloud_cover_ratio: float = 0.2,
) -> dict[str, Any]:
    """Calculates Potential Daily Solar Surface Radiation (W/m^2).

    Args:
        latitude_deg: Latitude in decimal degrees [-90, 90].
        day_of_year: Day of year integer [1, 365] (default 172 summer solstice).
        cloud_cover_ratio: Cloud cover fraction [0, 1] (default 0.2).

    Returns:
        Dict containing peak solar radiation (W/m^2), daily insolation (kWh/m^2),
        and solar declination.
    """
    if abs(latitude_deg) > 90.0:
        raise ValueError("latitude_deg must be between -90 and 90.")
    if day_of_year < 1 or day_of_year > 366:
        raise ValueError("day_of_year must be between 1 and 366.")

    lat_rad = np.radians(latitude_deg)
    declination = np.radians(23.45 * np.sin(np.radians(360.0 / 365.0 * (day_of_year - 81))))

    ws_arg = np.clip(-np.tan(lat_rad) * np.tan(declination), -1.0, 1.0)
    sunset_hour_angle = np.arccos(ws_arg)

    g_sc = 1367.0
    dr = 1.0 + 0.033 * np.cos(np.radians(360.0 * day_of_year / 365.0))

    h_extra = (
        (24.0 * 60.0 / np.pi)
        * g_sc
        * dr
        * (
            sunset_hour_angle * np.sin(lat_rad) * np.sin(declination)
            + np.cos(lat_rad) * np.cos(declination) * np.sin(sunset_hour_angle)
        )
    )

    solar_radiation_wh_m2 = h_extra * (0.75 - 0.5 * cloud_cover_ratio) / 3600.0
    peak_wm2 = float(g_sc * dr * (0.75 - 0.5 * cloud_cover_ratio))

    return {
        "peak_solar_radiation_wm2": peak_wm2,
        "daily_insolation_kwh_m2": float(solar_radiation_wh_m2 / 1000.0),
        "solar_declination_deg": float(np.degrees(declination)),
        "daylight_hours": float(np.degrees(2.0 * sunset_hour_angle) / 15.0),
    }
