# -*- coding: utf-8 -*-
"""Urban heat comfort and microclimate vulnerability models."""

from __future__ import annotations

from typing import Optional

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

