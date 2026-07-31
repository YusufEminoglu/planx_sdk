# -*- coding: utf-8 -*-
"""Urban Morphology Spacemate, Density Matrix, Fractal & Block Porosity Engines."""

from __future__ import annotations

from typing import Any

import numpy as np


def spacemate_density_matrix(
    fsi: float | np.ndarray,
    gsi: float | np.ndarray,
    layers: float | np.ndarray | None = None,
) -> dict[str, Any]:
    """Calculates Spacemate Density Matrix metrics (FSI, GSI, OSR, L) and typologies.

    Follows Berghauser Pont & Haupt urban density taxonomy.

    Args:
        fsi: Floor Space Index (total floor area / site area).
        gsi: Ground Space Index (building footprint area / site area).
        layers: Average layer height L. If None, computed as FSI / GSI.

    Returns:
        Dict containing FSI, GSI, OSR (Open Space Ratio), L, and typology label.
    """
    fsi_arr = np.asarray(fsi, dtype=np.float64)
    gsi_arr = np.clip(np.asarray(gsi, dtype=np.float64), 1e-4, 1.0)

    if layers is None:
        l_arr = fsi_arr / gsi_arr
    else:
        l_arr = np.asarray(layers, dtype=np.float64)

    osr_arr = (1.0 - gsi_arr) / np.maximum(fsi_arr, 1e-4)

    typology = []
    fsi_flat = np.atleast_1d(fsi_arr)
    gsi_flat = np.atleast_1d(gsi_arr)

    for i in range(len(fsi_flat)):
        f, g = fsi_flat[i], gsi_flat[i]
        if f > 2.0 and g > 0.4:
            typology.append("High-Density Courtyard")
        elif f > 2.0 and g <= 0.4:
            typology.append("High-Rise Tower Block")
        elif f <= 1.0 and g <= 0.3:
            typology.append("Low-Density Suburban")
        elif g > 0.5:
            typology.append("Mid-Rise Dense Block")
        else:
            typology.append("Open Urban Fabric")

    return {
        "fsi": fsi_arr,
        "gsi": gsi_arr,
        "osr": osr_arr,
        "layers_l": l_arr,
        "typology_class": typology[0] if np.ndim(fsi) == 0 else typology,
    }


def fractal_dimension_box_counting(
    binary_footprint_grid: np.ndarray,
    min_box_size: int = 2,
    max_box_size: int = 32,
) -> dict[str, Any]:
    """Calculates Box-Counting Fractal Dimension (D) of urban footprint grid.

    Args:
        binary_footprint_grid: 2D boolean or binary array (H, W).
        min_box_size: Starting box scale size.
        max_box_size: Maximum box scale size.

    Returns:
        Dict containing fractal_dimension D, box_sizes, and counts.
    """
    grid = np.asarray(binary_footprint_grid, dtype=bool)
    h, w = grid.shape

    sizes = []
    counts = []

    box = min_box_size
    while box <= min(max_box_size, h, w):
        h_boxes = h // box
        w_boxes = w // box
        count = 0
        for r in range(h_boxes):
            for c in range(w_boxes):
                sub = grid[r * box : (r + 1) * box, c * box : (c + 1) * box]
                if np.any(sub):
                    count += 1
        sizes.append(box)
        counts.append(count)
        box *= 2

    if len(sizes) > 1 and len(counts) > 1:
        log_inv_r = np.log(1.0 / np.array(sizes, dtype=np.float64))
        log_n = np.log(np.array(counts, dtype=np.float64) + 1e-12)
        slope, _ = np.polyfit(log_inv_r, log_n, 1)
        fractal_d = float(slope)
    else:
        fractal_d = 1.0

    return {
        "fractal_dimension": abs(fractal_d),
        "box_sizes": sizes,
        "box_counts": counts,
    }


def block_porosity_and_grain_index(
    building_areas_m2: np.ndarray,
    block_area_m2: float,
    block_perimeter_m: float,
) -> dict[str, Any]:
    """Calculates Block Porosity and Urban Grain Coarseness Index.

    Args:
        building_areas_m2: Array of building footprint areas in m^2.
        block_area_m2: Total block area in m^2.
        block_perimeter_m: Total block perimeter in meters.

    Returns:
        Dict containing porosity ratio, grain coarseness index, and building count.
    """
    b_areas = np.asarray(building_areas_m2, dtype=np.float64)
    built_area = float(np.sum(b_areas))
    porosity = max(0.0, 1.0 - (built_area / max(block_area_m2, 1e-6)))

    mean_b_area = float(np.mean(b_areas)) if len(b_areas) > 0 else 0.0
    grain_index = (mean_b_area / max(block_area_m2, 1e-6)) * 100.0

    return {
        "porosity_ratio": porosity,
        "grain_coarseness_index": grain_index,
        "building_count": len(b_areas),
        "mean_building_area_m2": mean_b_area,
    }
