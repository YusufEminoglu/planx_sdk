# -*- coding: utf-8 -*-
"""Generative Urban Subdivision & Solar Rights Envelope Engines."""

from __future__ import annotations

from typing import Any

import numpy as np


def recursive_parcel_subdivision(
    bbox_width_m: float,
    bbox_height_m: float,
    min_parcel_area_m2: float = 300.0,
    max_aspect_ratio: float = 3.0,
    depth: int = 0,
    max_depth: int = 4,
) -> dict[str, Any]:
    """Generates Recursive OBB Parcel Subdivision Layout.

    Args:
        bbox_width_m: Bounding box width in meters.
        bbox_height_m: Bounding box height in meters.
        min_parcel_area_m2: Minimum allowable parcel area in m^2.
        max_aspect_ratio: Maximum allowable parcel aspect ratio.
        depth: Current recursion depth.
        max_depth: Maximum recursion depth limit.

    Returns:
        Dict containing parcel polygons coordinates list, parcel count, and mean parcel area.
    """
    parcels: list[dict[str, Any]] = []

    def _subdivide(x: float, y: float, w: float, h: float, current_depth: int) -> None:
        area = w * h
        aspect = max(w / max(h, 1e-6), h / max(w, 1e-6))

        if (
            current_depth >= max_depth
            or area < min_parcel_area_m2 * 2.0
            or aspect > max_aspect_ratio
        ):
            coords = [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
            parcels.append({"coordinates": coords, "area_m2": float(area), "width": w, "height": h})
            return

        if w >= h:
            split = w * 0.5
            _subdivide(x, y, split, h, current_depth + 1)
            _subdivide(x + split, y, w - split, h, current_depth + 1)
        else:
            split = h * 0.5
            _subdivide(x, y, w, split, current_depth + 1)
            _subdivide(x, y + split, w, h - split, current_depth + 1)

    _subdivide(0.0, 0.0, bbox_width_m, bbox_height_m, depth)
    areas = [float(p["area_m2"]) for p in parcels]

    return {
        "parcels": parcels,
        "parcel_count": len(parcels),
        "mean_parcel_area_m2": float(np.mean(areas)),
        "min_parcel_area_m2": float(np.min(areas)),
    }


def solar_envelope_buildable_volume(
    parcel_width_m: float,
    parcel_depth_m: float,
    sun_altitude_angle_deg: float = 45.0,
    setback_m: float = 3.0,
    max_height_m: float = 30.0,
) -> dict[str, Any]:
    """Generates Buildable 3D Volume satisfying Solar Rights Solar Envelope.

    Args:
        parcel_width_m: Parcel width in meters.
        parcel_depth_m: Parcel depth in meters.
        sun_altitude_angle_deg: Solar altitude angle in degrees.
        setback_m: Front and side setback distance in meters.
        max_height_m: Maximum zoning height limit in meters.

    Returns:
        Dict containing maximum solar height, total buildable volume (m^3),
        and buildable footprint area.
    """
    w_eff = max(0.0, parcel_width_m - 2.0 * setback_m)
    d_eff = max(0.0, parcel_depth_m - 2.0 * setback_m)
    footprint_area = w_eff * d_eff

    rad = np.radians(sun_altitude_angle_deg)
    solar_height_cap = float(setback_m * np.tan(rad))
    allowed_height = float(min(max_height_m, max(3.0, solar_height_cap + 12.0)))

    volume_m3 = float(footprint_area * allowed_height * 0.85)

    return {
        "buildable_volume_m3": volume_m3,
        "buildable_footprint_m2": float(footprint_area),
        "allowed_building_height_m": allowed_height,
        "solar_height_cap_m": solar_height_cap,
    }
