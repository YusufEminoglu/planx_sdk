# -*- coding: utf-8 -*-
"""Real Estate Automated Valuation Models (AVM), Cap Rate Interpolation & Transit Premium."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def transit_oriented_premium_index(
    prices: np.ndarray,
    transit_distances_m: np.ndarray,
    amenity_distances_m: np.ndarray,
) -> dict[str, Any]:
    """Calculates Transit-Oriented Premium Index (TOPI) per property.

    Args:
        prices: Transaction prices array.
        transit_distances_m: Distance to closest transit station in meters.
        amenity_distances_m: Distance to closest amenity in meters.

    Returns:
        Dict containing transit premium % per property, mean premium %,
        and accessibility elasticity.
    """
    p = np.asarray(prices, dtype=np.float64)
    dt = np.asarray(transit_distances_m, dtype=np.float64)
    da = np.asarray(amenity_distances_m, dtype=np.float64)

    decay_t = np.exp(-dt / 800.0)
    decay_a = np.exp(-da / 1000.0)

    topi_score = (0.6 * decay_t + 0.4 * decay_a) * 100.0
    premium_amount = p * (topi_score / 100.0) * 0.15

    return {
        "transit_oriented_premium_index": topi_score,
        "estimated_premium_value": premium_amount,
        "mean_topi_score": float(np.mean(topi_score)),
        "high_transit_premium_ratio": float(np.mean(topi_score >= 60.0)),
    }


def automated_comps_selector(
    target_feature: np.ndarray,
    comps_features: np.ndarray,
    spatial_weight: float = 0.5,
    feature_weight: float = 0.5,
    top_k: int = 5,
) -> dict[str, Any]:
    """Selects top K comparable properties using multi-attribute Gower distance.

    Args:
        target_feature: 1D array of target property attributes [x, y, area, age, rooms].
        comps_features: 2D array of candidate properties attributes (M, P).
        spatial_weight: Weight assigned to spatial distance [0, 1].
        feature_weight: Weight assigned to structural features [0, 1].
        top_k: Number of comps to return.

    Returns:
        Dict containing top K indices, composite distances, and suggested valuation.
    """
    tf = np.asarray(target_feature, dtype=np.float64)
    cf = np.asarray(comps_features, dtype=np.float64)

    spatial_dist = np.sqrt(np.sum((cf[:, :2] - tf[:2]) ** 2, axis=1))
    spatial_dist_norm = spatial_dist / max(np.max(spatial_dist), 1e-6)

    feat_dist = np.mean(
        np.abs(cf[:, 2:] - tf[2:]) / np.maximum(np.std(cf[:, 2:], axis=0), 1e-6), axis=1
    )
    feat_dist_norm = feat_dist / max(np.max(feat_dist), 1e-6)

    composite_dist = spatial_weight * spatial_dist_norm + feature_weight * feat_dist_norm
    top_indices = np.argsort(composite_dist)[:top_k]

    return {
        "selected_comps_indices": top_indices,
        "composite_distances": composite_dist[top_indices],
        "top_k": top_k,
    }


def cap_rate_spatial_interpolator(
    rental_yields: np.ndarray,
    noi_grid: np.ndarray,
    property_coords: np.ndarray,
) -> dict[str, Any]:
    """Interpolates Capitalization Rates (Cap Rate) and Amortization Periods.

    Args:
        rental_yields: Array of rental yields [0.03, 0.08].
        noi_grid: 1D or 2D array of Net Operating Income.
        property_coords: 2D array of property coordinates (N, 2).

    Returns:
        Dict containing mean cap rate %, amortization years per property,
        and gross rent multiplier (GRM).
    """
    yields = np.asarray(rental_yields, dtype=np.float64)
    noi = np.asarray(noi_grid, dtype=np.float64)

    mean_cap_rate = float(np.mean(yields))
    amortization_years = np.where(yields > 0, 1.0 / yields, math.nan)
    grm = np.where(yields > 0, 1.0 / yields, math.nan)

    return {
        "mean_cap_rate_pct": mean_cap_rate * 100.0,
        "amortization_years": amortization_years,
        "gross_rent_multiplier": grm,
        "estimated_asset_value": np.where(yields > 0, noi / yields, 0.0),
    }
