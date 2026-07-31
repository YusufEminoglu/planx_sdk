# -*- coding: utf-8 -*-
"""Cellular Automata (CA) Urban Growth & Land Use Cover Change (LUCC) Simulators."""

from __future__ import annotations

from typing import Any

import numpy as np


def sleuth_cellular_automata_growth(
    urban_grid: np.ndarray,
    slope_grid: np.ndarray,
    transport_distance_grid: np.ndarray,
    exclusion_grid: np.ndarray,
    diffusion_coef: float = 50.0,
    breed_coef: float = 50.0,
    spread_coef: float = 50.0,
    slope_coef: float = 50.0,
    road_coef: float = 50.0,
    steps: int = 10,
) -> dict[str, Any]:
    """Runs SLEUTH Cellular Automata Urban Sprawl & Growth Simulator.

    Models Spontaneous, Diffuse, Organic, and Road-Influenced urban expansion.

    Args:
        urban_grid: 2D boolean grid (True = urbanised).
        slope_grid: 2D slope grid (degrees).
        transport_distance_grid: 2D distance to transport network (meters).
        exclusion_grid: 2D boolean grid (True = excluded from development).
        diffusion_coef: Spontaneous growth coefficient [0, 100].
        breed_coef: New spreading center coefficient [0, 100].
        spread_coef: Organic growth coefficient [0, 100].
        slope_coef: Slope resistance coefficient [0, 100].
        road_coef: Road gravity coefficient [0, 100].
        steps: Number of simulation time steps.

    Returns:
        Dict containing projected urban grid, growth cell count per step, and urbanisation ratio.
    """
    grid = np.asarray(urban_grid, dtype=bool).copy()
    slope = np.asarray(slope_grid, dtype=np.float64)
    dist_road = np.asarray(transport_distance_grid, dtype=np.float64)
    excl = np.asarray(exclusion_grid, dtype=bool)

    h, w = grid.shape
    slope_resistance = np.exp(-slope * (slope_coef / 100.0))
    road_gravity = np.exp(-dist_road / max(100.0, 1000.0 * (1.0 - road_coef / 100.0)))
    prob = slope_resistance * road_gravity * (~excl)

    history = [int(np.sum(grid))]

    for _ in range(steps):
        new_urban = grid.copy()
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if grid[r, c] or excl[r, c]:
                    continue

                neighbors = int(np.sum(grid[r - 1 : r + 2, c - 1 : c + 2])) - int(grid[r, c])
                growth_p = prob[r, c] * (
                    0.2 * (diffusion_coef / 100.0)
                    + 0.3 * (breed_coef / 100.0 if neighbors == 1 else 0.0)
                    + 0.5 * (spread_coef / 100.0 if neighbors >= 2 else 0.0)
                )

                if growth_p > 0.35:
                    new_urban[r, c] = True

        grid = new_urban
        history.append(int(np.sum(grid)))

    return {
        "simulated_urban_grid": grid,
        "urban_cells_history": history,
        "final_urbanisation_ratio": float(np.mean(grid)),
        "new_urban_cells_count": history[-1] - history[0],
    }


def markov_transition_probability_matrix(
    raster_t1: np.ndarray,
    raster_t2: np.ndarray,
    num_classes: int = 5,
) -> dict[str, Any]:
    """Estimates Markov Land Cover Transition Probability Matrix.

    Args:
        raster_t1: 2D integer land cover array at time T1.
        raster_t2: 2D integer land cover array at time T2.
        num_classes: Number of discrete land cover classes.

    Returns:
        Dict containing transition probability matrix (K, K) and transition counts.
    """
    r1 = np.asarray(raster_t1, dtype=np.int32).ravel()
    r2 = np.asarray(raster_t2, dtype=np.int32).ravel()

    matrix = np.zeros((num_classes, num_classes), dtype=np.float64)

    for i in range(len(r1)):
        c1, c2 = r1[i], r2[i]
        if 0 <= c1 < num_classes and 0 <= c2 < num_classes:
            matrix[c1, c2] += 1.0

    sums = np.sum(matrix, axis=1, keepdims=True)
    prob_matrix = np.where(sums > 0, matrix / np.maximum(sums, 1e-12), 0.0)

    return {
        "transition_probability_matrix": prob_matrix,
        "transition_counts_matrix": matrix,
        "stable_cells_ratio": float(np.trace(matrix) / max(np.sum(matrix), 1.0)),
    }
