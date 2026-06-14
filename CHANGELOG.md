# Changelog

All notable changes to the PlanX SDK project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0a0] - 2026-06-15
### Added
- Pre-release of the 0.2.0 alpha version for integration testing.

---

## [0.1.20] - 2026-06-15
### Added
- `create_knn_weights` and `create_distance_band_weights` under `planx.geostats.weights` to generate spatial neighbors and weights matrices using SciPy `cKDTree`.

---

## [0.1.19] - 2026-06-15
### Added
- `wildfire_risk_index` model in `planx.resilience.wildfire` combining terrain slope, direction aspect, and vegetation density factors.
- `_calculate_terrain_factors` helper to calculate slope and aspect using Horn's method.

---

## [0.1.18] - 2026-06-15
### Added
- `capacitated_location_allocation` in `planx.suitability.facility` to assign demand points to closest facilities under capacity limits.

---

## [0.1.17] - 2026-06-15
### Added
- `idw_to_points` and `idw_to_grid` under `planx.geostats.interpolation` for Inverse Distance Weighting spatial interpolation using fast `cKDTree`.

---

## [0.1.16] - 2026-06-15
### Added
- `landslide_susceptibility` model under `planx.resilience.landslide` using Horn's 8-neighbor slope calculation, soil stability, and LULC factor weights.

---

## [0.1.15] - 2026-06-15
### Changed
- Translated the entire `README.md` documentation and code examples to English.

---

## [0.1.14] - 2026-06-15
### Added
- `coastal_flood_inundation` connected bathtub model using `scipy.ndimage.label` 8-connectivity.
- Documented coastal flood inundation and pluvial flood susceptibility.

---

## [0.1.9] - [0.1.13]
### Added
- `planx.resilience.infrastructure` containing network disruption, service loss, bottlenecks, and debris clearance priorities.
- `greedy_p_median` and `greedy_lscp` under `planx.suitability`.
- `enhanced_2sfca`, `spatial_equity_gini`, and `service_area_coverage` under `planx.spatial`.
