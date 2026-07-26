# Changelog

All notable changes to the PlanX SDK project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.6.0] - 2026-07-27
### Added
- **Geographically Weighted Logistic Regression (GWLR)** (`calculate_gwlr`) in `planx.geostats` for spatially varying binary outcome models using Iteratively Reweighted Least Squares (IRLS).
- **Best-Worst Method (BWM)** (`bwm_weights`) in `planx.suitability.weights` for min-max linear programming MCDA criterion weight optimization.
- **Pareto Multi-Objective Facility Location** (`pareto_facility_location`) in `planx.suitability.facility` evaluating trade-offs between population coverage, travel distance, and Gini equity.
- **Interdependent Infrastructure Cascading Failure Simulation** (`simulate_interdependent_infrastructure_cascade`) in `planx.resilience.infrastructure` for coupled power-water failure propagation.
- **Street Network Morphometry & Grain Profiler** (`street_network_morphometry`) in `planx.spatial.walkability` computing nodal degree distribution, link lengths, planar meshedness $\alpha$, and connectivity.

---

## [0.5.0] - 2026-07-27
### Added
- **Spatial Econometrics Engine** (`fit_spatial_lag_model`, `fit_spatial_error_model`) in `planx.geostats` for 2SLS Spatial Autoregressive (SLM/SAR) and Generalized Cochrane-Orcutt Spatial Error (SEM) regression modeling.
- **Outranking MCDA Methods** (`electre_i_method`, `electre_iii_method`) in `planx.suitability.mcda` supporting concordance/discordance matrices and pseudo-criteria threshold outranking relations.
- **Continuous Kernel Facility Location** (`mclp_distance_decay`) in `planx.suitability.facility` for exponential, Gaussian, and linear distance-attenuated maximal coverage optimization.
- **Disaster Evacuation Route Optimization** (`evacuation_route_optimization`) in `planx.resilience.evacuation` for capacity-constrained network routing, clearance time estimation, and bottleneck identification.
- **Coastal Storm Surge Inundation Model** (`coastal_surge_inundation`) in `planx.resilience.flood` using 8-neighbor hydrologic connectivity flood-fill with coastal distance decay.
- **Space Syntax Angular Segment Centrality** (`angular_segment_centrality`) in `planx.spatial.centrality` for turn-angle deflection shortest paths, NAIn, and NACh normalization metrics.

---

## [0.4.1] - 2026-07-27
### Added
- **PROMETHEE II Method** (`promethee_ii_method`) in `planx.suitability.mcda` for outranking-based MCDA decision analysis.
- **Bivariate Global Moran's I** (`calculate_bivariate_moran`) and **Bivariate Local Moran's I** (`calculate_local_bivariate_moran`) in `planx.geostats.stats_engines` for multi-variable spatial autocorrelation diagnostics.
- **Spatial Lag Calculator** (`calculate_spatial_lag`) helper function in `planx.geostats.stats_engines`.

---

## [0.4.0] - 2026-07-20
### Changed
- feat: harden engine validation and release quality
- feat: add 3D Solar Envelope Analyst model

---

## [0.3.3] - 2026-07-12
### Changed
- feat: add Sky View Factor (SVF) and Local Climate Zone (LCZ) models
- feat: add Intersection Density & Closeness Profiler model
- feat: add Average Route Circuity model
- feat: add PageRank Centrality and Street Orientation Entropy models
- fix: resolve lint and mypy type checking issues and add 15m city tests
- feat: add Pedestrian Route Directness (PRD) and Equity-Weighted Accessibility models
- feat: add Walk Score calculator and Transit-Oriented Development (TOD) Index models
- feat: add 3SFCA accessibility model and optimal tree canopy greening locator
- refactor: move decay validation to top of choice_centrality_una
- test: add edge cases and validations for comfort routing and choice centrality decay methods
- test: add edge cases and validations for Getis-Ord, central feature, and Lee's L in stats engines
- test: expand coverage tests for spatial weights and IDW/Kriging interpolation
- test: add edge-case and parameter validation tests for gravity accessibility models
- feat: implement advanced active mobility, walkability comfort, and spatial mismatch analytics
- feat: port QGIS plugin embedded analytics engine and integrate test suite
- test: add test coverage for facility location allocation and resilience models
- feat: implement Local Geary's C spatial statistic and tests

---

## [0.3.2] - 2026-07-02
### Added
- **Local Geary's C** in `planx.geostats` for Anselin-style local cluster and spatial outlier detection, using conditional permutation inference to complement the existing Local Moran's I and Global Geary's C statistics.

---

## [0.3.0] - 2026-06-16
### Added
- **Huff Gravity Market Model** in `planx.spatial` to compute Choice probabilities for retail and service attraction.
- **Kernel Density 2SFCA (KD2SFCA)** in `planx.spatial` supporting continuous kernels (Quartic, Gaussian, Epanechnikov).
- **Global Geary's C** in `planx.geostats` for measuring spatial autocorrelation and local differences.
- **Ordinary Kriging Interpolation** in `planx.geostats` supporting Spherical, Exponential, Gaussian, and Linear semivariograms.
- **TOPSIS Method** in `planx.suitability` for ideal-worst distance-based MCDA ranking.
- **VIKOR Method** in `planx.suitability` for conflicting criteria compromise MCDA ranking.
- **Network Criticality Index (NCI)** in `planx.resilience` to evaluate node and link vulnerability under disruption.
- **Urban Heat Island (UHI) Intensity proxy** in `planx.resilience` based on albedo, NDVI, building footprint/height, and wind.
- **Socio-Economic Flood Risk Index** in `planx.resilience` combining hazard depth, building exposure, and SVI.
- **Debris Clearance Routing (Greedy TSP)** in `planx.resilience` to optimize road clearing sequences from a depot.

---

## [0.2.0] - 2026-06-16
### Changed
- Add a manual release-prep workflow
- Stop tracking .coverage and ignore coverage artifacts
- Add coverage reporting, stricter lint, pre-commit, and CI badge
- Ship inline types (py.typed) and add Dependabot for Actions
- Add CI workflow and make the codebase type-clean
- Harden PyPI publish workflow and migrate actions to Node 24

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
