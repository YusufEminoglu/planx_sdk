# Changelog

All notable changes to the PlanX SDK project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.7.0] - 2026-07-29
### Added
- **Spatial Panel SARMA Model** (`fit_spatial_sarma_panel`) in `planx.geostats` for joint Spatial Lag + Spatial Error panel data regression using 2SLS stage 1 lag parameter estimation and GS2SLS stage 2 error transformation.
- **Stormwater Retention Basin Design Engine** (`stormwater_retention_basin_design`) in `planx.resilience.flood` for green infrastructure storage volume, infiltration surface area, and drain-down time engineering based on SCS Curve Number and soil K_sat.
- **Low-Stress Bicycle Network Connectivity Profiler** (`bike_network_low_stress_connectivity`) in `planx.spatial.walkability` evaluating bicycle Level of Traffic Stress (LTS 1-4) low-stress island connected components and permeability ratios.
- **Spatial Huff Gravity Retail Market Share Model** (`huff_retail_market_share`) in `planx.spatial.accessibility` for multi-store attraction, distance exponent decay, customer capture, and primary trade area market share projections.

---

## [1.6.0] - 2026-07-29
### Added
- **Spatial Panel Tobit Model** (`fit_spatial_tobit_panel`) in `planx.geostats` for left/zero-censored spatio-temporal panel data regression using 2SLS Spatial Lag initial parameters and truncated normal Mills ratio expected value adjustments.
- **Urban Tree Canopy Placement Optimizer** (`optimize_tree_canopy_greening`) in `planx.resilience.heat` for multi-objective greedy tree planting site selection maximizing LST cooling, air quality mitigation, and pedestrian exposure with Gaussian spatial cooling decay.
- **Multi-Modal 15-Minute City Score & Equity Engine** (`calculate_multimodal_15m_city`) in `planx.spatial.accessibility` evaluating walking, cycling, and public transit multi-modal travel time scores across essential urban service categories with Gini equity decomposition.
- **3D Building Solar Envelope & Shadow Analysis Engine** (`calculate_building_solar_envelope`) in `planx.spatial.centrality` for 3D building mass shadow length/projection and ground/roof solar access envelope calculations given solar altitude and azimuth angles.

---

## [1.5.0] - 2026-07-29
### Added
- **Spatial Panel Regression Engine** (`fit_spatial_panel_model`) in `planx.geostats` for spatio-temporal panel econometrics supporting 2SLS Spatial Panel Lag and FGLS Spatial Panel Error models.
- **Healthcare Equity Index** (`healthcare_equity_index`) in `planx.spatial.accessibility` combining E2SFCA 3-step decay accessibility with demographic group vulnerability weights and spatial Gini equity decomposition.
- **Dynamic Evacuation Traffic Bottleneck Simulator** (`dynamic_evacuation_bottlenecks`) in `planx.resilience.evacuation` for dynamic traffic assignment with BPR congestion delays, spatial queue tracking, and critical bottleneck edge identification.
- **Structural Seismic Damage & Economic Loss Curve** (`seismic_damage_loss_curve`) in `planx.resilience.seismic` for Hazus-compatible lognormal fragility damage state probability curves and building monetary loss estimations across Peak Ground Acceleration (PGA) spectrums.

---

## [1.4.0] - 2026-07-29
### Added
- **Space-Time Cube Generator** (`create_space_time_cube`) in `planx.geostats` for 3D spatial (x, y) and temporal (t) bin aggregation supporting mean, sum, count, min, max, and std metrics.
- **Fuzzy VIKOR MCDA Method** (`fuzzy_vikor_method`) in `planx.suitability.mcda` for Triangular Fuzzy Number (TFN) compromise ranking with defuzzification, S/R group utility/regret, Q index, and C1/C2 stability condition testing.
- **Network-Based Voronoi Service Area** (`network_voronoi_allocation`) in `planx.spatial.accessibility` for multi-source graph-based shortest path network Voronoi allocation with demand coverage ratio calculation.
- **Compound Multi-Hazard Cascade Simulation** (`compound_hazard_cascade`) in `planx.resilience.synthesis` for iterative cascading multi-hazard triggering and amplification modeling with saturating damage indices.

---

## [1.3.0] - 2026-07-29
### Added
- **Emerging Hot Spot Analysis** (`emerging_hotspot_analysis`) in `planx.geostats` for spatio-temporal trend detection combining Getis-Ord Gi* hot spot z-scores with Mann-Kendall trend testing to classify 17 space-time patterns (new, consecutive, intensifying, persistent, diminishing, sporadic, oscillating, historical hot/cold spots).
- **Fuzzy TOPSIS MCDA Method** (`fuzzy_topsis_method`) in `planx.suitability.mcda` for Triangular Fuzzy Number (TFN) extension of TOPSIS handling linguistic/uncertain evaluations with fuzzy normalization, weighted vertex distance to FPIS/FNIS, and closeness coefficient ranking.
- **Public Transit Frequency Accessibility Index** (`transit_frequency_accessibility`) in `planx.spatial.accessibility` for gravity-model transit accessibility combining stop proximity distance decay (Gaussian/exponential/linear), service headway frequency scoring, and route diversity weighting.
- **SCS Unit Hydrograph** (`scs_unit_hydrograph`) in `planx.resilience.flood` for NRCS dimensionless unit hydrograph storm runoff routing with SCS Curve Number excess rainfall, triangular peak discharge, and time-interpolated discharge hydrograph generation.

---

## [1.2.0] - 2026-07-28
### Added
- **Geographically Weighted PCA (GWPCA)** (`calculate_gwpca`) in `planx.geostats` for spatially varying Principal Components Analysis with local eigenvalue decomposition and kernel-weighted covariance matrices.
- **EDAS MCDA Method** (`edas_method`) in `planx.suitability.mcda` for Evaluation Based on Distance from Average Solution ranking with Positive/Negative Distance Appraisal Scores.
- **Pedestrian Level of Service (PLOS)** (`pedestrian_level_of_service`) in `planx.spatial.walkability` for HCM-based sidewalk quality grading (LOS A-F) from width, flow, and traffic stress factors.
- **Urban Flood Detention Basin Sizing** (`detention_basin_sizing`) in `planx.resilience.flood` for SCS Curve Number pre-/post-development runoff-based detention storage volume and peak inflow estimation.

---

## [1.1.0] - 2026-07-27
### Added
- **SARMA Spatial Model (Spatial Autoregressive Moving Average)** (`fit_spatial_sarma_model`) in `planx.geostats` for 2SLS joint Spatial Lag + Spatial Error regression modeling.
- **COPRAS MCDA Method** (`copras_method`) in `planx.suitability.mcda` for Complex Proportional Assessment under benefit and cost criteria.
- **3D Building Solar Radiation Potential** (`calculate_building_solar_radiation`) in `planx.resilience.heat` for Sky View Factor (SVF) weighted rooftop solar irradiance and PV electricity generation (kWh/yr).
- **Urban Heat Vulnerability Index (HVI)** (`urban_heat_vulnerability_index`) in `planx.resilience.heat` for multi-component heat risk index synthesizing exposure, sensitivity, and adaptive capacity.

---

## [1.0.0] - 2026-07-27
### Added
- **Spatially Constrained Regionalization (SKATER)** (`skater_spatial_clustering`) in `planx.geostats` for Minimum Spanning Tree (MST) edge-cutting spatially contiguous regional clustering.
- **ARAS MCDA Method** (`aras_method`) in `planx.suitability.mcda` for Additive Ratio Assessment with optimal reference alternative $S_0$ and utility degree $K_i$.
- **Street Network Rose Spectrum & Directional Anisotropy Index** (`street_orientation_rose_spectrum`) in `planx.spatial.walkability` for polar angle binning distribution and directional anisotropy index.
- **Urban Stormwater Peak Runoff Engine** (`urban_stormwater_peak_runoff`) in `planx.resilience.flood` for Rational Method peak discharge $Q = 0.00278 \cdot C \cdot I \cdot A$ ($m^3/s$) and runoff volume modeling.

---

## [0.9.0] - 2026-07-27
### Added
- **Weighted 2D Kernel Density Estimation (Weighted KDE)** (`calculate_weighted_kde`) in `planx.geostats` for magnitude-weighted 2D spatial event density mapping.
- **Ripley's Cross-K Function** (`calculate_ripleys_cross_k`) in `planx.geostats` for multi-type point pattern co-location attraction/repulsion testing.
- **WASPAS MCDA Method** (`waspas_method`) in `planx.suitability.mcda` for Weighted Aggregated Sum Product Assessment combining WSM and WPM models.
- **DEMATEL Causal Matrix Analysis** (`dematel_method`) in `planx.suitability.weights` for Decision Making Trial and Evaluation Laboratory cause/effect prominence analysis.
- **Cul-de-sac Isolation Index** (`cul_de_sac_isolation_index`) in `planx.spatial.walkability` for dead-end node identification and network isolation ratio.
- **Earthquake Building Collapse & Casualty Model** (`earthquake_building_collapse_casualty`) in `planx.resilience.seismic` for structural fragility lognormal CDF collapse probability and casualty estimates.

---

## [0.8.0] - 2026-07-27
### Added
- **Geographically Weighted Summary Statistics (GWSS)** (`calculate_gwss`) in `planx.geostats` for local weighted mean, standard deviation, and skewness statistics.
- **MARCOS MCDA Method** (`marcos_method`) in `planx.suitability.mcda` for Measurement of Alternatives and Ranking according to COmpromise Solution.
- **FUCOM MCDA Weights** (`fucom_weights`) in `planx.suitability.weights` for Full Consistency Method linear programming criterion weighting.
- **Lawson Pedestrian Wind Comfort Index** (`calculate_wind_comfort_lawson`) in `planx.spatial.walkability` for street canyon wind speed amplification and comfort classification.
- **Wildfire Evacuation Encroachment Engine** (`wildfire_evacuation_encroachment`) in `planx.resilience.wildfire` for Rothermel-style fire front propagation velocity and dynamic safe evacuation buffer zones.

---

## [0.7.0] - 2026-07-27
### Added
- **Spatial Tobit Regression (SAR-Tobit)** (`fit_spatial_tobit_model`) in `planx.geostats` for 2SLS zero-censored spatial autoregressive modeling.
- **Local Moran's I with Benjamini-Hochberg FDR** (`calculate_local_moran_fdr`) in `planx.geostats` for False Discovery Rate multiple testing control under spatial autocorrelation.
- **Fuzzy AHP MCDA Weights** (`fuzzy_ahp_weights`) in `planx.suitability.weights` using Triangular Fuzzy Numbers $(l, m, u)$ and Chang's extent analysis.
- **Spatial MCDA Monte Carlo Sensitivity Engine** (`mcda_sensitivity_monte_carlo`) in `planx.suitability.mcda` evaluating alternative rank stability under criteria weight noise.
- **Space Syntax Axial-to-Segment Line Conversion** (`axial_to_segment_conversion`) in `planx.spatial.centrality` for splitting continuous axial lines at all topological intersections.
- **Tree Canopy Microclimate Cooling Model** (`tree_canopy_microclimate_cooling`) in `planx.resilience.heat` calculating air temperature reduction $\Delta T_{cool} (^\circ C)$ based on LAI and canopy distance decay.

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
