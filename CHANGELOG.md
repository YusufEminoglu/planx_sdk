# Changelog

All notable changes to the PlanX SDK project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.16.0] - 2026-07-31
### Added
- **Seismic Liquefaction Potential Index Engine (`planx.resilience.seismic`)**: `seismic_liquefaction_potential_index` (Iwasaki LPI method with SPT-N soil depth layers and Peak Ground Acceleration).

---

## [2.15.0] - 2026-07-31
### Added
- **Traffic Assignment & Mobility Engineering (`planx.mobility`)**: Frank-Wolfe Traffic Assignment User Equilibrium (UE), BPR link congestion performance function, Gravity Model OD demand matrix estimator, and Furness/Fratar OD matrix balancing.
- **Climate Adaptation & Ecosystem Services (`planx.climate`)**: Urban Canopy Annual CO2 Sequestration estimator and Green Roof Stormwater Retention Capacity & Peak Hydrograph Attenuation calculator.
- **3D Spatio-Temporal Kriging (`planx.geostats`)**: `spatio_temporal_kriging` 3D space-time interpolation engine (X, Y, Time).

---

## [2.14.0] - 2026-07-31
### Added
- **Spatial Real Estate & Land Value Capture Lab (`planx.realestate`)**: Spatial Hedonic Pricing models (SLM, SEM, OLS), Difference-in-Differences Land Value Capture (LVC) Uplift, Transit-Oriented Premium Index (TOPI), Automated Comparable Selection (Gower comps), and Capitalization Rate Spatial Interpolation.
- **Urban Morphology & Spacemate Density Lab (`planx.urban_morphology`)**: Spacemate Density Matrix (FSI, GSI, OSR, L) typology classification, Box-counting Fractal Dimension ($D$), and Block Porosity & Urban Grain Coarseness Index.
- **Cellular Automata Urban Sprawl & LUCC Growth Simulator (`planx.cellular_automata`)**: SLEUTH-like Cellular Automata urban growth engine and Markov Land Cover Transition Matrix estimator.
- **Urban Canopy & Microclimate Physics Lab (`planx.urban_physics`)**: Frontal Area Index ($\lambda_F$), Roughness Length ($z_0$), Displacement Height ($d_0$), and Cool Surface Albedo Uplift Thermal Cooling Potential.
- **Generative Urban Subdivision & Solar Rights Lab (`planx.generative`)**: Recursive OBB Parcel Layout Subdivision and 3D Buildable Envelope with Solar Rights Protection.

---

## [2.13.0] - 2026-07-31
### Added
- **Urban Resilience Accessibility Deficit & Intervention Priority Classifier** (`accessibility_deficit_class`, `composite_risk_class`, `intervention_priority_class`) in `planx.resilience.flood`.
- **Space Syntax Urban Integration & Choice Engine** (`calculate_space_syntax_integration`) in `planx.spatial.accessibility`.
- **Urban Sprawl & Land Use Entropy Balancer Engine** (`calculate_landuse_entropy_balance`) in `planx.spatial.accessibility`.
- **Solar Surface Radiation & Insolation Engine** (`calculate_solar_radiation_surface`) in `planx.resilience.heat`.

---

## [2.12.0] - 2026-07-31
### Added
- **Spatial Weights Auto-Band & Adjacency Builder** (`auto_distance_band`, `build_weights`, `neighbour_counts`) in `planx.geostats.weights`.
- **Urban Microform Procedural Shape Generator Engine** (`gen_dikdortgen`, `gen_L`, `gen_U`) in `planx.suitability.facility`.

---

## [2.11.0] - 2026-07-31
### Added
- **Spatio-Temporal Time-Series Forecasting Studio** (`exponential_smoothing`, `arima_forecast`, `random_forest_forecast`, `forecast_cell_series`, `forecast_metrics`, `backtest_series`) in `planx.geostats`.
- **Graph Network Edge Criticality & Robustness Engine** (`edge_criticality`) in `planx.spatial.accessibility`.
- **Connected Raster Candidate Site Extraction & Ranking Engine** (`label_components`, `rank_sites`) in `planx.suitability.facility`.
- **Spatial Autocorrelation & Residual Diagnostic Summary Engine** (`residual_spatial_autocorrelation_summary`, `regression_quality_summary`) in `planx.geostats`.

---

## [2.10.0] - 2026-07-31
### Added
- **Emerging Hot Spot Analysis & Spatio-Temporal $G_i^*$ Engine** (`getis_ord_g_star`, `getis_ord_gi_star_matrix`, `mann_kendall_test`, `classify_ehsa_pattern`) in `planx.geostats`.
- **Spatio-Temporal Time-Series Anomaly Detection Engine** (`robust_zscores`, `detect_anomalies`, `classify_trend`, `sen_intercept`) in `planx.geostats`.
- **Objective Criteria Weighting Suite (AHP, CRITIC, Shannon Entropy, PCA)** (`calculate_ahp_weights`, `ahp_weights_from_json`, `calculate_entropy_weights`, `calculate_critic_weights`, `calculate_pca_weights`) in `planx.suitability`.
- **Area-Weighted K-Means Clustering Engine for Land Readjustment** (`area_weighted_kmeans`) in `planx.suitability.facility`.

---

## [2.9.2] - 2026-07-31
### Fixed
- **Mypy Type Annotations**: Added explicit type annotations for `selected` micro-hub lists and `queue` deques to satisfy strict Mypy type-checking.

---

## [2.9.1] - 2026-07-31
### Fixed
- **CI Linting & Code Formatting**: Resolved Ruff E501 line-length and F821 typing import errors across all modules.

---

## [2.9.0] - 2026-07-31
### Added
- **Spatio-Temporal Panel Vector Autoregression** (`fit_spatial_pvar`) in `planx.geostats` for dynamic multi-variable spatio-temporal panel vector autoregression.
- **Urban Resilience Multi-Hazard Compound Risk Aggregator** (`compound_hazard_risk_aggregator`) in `planx.resilience.flood` for joint compound hazard risk modeling across flood, heat, and seismic hazards.
- **Urban Logistics Last-Mile Micro-Hub Location-Allocation Engine** (`logistics_microhub_location_allocation`) in `planx.suitability.facility` for cargo bike distribution micro-hub siting and VKT optimization.
- **Spherical Fuzzy TOPSIS Method** (`spherical_fuzzy_topsis`) in `planx.suitability.mcda` for multi-criteria evaluation under Spherical Fuzzy Sets.

---

## [2.8.0] - 2026-07-31
### Added
- **Spatial Panel Probit Model with Spatial Lag** (`fit_spatial_panel_probit_lag`) in `planx.geostats` for binary panel regression with spatial autoregressive lag and marginal effects.
- **Green Infrastructure Cooling Effect & Park Cool Island Simulator** (`green_infra_cooling_engine`) in `planx.resilience.heat` for Park Cool Island (PCI) temperature reduction decay modeling.
- **15-Minute City Multi-Modal Accessibility Equity Analyzer** (`fifteen_minute_city_equity_analyzer`) in `planx.spatial.accessibility` for cumulative 15m opportunity scoring across 6 essential urban service domains.
- **Hesitant Fuzzy DEMATEL Causal Mapping Engine** (`hesitant_fuzzy_dematel`) in `planx.suitability.mcda` for causal relationship matrix evaluation under hesitant fuzzy set environments.

---

## [2.7.0] - 2026-07-31
### Added
- **Spatial Panel Regime Regression** (`fit_spatial_panel_regimes`) in `planx.geostats` for structural break analysis and regime-specific panel coefficients.
- **Seismically Induced Landslide Susceptibility Engine** (`seismic_landslide_susceptibility_engine`) in `planx.resilience.flood` for infinite slope safety factor under pseudo-static PGA.
- **Paratransit & DRT Dispatch Optimizer** (`drt_dispatch_optimizer`) in `planx.spatial.accessibility` for dynamic demand-responsive transport dispatch and VKT reduction.
- **Picture Fuzzy TOPSIS MCDA Method** (`picture_fuzzy_topsis`) in `planx.suitability.mcda` for multi-criteria evaluation under positive, neutral, and negative membership degrees.

---

## [2.6.0] - 2026-07-31
### Added
- **Spatio-Temporal Geographically Weighted Ridge Regression** (`fit_st_gwrr`) in `planx.geostats` for localized spatio-temporal regression with L2 ridge regularization.
- **Urban Coastal Tsunami Inundation & Vertical Evacuation Router** (`tsunami_evacuation_routing_engine`) in `planx.resilience.flood` for tsunami wave attenuation and vertical refuge building allocation.
- **Urban Sky View Factor & Solar Envelope Profiler** (`canopy_sky_view_factor_profiler`) in `planx.spatial.accessibility` for 3D Sky View Factor and solar envelope shading analysis.
- **Fuzzy COPRAS Method** (`fuzzy_copras_method`) in `planx.suitability.mcda` for Complex Proportional Assessment using Triangular Fuzzy Numbers.

---

## [2.5.0] - 2026-07-31
### Added
- **Spatial Panel Error Components Model** (`fit_spatial_panel_sem`) in `planx.geostats` for spatio-temporal panel error autocorrelation.
- **Wildfire Urban Interface Ember Transport Simulator** (`wui_ember_transport_simulator`) in `planx.resilience.heat` for firebrand spotting distance distribution.
- **Public Transit Fleet Electrification Scheduler** (`transit_fleet_electrification_scheduler`) in `planx.spatial.accessibility` for bus fleet charging optimization subject to grid power caps.
- **Rough TOPSIS MCDA Method** (`rough_topsis_method`) in `planx.suitability.mcda` for decision modeling with lower and upper rough approximation boundary matrices.

---

## [2.4.0] - 2026-07-31
### Added
- **Spatio-Temporal Panel Tobit Spatial Lag Model** (`fit_spatial_panel_tobit_lag`) in `planx.geostats` for censored panel regression with spatial autoregressive lag.
- **Urban Heat Wave Health Vulnerability Engine** (`heatwave_health_vulnerability_engine`) in `planx.resilience.heat` for heat index calculation, vulnerable demographics, and AC coverage deficit scoring.
- **Transit-Oriented Development (TOD) Spatial Diversity Profiler** (`tod_spatial_diversity_index`) in `planx.suitability.facility` for Shannon land-use mix entropy, FAR intensity, and transit catchment evaluation.
- **Intuitionistic Fuzzy VIKOR Method** (`if_vikor_method`) in `planx.suitability.mcda` for multi-criteria compromise ranking under intuitionistic fuzzy preference weights.

---

## [2.3.0] - 2026-07-31
### Added
- **Spatial Panel Seemingly Unrelated Regression** (`fit_spatial_panel_sur`) in `planx.geostats` for multi-equation spatio-temporal systems with cross-equation error covariance and 2SLS spatial lag.
- **Urban Pluvial Flash Flood & Drainage Capacity Simulator** (`pluvial_flash_flood_simulator`) in `planx.resilience.flood` for storm runoff generation via SCS Curve Number and pipe capacity surcharge modeling.
- **First-Mile/Last-Mile Micro-Mobility Equity Index** (`micromobility_equity_index`) in `planx.spatial.accessibility` for dockless scooter/bike supply relative to transit hub proximity and socio-economic vulnerability.
- **Neutrosophic WASPAS MCDA Method** (`neutrosophic_waspas_method`) in `planx.suitability.mcda` for multi-criteria evaluation under single-valued neutrosophic truth, indeterminacy, and falsity degrees.

---

## [2.2.0] - 2026-07-29
### Added
- **Dynamic Spatial Panel GMM Estimator** (`fit_spatial_dynamic_panel_gmm`) in `planx.geostats` for Arellano-Bond / Blundell-Bond style dynamic spatial panel estimation ($y_{i,t} = \gamma y_{i,t-1} + \rho W y_{i,t} + X_{i,t} \beta + \mu_i + \varepsilon_{i,t}$) with first-difference transformation and 2SLS GMM instrumental variables.
- **Urban Wind Comfort & Canopy Aerodynamic Drag Simulator** (`wind_canopy_aerodynamic_drag_simulator`) in `planx.resilience.heat` evaluating canopy momentum absorption, wind speed attenuation profiles, and Lawson pedestrian wind comfort classifications (Categories 1-5).
- **Multi-Depot EV Capacitated Vehicle Routing Engine** (`ev_cvrp_multi_depot_routing`) in `planx.spatial.accessibility` for multi-depot EV routing with battery SOC depletion rates, vehicle load capacity constraints, and en-route fast charger visits.
- **Interval-Valued Intuitionistic Fuzzy TOPSIS Method** (`ivif_topsis_method`) in `planx.suitability.mcda` for multi-criteria decision evaluation under expert hesitation and interval-valued intuitionistic fuzzy sets $[(\mu_L, \mu_U), (\nu_L, \nu_U)]$.

---

## [2.1.0] - 2026-07-29
### Added
- **Spatial Panel Zero-Inflated Count Model** (`fit_spatial_zip_panel`) in `planx.geostats` for zero-inflated Poisson and Negative Binomial (ZIP / ZINB) panel regression using EM mixture modeling, weighted logit structural zero probabilities, and 2SLS spatial lag estimation.
- **Urban Surface Cool Island & Albedo Simulator** (`surface_cool_island_simulator`) in `planx.resilience.heat` evaluating Land Surface Temperature (LST) and PET outdoor thermal comfort mitigation from albedo modifications (cool roofs/pavements) and vegetation evapotranspiration.
- **EV Fleet Charging Station Location-Allocation Engine** (`ev_fleet_charging_location_allocation`) in `planx.suitability.facility` for multi-objective EV fleet charging depot siting, minimizing detour distances while honoring depot power capacity ($kW$) constraints.
- **Multi-Modal Transit Isochrone Profiler** (`multimodal_transit_isochrone_profiler`) in `planx.spatial.accessibility` calculating multi-modal travel times, walking access/egress speeds, initial headway waiting times, transit in-vehicle travel, and transfer friction penalties.

---

## [2.0.0] - 2026-07-29
### Added
- **Spatial Panel Count Regression Engine** (`fit_spatial_count_panel`) in `planx.geostats` for spatio-temporal Poisson and Negative Binomial count data modeling with 2SLS spatial lag instrumental variables, IRLS log-link GLM estimation, and dispersion alpha calculation.
- **Coastal Storm Surge & Sea Level Rise Inundation Engine** (`coastal_storm_surge_inundation_engine`) in `planx.resilience.flood` simulating hydrologic connectivity flood filling over DEM grids with sea level rise projections, Manning surface friction headloss penalties, and hazard severity classification.
- **EV Charging Station Spatial Accessibility & Grid Stress Index** (`ev_charging_accessibility_index`) in `planx.spatial.accessibility` combining 2SFCA accessibility decay to Level 2 AC and Level 3 DC Fast chargers with station power capacities (kW), transformer grid constraints, and Gini equity decomposition.
- **SPOTIS MCDA Method** (`spotis_method`) in `planx.suitability.mcda` for Stable Preference Ordering Towards Ideal Solution multi-criteria evaluation with criteria domain bounds $[S_{j,min}, S_{j,max}]$, preventing rank reversal.

---

## [1.9.0] - 2026-07-29
### Added
- **Spatial Panel Quantile Regression Engine** (`fit_spatial_quantile_panel`) in `planx.geostats` for quantile-specific spatio-temporal panel data regression using 2SLS instrumental variable stage 1 and HiGHS linear programming pinball loss minimization stage 2.
- **Urban Energy Vulnerability & Fuel Poverty Index** (`urban_energy_vulnerability_index`) in `planx.resilience.social` evaluating neighborhood fuel poverty risks, building efficiency burdens, income inelasticity, and climate temperature exposure.
- **Transit-Oriented Development (TOD) Node Evaluation Engine** (`evaluate_tod_node_suitability`) in `planx.suitability.facility` for multi-criteria 5D TOD node assessment across density, diversity, design, transit frequency, and parking supply penalty factors.
- **Microclimate Pedestrian Heat Route Optimizer** (`microclimate_pedestrian_route_optimizer`) in `planx.spatial.walkability` evaluating heatwave-resilient pedestrian paths using tree canopy shade ratios, Land Surface Temperature (LST) discomfort multipliers, and Dijkstra shortest path routing.

---

## [1.8.0] - 2026-07-29
### Added
- **Spatial Panel Probit Model** (`fit_spatial_probit_panel`) in `planx.geostats` for binary outcome (0/1) spatio-temporal panel data regression using Probit GLM latent variable approximation and 2SLS Spatial Lag parameter estimation.
- **Parking Spatial Mismatch Index** (`parking_spatial_mismatch_index`) in `planx.spatial.accessibility` evaluating urban parking supply-demand spatial mismatch ratios, walking threshold distance decay, and deficit/surplus zone Gini equity.
- **Wildfire Evacuation Front & Buffer Simulator** (`wildfire_evacuation_front_buffer`) in `planx.resilience.wildfire` using Rothermel Rate of Spread (ROS) formulas with wind and slope multipliers for dynamic fire front expansion and safety perimeter buffers.
- **Structural Earthquake Debris & Road Blockage Simulator** (`seismic_road_blockage_simulation`) in `planx.resilience.seismic` modeling building collapse debris projection onto street right-of-ways and street blockage probabilities for disaster response corridors.

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
