# PlanX SDK (Software Development Kit)

[![PyPI version](https://img.shields.io/pypi/v/planx-sdk.svg)](https://pypi.org/project/planx-sdk/)
[![Python version support](https://img.shields.io/pypi/pyversions/planx-sdk.svg)](https://pypi.org/project/planx-sdk/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GitHub Code Formatting](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

PlanX SDK is the official core library of the PlanX QGIS plugin ecosystem, containing spatial analysis, spatial statistics, network routing, and urban resilience computation engines. It is designed to run headless, independent of the QGIS interface, and is powered entirely by pure Python, NumPy, and SciPy.

With this SDK, you can:
1. **Run Headless & Independent:** Execute complex spatial algorithms without launching QGIS (e.g., in Jupyter Notebooks, standalone scripts, servers, or web applications).
2. **Perform Fast Testing:** Easily run unit tests via `pytest` in a headless environment.
3. **Ensure Centralized Management:** Keep all plugins and analytical tools consistent by updating a single core package (installable via `pip install planx-sdk`).

---

## 📂 Project Structure (Directory Structure)

```text
planx_sdk/
  ├── .github/workflows/          # GitHub Actions (Automated PyPI publishing)
  ├── pyproject.toml              # Modern package metadata, dependencies, and settings (PEP 517/621)
  ├── README.md                   # Project documentation
  ├── LICENSE                     # License file
  ├── .gitignore                  # Git ignore rules for caches and builds
  ├── src/                        # Source code (Standard src-layout)
  │   └── planx/                  # Main package directory
  │       ├── __init__.py         # Versioning and package level definitions
  │       ├── spatial/            # Core spatial network analysis and space syntax engine
  │       │   ├── __init__.py
  │       │   ├── paths.py        # Shortest path algorithms (Dijkstra, SciPy integration)
  │       │   ├── centrality.py   # Closeness, Betweenness, Eigenvector, and Criticality centrality
  │       │   └── accessibility.py# Hansen Gravity and Cumulative Opportunities accessibility models
  │       ├── geostats/           # Spatial statistics and spatial autocorrelation engines
  │       │   ├── __init__.py
  │       │   ├── stats_engines.py# Getis-Ord Gi*, Local/Global Moran's I, OLS, GWR, SDE, k-means
  │       │   └── interpolation.py# Spatial interpolation algorithms (e.g. IDW, nearest neighbor)
  │       ├── suitability/        # Raster-based MCDA (Multi-Criteria Decision Analysis) engine
  │       │   ├── __init__.py
  │       │   ├── mcda.py         # Normalization methods (Sigmoid, Gaussian, Min-Max) and WLC
  │       │   ├── facility.py     # Facility location optimization (MCLP, p-Median, LSCP)
  │       │   └── weights.py      # Decision matrix weighting methods (AHP, Entropy, CRITIC, PCA)
  │       └── resilience/         # Urban resilience, disaster risk, and hazard simulation engines
  │           ├── __init__.py
  │           ├── seismic.py      # Monte Carlo seismic structural damage and debris propagation simulation
  │           ├── flood.py        # DEM-based pluvial (surface water) and connected coastal flood models
  │           ├── landslide.py    # Terrain slope (Horn's method), soil and LULC landslide screening
  │           ├── wildfire.py     # Terrain slope/aspect and fuel-based wildfire risk index
  │           ├── social.py       # Social Vulnerability Index (SVI) screening and analysis
  │           ├── heat.py         # Urban heat comfort risk and green space deficit screening model
  │           ├── synthesis.py    # Multi-hazard composite index and equity-oriented priority synthesis
  │           └── infrastructure.py# Infrastructure disruption, service loss, and bottleneck analysis
  └── tests/                      # Unit tests
```

---

## 💡 Core Features and Usage Examples

### 1. Urban Accessibility Analysis (`planx.spatial`)
Calculates accessibility from origin points (e.g., residential areas) to destination points (e.g., hospitals) using travel distance matrices, gravity (attraction) weights, or cumulative opportunities.

```python
import numpy as np
from planx.spatial import gravity_accessibility

# Distance Matrix (O x D): Distance from 2 origins to 3 destinations (in meters)
dists = np.array([
    [150.0, 300.0, 900.0],
    [500.0, 100.0, 1200.0]
])
# Destination capacity/attraction weights (e.g., hospital bed capacity)
weights = np.array([50.0, 100.0, 250.0])

# Gravity accessibility using exponential decay function (Hansen Index)
accessibility = gravity_accessibility(
    dists, weights, decay_method="exponential", beta=0.002, cutoff=1000.0
)
print("Accessibility Scores:", accessibility)

# 2. E2SFCA (Enhanced Two-Step Floating Catchment Area) and Gini Inequality Index
from planx.spatial import enhanced_2sfca, spatial_equity_gini

d_matrix = np.array([[10.0, 50.0], [30.0, 10.0]])
hospitals_supply = np.array([10.0, 20.0])  # Bed capacities
neighborhoods_demand = np.array([100.0, 200.0])  # Population

# E2SFCA scores with 40.0 min/km cutoff and linear decay method
access_scores = enhanced_2sfca(
    d_matrix, hospitals_supply, neighborhoods_demand, cutoff=40.0, decay_method="linear"
)
print("E2SFCA Accessibility Scores:", access_scores)  # [0.06, 0.12]

# Spatial accessibility equity analysis using population-weighted Gini coefficient
gini_coeff = spatial_equity_gini(access_scores, neighborhoods_demand)
print("Accessibility Gini Coefficient:", gini_coeff)

# 3. Service Area and Population Coverage Analysis (Isochrone / Service Area)
from planx.spatial import service_area_coverage

# Simple road network representation (CSR format, 3 nodes)
ind_ptr = np.array([0, 1, 3, 4], dtype=np.int64)
adj_nodes = np.array([1, 0, 2, 1], dtype=np.int64)
edge_w = np.array([1.5, 1.5, 2.5, 2.5], dtype=np.float64)  # Travel time in minutes

node_pop = np.array([100.0, 200.0, 300.0])  # Node population
facilities = np.array([0])                  # Index of facility node
thresholds = [1.0, 2.0, 5.0]                # Service time thresholds (minutes)

service_areas = service_area_coverage(
    ind_ptr, adj_nodes, edge_w, n=3, facilities=facilities,
    thresholds=thresholds, node_population=node_pop
)
print("Reachable Nodes at 2.0 Minutes:", service_areas[2.0]["reachable_nodes"])  # [0, 1]
print("Total Covered Population at 2.0 Minutes:", service_areas[2.0]["population_covered"])  # 300.0
```

### 2. Facility Location Optimization (`planx.suitability`)
Optimizes the placement of facilities (e.g., emergency assembly areas, shelters) to maximize coverage (**MCLP**, **LSCP**) or minimize average travel distance (**p-Median**).

```python
import numpy as np
from planx.suitability import greedy_mclp, greedy_p_median, greedy_lscp

candidates = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])  # Candidate shelter coordinates
demands = np.array([[1.0, 1.0], [11.0, 11.0], [25.0, 25.0]])     # Building demand coordinates
populations = np.array([100.0, 250.0, 500.0])                    # Demand population

# 1. MCLP: Select K=2 shelters to maximize coverage within 15.0 units walking distance
selected_mclp, added_pop, cum_pop = greedy_mclp(
    candidates, demands, populations, max_distance=15.0, k=2
)
print("MCLP Selected Facility Indices:", selected_mclp)  # [2, 1]

# 2. p-Median: Select P=2 facilities to minimize total weighted distance cost
selected_pmed, costs = greedy_p_median(
    candidate_coords=candidates, demand_coords=demands, demand_pop=populations, p=2
)
print("p-Median Selected Facility Indices:", selected_pmed)  # [2, 1]
print("Cumulative Step Costs:", costs)

# 3. LSCP: Find min facilities to cover at least 80% of population within 15.0 units distance
selected_lscp, final_coverage = greedy_lscp(
    candidates, demands, demand_pop=populations, max_distance=15.0, target_coverage=0.8
)
print("LSCP Selected Facility Indices:", selected_lscp)  # [2] (One facility covers 87.5%)

# 4. Capacitated Location Allocation: Assign demands to closest facilities with capacity limits
from planx.suitability import capacitated_location_allocation

facilities = np.array([[0.0, 0.0], [10.0, 0.0]])
capacities = np.array([150.0, 200.0])
demands_coords = np.array([[1.0, 0.0], [9.0, 0.0], [2.0, 0.0]])
demands_pop = np.array([100.0, 150.0, 80.0])

allocations, unassigned, usage = capacitated_location_allocation(
    facilities, capacities, demands_coords, demands_pop
)
print("Allocations per facility:", allocations)
print("Unassigned demands:", unassigned)

```

### 3. Seismic Damage and Debris Propagation (`planx.resilience`)
Runs a stochastic Monte Carlo simulation to evaluate building collapse probabilities based on building attributes (construction year, number of floors) and earthquake magnitude. Computes debris volume and outward collapse radius.

```python
import numpy as np
from planx.resilience import simulate_seismic_debris

areas = np.array([120.0, 200.0, 80.0])  # Building footprint areas (m2)
floors = np.array([4.0, 8.0, 2.0])      # Number of floors
years = np.array([1990, 2005, 2021])     # Construction years

# Earthquake scenario (7.2 Mw)
probs, collapsed, radii, volumes = simulate_seismic_debris(
    areas, floors, years, magnitude=7.2, seed=42
)
print("Building Collapse State (0: Intact, 1: Collapsed):", collapsed)
print("Debris Radii (m):", radii)
```

### 4. Social Vulnerability Index (SVI) (`planx.resilience`)
Normalizes and aggregates demographic indicators (e.g., elderly share, low-income population, disability rate) using Min-Max scaling and Weighted Linear Combination to produce vulnerability ranks.

```python
import numpy as np
from planx.resilience import social_vulnerability_index

# Demographic data for 3 neighborhoods
indicators = {
    "elderly": np.array([10.0, 50.0, 100.0]),
    "low_income": np.array([200.0, 100.0, 50.0])
}
# Indicator weights
weights = {
    "elderly": 0.5,
    "low_income": 0.5
}

scores, classes = social_vulnerability_index(indicators, weights)
print("Social Vulnerability Scores:", scores)
print("Vulnerability Classes:", classes)  # ['Moderate', 'Moderate', 'Moderate']
```

### 5. Urban Heat Comfort Risk (`planx.resilience`)
Calculates urban heat exposure/risk scores (0-100) by combining impervious surface share, building density, green space deficit, walking distance to cooling areas, and vulnerable population hotspots.

```python
import numpy as np
from planx.resilience import urban_heat_comfort_risk

# Data for a 2x2 grid
impervious = np.array([[0.8, 0.2], [0.5, 0.1]])          # Impervious share [0-1]
buildings = np.array([[0.6, 0.1], [0.4, 0.05]])          # Building footprint share [0-1]
green = np.array([[0.1, 0.8], [0.3, 0.9]])               # Green cover share [0-1]
cooling_dists = np.array([[300.0, 50.0], [200.0, 20.0]])  # Distance to cooling area (meters)
vuln_assets = np.array([[2, 0], [1, 0]])                  # Vulnerable assets count (e.g., schools)

scores, classes = urban_heat_comfort_risk(
    impervious, buildings, green, cooling_dists, vuln_assets, cooling_distance=400.0
)
print("Heat Comfort Risk Scores:\n", scores)
```

### 6. Decision Making and Weighting Methods (`planx.suitability`)
Provides analytic tools for Multi-Criteria Decision Analysis (MCDA) weighting, including AHP (Analytic Hierarchy Process), Entropy, CRITIC, and PCA. Includes helper functions to extract pairwise parameters.

```python
import numpy as np
from planx.suitability import ahp_weights, entropy_weights

# 1. AHP (Analytic Hierarchy Process) with consistency ratio checking
# 3x3 pairwise comparison matrix
matrix = np.array([
    [1.0, 2.0, 3.0],
    [0.5, 1.0, 2.0],
    [0.33, 0.5, 1.0]
])
weights, cr = ahp_weights(matrix)
print("AHP Weights:", weights)
print("Consistency Ratio (CR):", cr)

# 2. Entropy Weighting Method
# Decision matrix with 4 alternatives (rows) and 3 criteria (columns)
decision_matrix = np.array([
    [10.0, 100.0, 0.1],
    [20.0, 50.0, 0.2],
    [15.0, 80.0, 0.15],
    [30.0, 20.0, 0.3]
])
ent_weights = entropy_weights(decision_matrix)
print("Entropy Weights:", ent_weights)
```

### 7. Multi-Hazard Synthesis and Equitable Prioritization (`planx.resilience`)
Synthesizes multiple hazard exposure layers into a single Composite Hazard Index. Integrates the Social Vulnerability Index (SVI) as a priority amplifier to identify high-hazard, high-vulnerability areas for targeted climate adaptation.

```python
import numpy as np
from planx.resilience import multi_hazard_composite, equity_adjusted_priority

# 1. Multi-Hazard Composite Index
hazards = {
    "heat": np.array([80.0, 20.0, 50.0]),
    "flood": np.array([40.0, 10.0, np.nan])  # Automatically masks missing values
}
weights = {"heat": 0.6, "flood": 0.4}

scores, classes, dominant, diversity, drivers = multi_hazard_composite(hazards, weights)
print("Synthesized Composite Scores:", scores)  # [64. 16. 50.]
print("Dominant Hazards:", dominant)  # ['heat', 'heat', 'heat']
print("Multi-Stress Diversity Scores:", diversity)

# 2. Social Vulnerability (SVI) Equity-Adjusted Priority Scoring
svi = np.array([10.0, 90.0, 50.0])  # Social Vulnerability Index
eq_scores, raw, factors, eq_classes = equity_adjusted_priority(scores, svi, equity_weight=0.5)
print("Equity-Adjusted Priority Scores:", eq_scores)
```

### 8. Infrastructure and Network Resilience Analysis (`planx.resilience`)
Simulates street network blocking (e.g., due to seismic debris or flooding) and computes network isolation rates, delay metrics, network bottlenecks, and debris clearance scheduling.

```python
import numpy as np
from planx.resilience import (
    simulate_network_disruption,
    infrastructure_service_loss,
    identify_critical_bottlenecks,
    prioritize_debris_clearance
)

# 3-node network represented in CSR format
indptr = np.array([0, 1, 3, 4], dtype=np.int64)
adj = np.array([1, 0, 2, 1], dtype=np.int64)
weights = np.array([1.5, 1.5, 2.5, 2.5], dtype=np.float64)  # Travel costs (minutes)

# 1. Simulate closure of intersection node 1
disrupted_weights = simulate_network_disruption(indptr, adj, weights, n=3, blocked_nodes=[1])
print("Post-disruption Edge Weights:", disrupted_weights)  # All edges will be inf

# 2. Analyze service loss statistics using pre- and post-disruption distance matrices
dists_pre = np.array([[10.0, 20.0], [15.0, 30.0]])
dists_post = np.array([[15.0, 20.0], [np.inf, np.inf]])  # Neighborhood 2 completely isolated

demands = np.array([100.0, 50.0])  # Population demand (Neighborhood 1: 100, Neighborhood 2: 50)
loss_stats = infrastructure_service_loss(dists_pre, dists_post, demands=demands)
print("Isolation Rate:", loss_stats["isolation_rate"])  # 0.333
print("Mean Delay (Minutes):", loss_stats["mean_delay"])  # 5.0
print("Service Vulnerability Index:", loss_stats["service_vulnerability_index"])

# 3. Identify critical network bottlenecks
pre_usage = np.array([10.0, 5.0, 20.0])
post_usage = np.array([12.0, 5.0, 35.0])
bottlenecks, increases = identify_critical_bottlenecks(pre_usage, post_usage, top_k=2)
print("Critical Bottleneck Edge Indices:", bottlenecks)  # [2, 0]
print("Traffic Load Increase Amounts:", increases)  # [15. 2.]

# 4. Prioritize debris clearance operations (Benefit/Cost ratio method)
blocked_roads = np.array([0, 2])                  # Blocked road indices
debris_volumes = np.array([15.0, 150.0])          # Debris volume on roads (m3)
edge_criticality = np.array([100.0, 20.0, 300.0, 50.0])  # Road network importance score

clearance_order, priority_scores = prioritize_debris_clearance(
    blocked_roads, debris_volumes, edge_criticality, cost_factor=1.0
)
print("Debris Clearance Priority Order:", clearance_order)  # [2, 0]
print("Clearance Priority Scores:", priority_scores)
```

### 9. Flood and Inundation Risk Analysis (`planx.resilience`)
Simulates pluvial (surface water runoff) susceptibility and coastal inundation scenarios to locate flood-prone regions and determine inundation depths.

```python
import numpy as np
from planx.resilience import pluvial_flood_susceptibility, coastal_flood_inundation

# 1. Pluvial (Surface Water) Flood Susceptibility (using DEM grid and curvature proxy)
dem_grid = np.array([
    [10.0, 12.0, 15.0],
    [8.0, 9.0, 11.0],
    [5.0, 7.0, 8.0]
])
# Cell size 10m, local neighborhood relief radius 15m
scores, classes = pluvial_flood_susceptibility(dem_grid, cell_size=10.0, neighborhood_radius=15.0)
print("Pluvial Flood Susceptibility Scores:\n", scores)

# 2. Coastal Inundation & Sea Level Rise (Hydrologically Connected Bathtub Model)
# Sea rise of 8.0m, flooding spreads from top-left (0,0) seed cell (8-connectivity)
sea_mask = np.zeros((3, 3), dtype=bool)
sea_mask[0, 0] = True  # Sea seed start location

flooded, water_depth = coastal_flood_inundation(dem_grid, water_level=8.0, sea_mask=sea_mask)
print("Inundated Area Mask:\n", flooded)
print("Water Depth Matrix:\n", water_depth)
```

### 10. Landslide Susceptibility Analysis (`planx.resilience`)
Combines terrain slope (calculated using Horn's method), soil, and land use/land cover (LULC) factor weights to locate areas prone to slope stability failure.

```python
import numpy as np
from planx.resilience import landslide_susceptibility

# 3x3 DEM grid with elevation changes
dem_grid = np.array([
    [100.0, 100.0, 100.0],
    [50.0, 50.0, 50.0],
    [0.0, 0.0, 0.0]
])

# Cell size of 10.0 meters
scores, classes = landslide_susceptibility(dem_grid, cell_size=10.0)
print("Landslide Susceptibility Scores:\n", scores)
```

### 11. Spatial Interpolation (`planx.geostats`)
Interpolates continuous values (e.g., temperature, rainfall, or pollution) from scattered point observations onto a grid or target point locations using Inverse Distance Weighting (IDW) powered by SciPy KDTree.

```python
import numpy as np
from planx.geostats import idw_to_points, idw_to_grid

# 4 corner observations
src_coords = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
src_values = np.array([10.0, 20.0, 30.0, 40.0])

# 1. Interpolate at target points (e.g., center point (5, 5))
target_coords = np.array([[5.0, 5.0]])
interpolated_points = idw_to_points(src_coords, src_values, target_coords, power=2.0)
print("Interpolated values at points:", interpolated_points)  # Expected: [25.0]

# 2. Interpolate onto a 2x2 grid (grid bounding box xmin, ymin, xmax, ymax)
grid, x, y = idw_to_grid(src_coords, src_values, (0.0, 0.0, 10.0, 10.0), cell_size=5.0)
print("Interpolated grid:\n", grid)
```

### 12. Wildfire Risk Index (`planx.resilience`)
Calculates wildfire exposure/risk based on terrain slope (steeper slopes increase spread rate), aspect (sun exposure alignment drying fuel), and vegetation/fuel density.

```python
import numpy as np
from planx.resilience import wildfire_risk_index

# 3x3 DEM grid
dem_grid = np.array([
    [10.0, 10.0, 10.0],
    [10.0, 10.0, 10.0],
    [10.0, 10.0, 10.0]
])
# 100% vegetation cover
veg_density = np.ones((3, 3))

scores, classes = wildfire_risk_index(dem_grid, cell_size=10.0, vegetation_density=veg_density)
print("Wildfire Risk Scores:\n", scores)
```

---

## 🛠️ Installation & Development

### 1. Standard Installation
You can install the SDK directly from PyPI:
```bash
pip install planx-sdk
```

### 2. Developer Mode (Editable Install)
To clone the repository and run in editable mode for local development:
```bash
git clone https://github.com/YusufEminoglu/planx_sdk.git
cd planx_sdk
pip install -e .[dev]
```

---

## 🧪 Tests and Code Standards

All modules are verified using unit tests in `pytest`. To execute the test suite:
```bash
pytest
```

We follow standard code styling guidelines verified using `ruff` and `black`. To check and format code:
```bash
black .
ruff check .
```

---

## 📝 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
