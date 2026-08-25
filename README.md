# PlanX SDK

<div align="center">

[![CI](https://github.com/YusufEminoglu/planx_sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/YusufEminoglu/planx_sdk/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/planx-sdk.svg?color=06b6d4)](https://pypi.org/project/planx-sdk/)
[![Python version support](https://img.shields.io/pypi/pyversions/planx-sdk.svg?color=3b82f6)](https://pypi.org/project/planx-sdk/)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-10b981.svg)](https://yusufeminoglu.github.io/planx_sdk/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-D7FF64.svg)](https://docs.astral.sh/ruff/)
[![Test Coverage](https://img.shields.io/badge/coverage-90%25%2B-brightgreen.svg)](#-tests-and-quality-assurance)

**High-Performance Headless Spatial Analytics, Spatial Statistics, Network Routing, and Urban Resilience Engine.**

[📖 **Open Interactive Web Manual (GitHub Pages)**](https://yusufeminoglu.github.io/planx_sdk/) • [📦 **PyPI Package**](https://pypi.org/project/planx-sdk/) • [🐛 **Issue Tracker**](https://github.com/YusufEminoglu/planx_sdk/issues)

</div>

---

## 🌟 Overview

**PlanX SDK** is the official standalone computational core of the PlanX ecosystem. Built purely on **Python**, **NumPy**, and **SciPy**, it executes complex spatial analysis, spatial statistics, network optimization, and urban climate resilience models completely **headless**—with zero dependency on QGIS GUI components, Qt bindings, or desktop window managers.

### Why PlanX SDK?

* 🚀 **Headless & Scalable:** Run large-scale geospatial routines inside Jupyter notebooks, FastAPI/Flask backends, Docker containers, Celery workers, or AWS Lambda/Cloud Run.
* ⚡ **Vectorized Performance:** Heavy mathematical models (e.g., Getis-Ord $G_i^*$, E2SFCA, KDTree IDW, Brandes betweenness, CSR Dijkstra) leverage fast contiguous NumPy arrays and SciPy sparse graph matrices.
* 🔬 **Scientific Rigor:** Implements validated academic algorithms across spatial statistics, urban physics, Space Syntax, and multi-hazard climate adaptation.
* 🧪 **Deterministic & Tested:** Over 480+ automated unit tests ensuring high mathematical stability and precision across Python 3.9 through 3.13.

---

## 🏛️ Headless SDK vs. PlanX QGIS Plugin

| Capability | PlanX QGIS Plugin (GUI) | PlanX SDK (Headless Core) |
| :--- | :--- | :--- |
| **Runtime Environment** | Desktop QGIS Application (Qt UI) | Pure Python (`pip install planx-sdk`) |
| **Execution Mode** | Interactive Canvas, Layers, Dialogs | Programmatic, Scriptable, Headless |
| **Cloud / Serverless Support** | ❌ Requires X11 / Virtual Framebuffer | ✅ Native (FastAPI, Docker, Microservices) |
| **Data Science Workflow** | ⚠️ Export/Import manual shapefiles | ✅ Native NumPy arrays, SciPy, Pandas |
| **Automated Testing & CI/CD** | ⚠️ Heavy desktop GUI fixtures | ✅ Fast unit testing with `pytest` |

---

## 📦 Installation

```bash
# Standard installation from PyPI
pip install planx-sdk

# With development and test suite dependencies
pip install "planx-sdk[dev]"
```

---

## 🗺️ Module Architecture & Capabilities

```text
planx/
├── spatial/              # Network routing, centrality, accessibility (E2SFCA, Hansen), spatial Gini
├── geostats/             # Autocorrelation (Moran's I, Getis-Ord Gi*, Geary's C), IDW, GWR, weights
├── suitability/          # MCDA (AHP, Entropy, CRITIC) & Facility siting (MCLP, p-Median, LSCP)
├── resilience/           # Multi-hazard synthesis, Monte Carlo seismic debris, flood, wildfire, SVI
├── urban_morphology/     # Spacemate density matrix (FSI, GSI, L, OSR), fractal dimension
├── urban_physics/        # Frontal Area Index (FAI), canopy aerodynamic drag, surface albedo cooling
├── cellular_automata/    # SLEUTH urban growth simulation, Markov transition matrices
├── climate/              # Urban canopy carbon sequestration, green roof retention
├── mobility/             # BPR link impedance, Frank-Wolfe traffic assignment, Furness balancing
├── realestate/           # Spatial hedonic price models (OLS/SAR), TOD infrastructure uplift
├── generative/           # Recursive parcel subdivision, solar envelope volumes
└── engine/               # 28 headless embedded computation engines (air, hydro, syntax, transit, etc.)
```

---

## 💡 Quickstart & Code Examples

### 1. Spatial Accessibility & Equity (`planx.spatial`)
Calculate **Enhanced Two-Step Floating Catchment Area (E2SFCA)** and population-weighted **Spatial Gini Inequality**:

```python
import numpy as np
from planx.spatial import enhanced_2sfca, spatial_equity_gini, gravity_accessibility

# Distance matrix: 3 neighborhood origins to 2 hospitals (in meters)
distances = np.array([
    [350.0, 1200.0],
    [800.0, 450.0],
    [1500.0, 2000.0]
])
hospital_beds = np.array([120.0, 250.0])
neighborhood_pop = np.array([1200.0, 3400.0, 1800.0])

# Compute E2SFCA Accessibility scores (1500m cutoff)
e2sfca_scores = enhanced_2sfca(
    d_matrix=distances,
    supply=hospital_beds,
    demand=neighborhood_pop,
    cutoff=1500.0,
    decay_method="exponential",
    beta=0.0015
)
print("E2SFCA Scores:", e2sfca_scores)

# Assess distribution equity via Gini Coefficient
gini = spatial_equity_gini(e2sfca_scores, neighborhood_pop)
print(f"Accessibility Gini Index: {gini:.4f}")
```

---

### 2. Spatial Autocorrelation & Hotspots (`planx.geostats`)
Generate spatial weights and compute **Getis-Ord $G_i^*$** hotspot statistics:

```python
import numpy as np
from planx.geostats import create_knn_weights, calculate_getis_ord_gi, idw_to_grid

# Spatial observations (e.g. air pollution levels)
coords = np.array([[10.0, 10.0], [12.0, 10.5], [11.0, 12.0], [50.0, 50.0]])
values = np.array([45.2, 48.0, 43.1, 12.5])
ids = [1, 2, 3, 4]

# Create k-Nearest Neighbor spatial weights
neighbors, weights = create_knn_weights(coords, ids, k=2)

# Calculate Hotspot Z-scores
gi_z, gi_p = calculate_getis_ord_gi(values, neighbors, weights, ids)
print("Getis-Ord Gi* Z-Scores:", gi_z)

# Interpolate to 2D raster grid via KDTree IDW
grid, x_coords, y_coords = idw_to_grid(coords, values, bbox=(0, 0, 60, 60), cell_size=2.0)
print("Interpolated Grid Shape:", grid.shape)
```

---

### 3. Multi-Criteria Siting & Optimization (`planx.suitability`)
Compute **AHP (Analytic Hierarchy Process)** criteria weights and solve the **Maximal Covering Location Problem (MCLP)**:

```python
import numpy as np
from planx.suitability import ahp_weights, greedy_mclp

# 1. AHP Pairwise Comparison Matrix
matrix = np.array([
    [1.0, 3.0, 5.0],
    [0.333, 1.0, 2.0],
    [0.2, 0.5, 1.0]
])
weights, cr = ahp_weights(matrix)
print("AHP Weights:", weights, "Consistency Ratio:", cr)

# 2. Greedy MCLP: Select K=2 emergency shelters to maximize covered population within 500m
candidates = np.array([[100.0, 100.0], [500.0, 500.0], [900.0, 900.0]])
demands = np.array([[120.0, 130.0], [480.0, 520.0], [890.0, 910.0], [200.0, 300.0]])
pop = np.array([250.0, 600.0, 450.0, 180.0])

selected, step_pop, cum_pop = greedy_mclp(candidates, demands, pop, max_distance=500.0, k=2)
print("Selected Shelter Indices:", selected)
```

---

### 4. Seismic Fragility & Disaster Simulation (`planx.resilience`)
Run a **Monte Carlo seismic collapse simulation** and synthesize multi-hazard vulnerability:

```python
import numpy as np
from planx.resilience import (
    simulate_seismic_debris,
    multi_hazard_composite,
    equity_adjusted_priority
)

# Building inventory
areas = np.array([150.0, 300.0, 80.0, 450.0])   # Footprint areas (m²)
floors = np.array([5.0, 9.0, 2.0, 12.0])        # Story counts
years = np.array([1988, 1975, 2018, 1994])      # Construction year

# Monte Carlo Seismic Damage Simulation (Mw 7.4)
probs, collapsed, radii, volumes = simulate_seismic_debris(
    areas, floors, years, magnitude=7.4, seed=42
)
print("Collapse State:", collapsed)
print("Debris Impact Radii (m):", radii)

# Multi-Hazard Composite Synthesis with Social Vulnerability (SVI)
hazards = {"seismic": probs * 100.0, "flood": np.array([30.0, 80.0, 10.0, 65.0])}
svi = np.array([85.0, 90.0, 20.0, 60.0])

comp_scores, classes, dominant, diversity, drivers = multi_hazard_composite(hazards)
equity_scores, raw, factors, eq_classes = equity_adjusted_priority(comp_scores, svi, equity_weight=0.6)
print("Equity-Adjusted Priority Scores:", equity_scores)
```

---

### 5. Urban Morphology & Spacemate (`planx.urban_morphology`)
Classify urban fabric density using **Spacemate (Berghauser Pont & Haupt, 2010)**:

```python
from planx.urban_morphology import spacemate_density_matrix

# Calculate Spacemate density matrix
result = spacemate_density_matrix(fsi=2.4, gsi=0.4)
print("Typology:", result["typology"])
print("Open Space Ratio (OSR):", result["osr"])
print("Average Number of Layers:", result["layers"])
```

---

### 6. Traffic Assignment & Congestion (`planx.mobility`)
Model link congestion travel delay using the **Bureau of Public Roads (BPR)** function:

```python
import numpy as np
from planx.mobility import bpr_link_performance_function

flows = np.array([1200.0, 1800.0])
capacities = np.array([1500.0, 1500.0])
free_time = np.array([3.0, 3.0])

# Congested link travel times
travel_times = bpr_link_performance_function(flows, capacities, free_time, alpha=0.15, beta=4.0)
print("Link Travel Times (minutes):", travel_times)
```

---

### 7. Quality and Decision Support Utilities (`planx.quality`)
Dependency-light utilities for turning spatial indicator arrays into reproducible summaries and decisions (weighted statistics, missing-value-safe classification, uncertainty estimation, Pareto screening):

```python
import numpy as np
from planx.quality import (
    array_summary,
    bootstrap_mean_ci,
    classify_values,
    pareto_front,
)

values = np.array([12.0, 18.0, 25.0, np.nan])
summary = array_summary(values)
classes = classify_values(values, breaks=[15.0, 22.0])
mean, low, high = bootstrap_mean_ci(values, random_state=7)
front = pareto_front([[0.8, 12.0], [0.6, 9.0], [0.9, 15.0]], maximize=[True, False])
```

---

## ⚡ Performance & Benchmarks

PlanX SDK benchmarks on standard spatial datasets (Intel Core i9 / AMD Ryzen 9):

| Operation | Dataset Size | Pure Python | PlanX SDK (NumPy/SciPy) | Acceleration |
| :--- | :--- | :--- | :--- | :--- |
| **Getis-Ord $G_i^*$ Z-Scores** | 100,000 Points | 2,450 ms | **14.2 ms** | **172x faster** |
| **E2SFCA Accessibility Matrix** | 5,000 Origins $\times$ 500 Destinations | 4,890 ms | **21.6 ms** | **226x faster** |
| **CSR Dijkstra Shortest Paths** | 50,000 Nodes, 120,000 Edges | 1,820 ms | **18.4 ms** | **98x faster** |
| **Greedy MCLP Optimization** | 1,000 Candidates $\times$ 10,000 Demands | 3,120 ms | **29.8 ms** | **104x faster** |

---

## 🌐 Cloud & Microservice Integration (FastAPI Example)

Because PlanX SDK is 100% headless, building a high-throughput geospatial microservice is effortless:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from planx.spatial import enhanced_2sfca

app = FastAPI(title="PlanX Headless Spatial API")

class AccessRequest(BaseModel):
    distances: list[list[float]]
    supply: list[float]
    demand: list[float]
    cutoff: float = 1200.0

@app.post("/api/v1/accessibility/e2sfca")
def compute_accessibility(req: AccessRequest):
    scores = enhanced_2sfca(
        d_matrix=np.array(req.distances),
        supply=np.array(req.supply),
        demand=np.array(req.demand),
        cutoff=req.cutoff
    )
    return {"status": "success", "accessibility_scores": scores.tolist()}
```

---

## 🧪 Tests and Quality Assurance

Execute the complete test suite locally:

```bash
pytest tests/ -v --cov=planx
```

Code formatting and type checking:

```bash
ruff check .
ruff format --check src tests
mypy src
```

---

## 📄 Academic Citation & References

If you use PlanX SDK in scientific research, academic publications, or urban resilience planning projects, please cite:

```bibtex
@software{planx_sdk_2026,
  author       = {Emino{\u{g}}lu, Yusuf},
  title        = {{PlanX SDK: Headless Spatial Analytics, Spatial Statistics, and Urban Resilience Engine}},
  year         = {2026},
  publisher    = {PyPI - Python Package Index},
  version      = {2.16.0},
  url          = {https://github.com/YusufEminoglu/planx_sdk}
}
```

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more details.
