# -*- coding: utf-8 -*-
"""PlanX Cookbook — Sponge City, UHI Mitigation, School Catchment & TOD."""

import planx

# 1. Sponge City Runoff
sponge = planx.climate.sponge_city.simulate_sponge_city_runoff(
    landuse_areas_m2={"impervious": 30000.0, "green": 20000.0},
    lid_facilities=[{"type": "bioswale", "area_m2": 800.0, "storage_depth_mm": 200.0}],
)
print(f"Sponge City Runoff Reduction: {sponge.runoff_reduction_percent:.1f}%")

# 2. UHI Heat Island Mitigation
uhi = planx.urban_physics.uhi_mitigation.simulate_uhi_temperature_reduction(
    urban_area_m2=50000.0, roof_area_m2=20000.0, pavement_area_m2=20000.0, cool_roof_fraction=0.70
)
print(f"UHI Air Temperature Drop: -{uhi.air_temp_reduction_c:.2f} °C")

# 3. School Catchments & Gini Equity
schools = [{"id": "S1", "capacity": 500, "position": (0, 0)}]
zones = [{"id": "Z1", "students": 200, "position": (100, 100)}]
catchment = planx.spatial.school_catchment.optimize_school_catchments(schools, zones)
print(f"Assigned Students: {catchment.total_assigned_students}/{catchment.total_students}")
