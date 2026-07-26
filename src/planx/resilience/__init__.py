# -*- coding: utf-8 -*-
"""
PlanX Urban Resilience Submodule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Models and simulations for evaluating urban vulnerability, seismic risk,
building collapse debris, and infrastructure recovery corridors.
"""

from .active_travel import (
    active_travel_equity_gini,
    calculate_tod_index,
    equity_weighted_accessibility,
    job_housing_spatial_mismatch,
    transport_mismatch_index,
)
from .evacuation import evacuation_route_optimization
from .flood import (
    coastal_flood_inundation,
    coastal_surge_inundation,
    pluvial_flood_susceptibility,
    socio_economic_flood_risk,
)
from .heat import (
    calculate_grid_sky_view_factor,
    calculate_solar_access,
    classify_local_climate_zones,
    optimize_canopy_placement,
    urban_heat_comfort_risk,
    urban_heat_island_intensity,
)
from .infrastructure import (
    debris_clearance_routing,
    identify_critical_bottlenecks,
    infrastructure_service_loss,
    network_criticality_index,
    prioritize_debris_clearance,
    simulate_network_disruption,
)
from .landslide import landslide_susceptibility
from .seismic import simulate_seismic_debris
from .social import social_vulnerability_index
from .synthesis import equity_adjusted_priority, multi_hazard_composite
from .wildfire import wildfire_risk_index

__all__ = [
    "simulate_seismic_debris",
    "pluvial_flood_susceptibility",
    "coastal_flood_inundation",
    "coastal_surge_inundation",
    "socio_economic_flood_risk",
    "landslide_susceptibility",
    "wildfire_risk_index",
    "social_vulnerability_index",
    "urban_heat_comfort_risk",
    "urban_heat_island_intensity",
    "multi_hazard_composite",
    "equity_adjusted_priority",
    "simulate_network_disruption",
    "infrastructure_service_loss",
    "identify_critical_bottlenecks",
    "prioritize_debris_clearance",
    "network_criticality_index",
    "debris_clearance_routing",
    "evacuation_route_optimization",
    "job_housing_spatial_mismatch",
    "active_travel_equity_gini",
    "transport_mismatch_index",
    "optimize_canopy_placement",
    "calculate_tod_index",
    "equity_weighted_accessibility",
    "calculate_grid_sky_view_factor",
    "classify_local_climate_zones",
    "calculate_solar_access",
]
