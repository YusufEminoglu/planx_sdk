# -*- coding: utf-8 -*-
"""
PlanX Urban Resilience Submodule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Models and simulations for evaluating urban vulnerability, seismic risk,
building collapse debris, and infrastructure recovery corridors.
"""

from .flood import coastal_flood_inundation, pluvial_flood_susceptibility, socio_economic_flood_risk
from .heat import urban_heat_comfort_risk, urban_heat_island_intensity
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
]
