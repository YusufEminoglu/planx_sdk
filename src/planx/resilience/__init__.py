# -*- coding: utf-8 -*-
"""
PlanX Urban Resilience Submodule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Models and simulations for evaluating urban vulnerability, seismic risk,
building collapse debris, and infrastructure recovery corridors.
"""

from .flood import coastal_flood_inundation, pluvial_flood_susceptibility
from .heat import urban_heat_comfort_risk
from .infrastructure import (
    identify_critical_bottlenecks,
    infrastructure_service_loss,
    prioritize_debris_clearance,
    simulate_network_disruption,
)
from .seismic import simulate_seismic_debris
from .social import social_vulnerability_index
from .synthesis import equity_adjusted_priority, multi_hazard_composite

__all__ = [
    "simulate_seismic_debris",
    "pluvial_flood_susceptibility",
    "coastal_flood_inundation",
    "social_vulnerability_index",
    "urban_heat_comfort_risk",
    "multi_hazard_composite",
    "equity_adjusted_priority",
    "simulate_network_disruption",
    "infrastructure_service_loss",
    "identify_critical_bottlenecks",
    "prioritize_debris_clearance",
]
