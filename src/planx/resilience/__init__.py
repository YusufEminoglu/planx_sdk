# -*- coding: utf-8 -*-
"""
PlanX Urban Resilience Submodule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Models and simulations for evaluating urban vulnerability, seismic risk,
building collapse debris, and infrastructure recovery corridors.
"""

from .seismic import simulate_seismic_debris

__all__ = [
    "simulate_seismic_debris",
]
