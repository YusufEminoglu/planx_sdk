# -*- coding: utf-8 -*-
"""PlanX Urban Demographics & Aging Accessibility Submodule."""

from .aging_mobility_index import (
    AgingMobilityResult,
    SeniorZoneProfile,
    calculate_silver_mobility_score,
)

__all__ = [
    "AgingMobilityResult",
    "SeniorZoneProfile",
    "calculate_silver_mobility_score",
]
