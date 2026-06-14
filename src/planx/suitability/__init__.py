# -*- coding: utf-8 -*-
"""
PlanX Suitability Lab Submodule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Core Multi-Criteria Decision Analysis (MCDA) mathematical engines,
including reclassification, normalization, and weighted linear combination.
"""

from .mcda import normalize_array, weighted_linear_combination

__all__ = [
    "normalize_array",
    "weighted_linear_combination",
]
