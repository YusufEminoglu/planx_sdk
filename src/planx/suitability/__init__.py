# -*- coding: utf-8 -*-
"""
PlanX Suitability Lab Submodule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Core Multi-Criteria Decision Analysis (MCDA) mathematical engines,
including reclassification, normalization, and weighted linear combination.
"""

from .facility import greedy_mclp
from .mcda import normalize_array, weighted_linear_combination
from .weights import (
    ahp_weights,
    critic_weights,
    decision_matrix_from_layers,
    entropy_weights,
    pca_weights,
)

__all__ = [
    "normalize_array",
    "weighted_linear_combination",
    "greedy_mclp",
    "ahp_weights",
    "decision_matrix_from_layers",
    "entropy_weights",
    "critic_weights",
    "pca_weights",
]
