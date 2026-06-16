# -*- coding: utf-8 -*-
"""
PlanX Suitability Lab Submodule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Core Multi-Criteria Decision Analysis (MCDA) mathematical engines,
including reclassification, normalization, and weighted linear combination.
"""

from .facility import (
    capacitated_location_allocation,
    greedy_lscp,
    greedy_mclp,
    greedy_p_median,
)
from .mcda import normalize_array, topsis_method, vikor_method, weighted_linear_combination
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
    "topsis_method",
    "vikor_method",
    "greedy_mclp",
    "greedy_p_median",
    "greedy_lscp",
    "capacitated_location_allocation",
    "ahp_weights",
    "decision_matrix_from_layers",
    "entropy_weights",
    "critic_weights",
    "pca_weights",
]
