# -*- coding: utf-8 -*-
"""Municipal Water Distribution Pipe Network Hydraulic Simulator (Hazen-Williams) for PlanX."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class PipeHydraulicSegment:
    pipe_id: str
    from_node: str
    to_node: str
    length_m: float
    diameter_mm: float
    roughness_c: float = 130.0  # Hazen-Williams C factor (e.g. 130 for ductile iron / PVC)
    flow_rate_lps: float = 25.0  # liters per second


@dataclass
class WaterNetworkResult:
    total_network_length_m: float
    max_flow_velocity_ms: float
    total_head_loss_m: float
    nodes_below_minimum_pressure_count: int  # e.g. < 20m pressure head
    pipe_diagnostics: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result to a JSON-ready dictionary."""
        return {
            "network_length_m": round(self.total_network_length_m, 1),
            "max_velocity_ms": round(self.max_flow_velocity_ms, 2),
            "total_head_loss_m": round(self.total_head_loss_m, 2),
            "low_pressure_nodes": self.nodes_below_minimum_pressure_count,
        }


def simulate_water_pipe_network(
    pipes: Sequence[PipeHydraulicSegment | dict[str, Any]],
    source_pressure_head_m: float = 60.0,
    min_allowable_pressure_head_m: float = 20.0,
) -> WaterNetworkResult:
    """Compute Hazen-Williams head loss, velocities, and residual pressure across pipe network."""
    pipe_list: list[PipeHydraulicSegment] = []
    for p in pipes:
        if isinstance(p, dict):
            pipe_list.append(
                PipeHydraulicSegment(
                    pipe_id=str(p.get("id", "PIPE")),
                    from_node=str(p.get("from", "N1")),
                    to_node=str(p.get("to", "N2")),
                    length_m=float(p.get("length_m", 100.0)),
                    diameter_mm=float(p.get("diameter_mm", 150.0)),
                    roughness_c=float(p.get("c_factor", 130.0)),
                    flow_rate_lps=float(p.get("flow_lps", 20.0)),
                )
            )
        else:
            pipe_list.append(p)

    tot_len = 0.0
    tot_head_loss = 0.0
    max_vel = 0.0
    low_press_count = 0
    diagnostics = []

    current_head = source_pressure_head_m

    for pipe in pipe_list:
        tot_len += pipe.length_m
        d_m = pipe.diameter_mm / 1000.0
        area_m2 = math.pi * (d_m / 2.0) ** 2

        # Flow velocity: v = Q / A (m/s)
        q_m3s = pipe.flow_rate_lps / 1000.0
        vel = q_m3s / max(1e-4, area_m2)
        max_vel = max(max_vel, vel)

        # Hazen-Williams head loss formula (SI metric):
        # h_f = 10.67 * L * Q^1.852 / (C^1.852 * D^4.87)
        h_loss = (10.67 * pipe.length_m * (q_m3s**1.852)) / (
            (pipe.roughness_c**1.852) * (d_m**4.87)
        )
        tot_head_loss += h_loss

        current_head -= h_loss
        if current_head < min_allowable_pressure_head_m:
            low_press_count += 1

        diagnostics.append(
            {
                "pipe_id": pipe.pipe_id,
                "velocity_ms": round(vel, 2),
                "head_loss_m": round(h_loss, 3),
                "residual_head_m": round(current_head, 2),
            }
        )

    return WaterNetworkResult(
        total_network_length_m=tot_len,
        max_flow_velocity_ms=max_vel,
        total_head_loss_m=tot_head_loss,
        nodes_below_minimum_pressure_count=low_press_count,
        pipe_diagnostics=diagnostics,
    )
