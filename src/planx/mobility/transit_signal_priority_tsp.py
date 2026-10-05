# -*- coding: utf-8 -*-
"""Smart Intersection Transit Signal Priority (TSP) & Bus Green Wave Simulator for PlanX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class SignalizedJunction:
    junction_id: str
    distance_along_corridor_m: float
    cycle_time_seconds: float = 90.0
    green_split_seconds: float = 35.0
    max_green_extension_seconds: float = 12.0
    early_green_truncation_seconds: float = 8.0


@dataclass
class TSPPerformanceResult:
    total_corridor_length_m: float
    baseline_travel_time_seconds: float
    tsp_optimized_travel_time_seconds: float
    bus_delay_reduction_pct: float
    junctions_granted_priority_count: int
    corridor_average_speed_kmh: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "corridor_length_m": round(self.total_corridor_length_m, 1),
            "baseline_time_s": round(self.baseline_travel_time_seconds, 1),
            "tsp_time_s": round(self.tsp_optimized_travel_time_seconds, 1),
            "delay_reduction_pct": round(self.bus_delay_reduction_pct, 1),
            "priority_junctions": self.junctions_granted_priority_count,
            "avg_speed_kmh": round(self.corridor_average_speed_kmh, 1),
        }


def simulate_transit_signal_priority(
    junctions: Sequence[SignalizedJunction | dict[str, Any]],
    bus_cruising_speed_kmh: float = 40.0,
    dwell_time_per_stop_seconds: float = 20.0,
) -> TSPPerformanceResult:
    """Simulate municipal bus corridor transit travel times with Green Extension
    and Early Green algorithms.
    """
    juncs: list[SignalizedJunction] = []
    for j in junctions:
        if isinstance(j, dict):
            juncs.append(
                SignalizedJunction(
                    junction_id=str(j.get("id", "JUNC")),
                    distance_along_corridor_m=float(j.get("distance_m", 500.0)),
                    cycle_time_seconds=float(j.get("cycle_s", 90.0)),
                    green_split_seconds=float(j.get("green_s", 35.0)),
                    max_green_extension_seconds=float(j.get("green_ext_s", 12.0)),
                    early_green_truncation_seconds=float(j.get("early_green_s", 8.0)),
                )
            )
        else:
            juncs.append(j)

    if not juncs:
        return TSPPerformanceResult(0.0, 0.0, 0.0, 0.0, 0, 0.0)

    juncs.sort(key=lambda x: x.distance_along_corridor_m)
    tot_len = juncs[-1].distance_along_corridor_m
    cruise_speed_mps = bus_cruising_speed_kmh / 3.6

    base_time = 0.0
    tsp_time = 0.0
    priority_count = 0

    current_dist = 0.0

    for junc in juncs:
        dist_seg = junc.distance_along_corridor_m - current_dist
        t_travel = dist_seg / max(1.0, cruise_speed_mps) + dwell_time_per_stop_seconds

        base_time += t_travel
        tsp_time += t_travel

        # Arrival phase at junction without TSP
        phase_in_cycle = base_time % junc.cycle_time_seconds
        if phase_in_cycle > junc.green_split_seconds:
            # Caught by Red light
            red_duration = junc.cycle_time_seconds - junc.green_split_seconds
            wait_time = (junc.cycle_time_seconds - phase_in_cycle)
            base_time += wait_time

        # Arrival with TSP
        tsp_phase = tsp_time % junc.cycle_time_seconds
        if tsp_phase <= junc.green_split_seconds + junc.max_green_extension_seconds:
            # Covered by normal green or green extension
            priority_count += 1
        elif (junc.cycle_time_seconds - tsp_phase) <= junc.early_green_truncation_seconds:
            # Covered by early green recall
            priority_count += 1
        else:
            # Still small remaining red wait
            tsp_wait = max(
                0.0,
                (junc.cycle_time_seconds - tsp_phase) - junc.early_green_truncation_seconds,
            )
            tsp_time += tsp_wait

        current_dist = junc.distance_along_corridor_m

    reduction = ((base_time - tsp_time) / max(1.0, base_time)) * 100.0
    avg_speed = (tot_len / max(1.0, tsp_time)) * 3.6

    return TSPPerformanceResult(
        total_corridor_length_m=tot_len,
        baseline_travel_time_seconds=base_time,
        tsp_optimized_travel_time_seconds=tsp_time,
        bus_delay_reduction_pct=max(0.0, reduction),
        junctions_granted_priority_count=priority_count,
        corridor_average_speed_kmh=avg_speed,
    )
