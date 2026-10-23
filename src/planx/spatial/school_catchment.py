# -*- coding: utf-8 -*-
"""Capacitated School Catchment & Educational Equity Spatial Allocator for PlanX."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class StudentDemandZone:
    zone_id: str
    student_count: int
    centroid: tuple[float, float]


@dataclass
class SchoolCapacityRecord:
    school_id: str
    capacity: int
    enrolled_students: int
    utilization_rate_pct: float
    assigned_zones: list[str]


@dataclass
class SchoolCatchmentResult:
    total_schools: int
    total_students: int
    total_assigned_students: int
    mean_student_walk_distance_m: float
    max_student_walk_distance_m: float
    overcapacity_schools_count: int
    # Gini coefficient of travel distance equity (0 = perfect equity)
    gini_accessibility_index: float
    schools: list[SchoolCapacityRecord]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result to a JSON-ready dictionary."""
        return {
            "total_schools": self.total_schools,
            "total_students": self.total_students,
            "mean_walk_distance_m": round(self.mean_student_walk_distance_m, 1),
            "max_walk_distance_m": round(self.max_student_walk_distance_m, 1),
            "overcapacity_count": self.overcapacity_schools_count,
            "gini_equity": round(self.gini_accessibility_index, 3),
        }


def optimize_school_catchments(
    schools_data: list[dict[str, Any]],  # List of dicts with 'id', 'capacity', 'position': (x, y)
    zones_data: list[dict[str, Any]],  # List of dicts with 'id', 'students', 'position': (x, y)
) -> SchoolCatchmentResult:
    """Capacitated greedy nearest allocation of neighborhood zones to schools
    minimizing travel distance.
    """
    school_records: dict[str, dict[str, Any]] = {}
    for s in schools_data:
        s_id = str(s.get("id", "school"))
        cap = int(s.get("capacity", 500))
        pos = s.get("position", (0.0, 0.0))
        school_records[s_id] = {
            "capacity": cap,
            "enrolled": 0,
            "assigned_zones": [],
            "pos": pos,
        }

    total_students = sum(int(z.get("students", 50)) for z in zones_data)
    assigned_students = 0
    all_distances_weighted: list[float] = []

    # Sort zones by demand size descending
    sorted_zones = sorted(zones_data, key=lambda z: int(z.get("students", 50)), reverse=True)

    for z in sorted_zones:
        z_id = str(z.get("id", "zone"))
        cnt = int(z.get("students", 50))
        z_pos = z.get("position", (0.0, 0.0))

        # Find best candidate school (closest with available capacity, or simply closest)
        candidate_schools = []
        for s_id, s_info in school_records.items():
            dist = math.hypot(z_pos[0] - s_info["pos"][0], z_pos[1] - s_info["pos"][1])
            remaining_cap = s_info["capacity"] - s_info["enrolled"]
            candidate_schools.append((dist, remaining_cap, s_id))

        candidate_schools.sort(key=lambda x: (x[1] < cnt, x[0]))

        if candidate_schools:
            best_dist, _, best_s_id = candidate_schools[0]
            school_records[best_s_id]["enrolled"] += cnt
            school_records[best_s_id]["assigned_zones"].append(z_id)
            assigned_students += cnt
            all_distances_weighted.extend([best_dist] * cnt)

    if all_distances_weighted:
        mean_dist = float(sum(all_distances_weighted) / max(1, len(all_distances_weighted)))
    else:
        mean_dist = 0.0
    max_dist = max(all_distances_weighted) if all_distances_weighted else 0.0

    # Calculate Gini coefficient of distances
    if all_distances_weighted and len(all_distances_weighted) > 1:
        sorted_d = sorted(all_distances_weighted)
        n_d = len(sorted_d)
        cum_d = sum((i + 1) * d for i, d in enumerate(sorted_d))
        tot_d = sum(sorted_d)
        gini = (2.0 * cum_d) / max(1e-4, n_d * tot_d) - (n_d + 1.0) / n_d
        gini = max(0.0, min(1.0, gini))
    else:
        gini = 0.0

    schools_out: list[SchoolCapacityRecord] = []
    overcapacity_cnt = 0

    for s_id, s_info in school_records.items():
        cap = s_info["capacity"]
        enr = s_info["enrolled"]
        util = (enr / max(1, cap)) * 100.0
        if enr > cap:
            overcapacity_cnt += 1

        schools_out.append(
            SchoolCapacityRecord(
                school_id=s_id,
                capacity=cap,
                enrolled_students=enr,
                utilization_rate_pct=round(util, 1),
                assigned_zones=s_info["assigned_zones"],
            )
        )

    return SchoolCatchmentResult(
        total_schools=len(schools_data),
        total_students=total_students,
        total_assigned_students=assigned_students,
        mean_student_walk_distance_m=mean_dist,
        max_student_walk_distance_m=max_dist,
        overcapacity_schools_count=overcapacity_cnt,
        gini_accessibility_index=gini,
        schools=schools_out,
    )
