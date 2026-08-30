# -*- coding: utf-8 -*-
"""Smart Municipal Electric Bus & Fleet Depot Charging Schedule Optimizer for PlanX."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class EVBusProfile:
    bus_id: str
    battery_capacity_kwh: float = 350.0
    arrival_soc_pct: float = 20.0
    target_departure_soc_pct: float = 90.0
    arrival_hour: int = 21  # 9 PM
    departure_hour: int = 6  # 6 AM
    max_charger_power_kw: float = 150.0  # DC Fast Charger


@dataclass
class ChargingDepotResult:
    total_energy_delivered_kwh: float
    peak_grid_load_kw: float
    total_charging_cost: float
    unmet_demand_buses_count: int
    hourly_depot_load_profile_kw: list[float]
    bus_schedules: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "energy_delivered_kwh": round(self.total_energy_delivered_kwh, 1),
            "peak_grid_load_kw": round(self.peak_grid_load_kw, 1),
            "total_cost": round(self.total_charging_cost, 2),
            "unmet_count": self.unmet_demand_buses_count,
        }


def optimize_ev_fleet_charging(
    buses: Sequence[EVBusProfile | dict[str, Any]],
    max_grid_transformer_capacity_kw: float = 800.0,
    electricity_price_schedule_per_kwh: Sequence[float] | None = None,
) -> ChargingDepotResult:
    """Smart charging schedule optimization leveling peak demand and prioritizing off-peak night tariff."""
    # Default 24h tariff: off-peak (00-06) = $0.08, normal = $0.14, peak (17-22) = $0.24
    if electricity_price_schedule_per_kwh is None:
        tariffs = [
            0.08, 0.08, 0.08, 0.08, 0.08, 0.08,  # 00:00 - 06:00 (Off-peak night)
            0.12, 0.14, 0.16, 0.16, 0.14, 0.14,  # 06:00 - 12:00
            0.14, 0.14, 0.15, 0.16, 0.18, 0.24,  # 12:00 - 18:00
            0.24, 0.22, 0.18, 0.14, 0.10, 0.08,  # 18:00 - 24:00
        ]
    else:
        tariffs = list(electricity_price_schedule_per_kwh)

    bus_list: list[EVBusProfile] = []
    for b in buses:
        if isinstance(b, dict):
            bus_list.append(
                EVBusProfile(
                    bus_id=str(b.get("bus_id", "BUS")),
                    battery_capacity_kwh=float(b.get("capacity_kwh", 350.0)),
                    arrival_soc_pct=float(b.get("arrival_soc", 20.0)),
                    target_departure_soc_pct=float(b.get("target_soc", 90.0)),
                    arrival_hour=int(b.get("arrival_hour", 21)),
                    departure_hour=int(b.get("departure_hour", 6)),
                    max_charger_power_kw=float(b.get("max_kw", 150.0)),
                )
            )
        else:
            bus_list.append(b)

    hourly_load = [0.0] * 24
    tot_energy = 0.0
    tot_cost = 0.0
    unmet = 0
    schedules = []

    for bus in bus_list:
        needed_kwh = (bus.target_departure_soc_pct - bus.arrival_soc_pct) / 100.0 * bus.battery_capacity_kwh
        if needed_kwh <= 0:
            continue

        # Available charging hours window (wrapping midnight)
        avail_hours = []
        h = bus.arrival_hour
        while h != bus.departure_hour:
            avail_hours.append(h)
            h = (h + 1) % 24

        # Sort available hours by cheapest electricity price
        sorted_hours = sorted(avail_hours, key=lambda hr: tariffs[hr])

        remaining_need = needed_kwh
        charged_by_hour = {}

        for hr in sorted_hours:
            if remaining_need <= 0:
                break
            # Check grid transformer capacity headroom
            headroom = max_grid_transformer_capacity_kw - hourly_load[hr]
            charge_rate = min(bus.max_charger_power_kw, headroom, remaining_need)
            if charge_rate > 0:
                hourly_load[hr] += charge_rate
                charged_by_hour[hr] = charge_rate
                remaining_need -= charge_rate
                tot_energy += charge_rate
                tot_cost += charge_rate * tariffs[hr]

        if remaining_need > 1.0:
            unmet += 1

        schedules.append({
            "bus_id": bus.bus_id,
            "needed_kwh": needed_kwh,
            "delivered_kwh": needed_kwh - remaining_need,
            "hourly_kw": charged_by_hour,
        })

    peak_load = max(hourly_load) if hourly_load else 0.0

    return ChargingDepotResult(
        total_energy_delivered_kwh=tot_energy,
        peak_grid_load_kw=peak_load,
        total_charging_cost=tot_cost,
        unmet_demand_buses_count=unmet,
        hourly_depot_load_profile_kw=hourly_load,
        bus_schedules=schedules,
    )
