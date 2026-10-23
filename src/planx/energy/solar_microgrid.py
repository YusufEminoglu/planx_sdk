# -*- coding: utf-8 -*-
"""Community Rooftop PV + Battery Storage Microgrid Energy Simulator for PlanX."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class BatteryStorageProfile:
    capacity_kwh: float = 200.0
    initial_charge_kwh: float = 100.0
    max_charge_rate_kw: float = 50.0
    max_discharge_rate_kw: float = 50.0
    roundtrip_efficiency: float = 0.92


@dataclass
class SolarMicrogridResult:
    total_pv_generation_kwh: float
    total_building_demand_kwh: float
    grid_import_kwh: float
    grid_export_kwh: float
    battery_energy_stored_kwh: float
    solar_self_consumption_pct: float
    self_sufficiency_ratio_pct: float  # SSR % (demand met locally)
    hourly_state_of_charge: list[float]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result to a JSON-ready dictionary."""
        return {
            "pv_generation_kwh": round(self.total_pv_generation_kwh, 1),
            "demand_kwh": round(self.total_building_demand_kwh, 1),
            "grid_import_kwh": round(self.grid_import_kwh, 1),
            "grid_export_kwh": round(self.grid_export_kwh, 1),
            "self_sufficiency_ratio_pct": round(self.self_sufficiency_ratio_pct, 1),
            "self_consumption_pct": round(self.solar_self_consumption_pct, 1),
        }


def simulate_community_solar_microgrid(
    rooftop_area_m2: float = 5000.0,
    solar_efficiency: float = 0.20,  # 20% modern monocrystalline PV
    daily_insolation_kwh_m2: float = 5.2,  # Average sunny day in kWh/m2
    total_daily_demand_kwh: float = 4500.0,
    battery_profile: BatteryStorageProfile | None = None,
) -> SolarMicrogridResult:
    """Simulate 24-hour diurnal community microgrid energy balance and battery state of charge."""
    battery = battery_profile or BatteryStorageProfile()

    # Total daily PV generation
    pv_total = rooftop_area_m2 * solar_efficiency * daily_insolation_kwh_m2

    # Diurnal solar bell curve profile over 24 hours (peak at hour 13)
    hourly_pv = []
    for h in range(24):
        if 6 <= h <= 18:
            fraction = math.sin((h - 6) / 12.0 * math.pi)
            hourly_pv.append(pv_total * (fraction / 7.64))  # normalized sum ~ pv_total
        else:
            hourly_pv.append(0.0)

    # Typical residential/commercial double-peak demand curve
    hourly_demand = []
    base_demand = total_daily_demand_kwh / 24.0
    for h in range(24):
        if h < 6:
            factor = 0.6
        elif 8 <= h <= 11:
            factor = 1.2
        elif 12 <= h <= 16:
            factor = 0.9
        elif 18 <= h <= 22:
            factor = 1.5
        else:
            factor = 0.8
        hourly_demand.append(base_demand * factor)

    # Normalize demand to match total_daily_demand_kwh
    d_sum = sum(hourly_demand)
    hourly_demand = [(d / d_sum) * total_daily_demand_kwh for d in hourly_demand]

    # Simulation step
    soc = battery.initial_charge_kwh
    soc_history = []
    grid_import = 0.0
    grid_export = 0.0
    direct_self_consumed = 0.0

    for h in range(24):
        gen = hourly_pv[h]
        dem = hourly_demand[h]

        if gen >= dem:
            direct_self_consumed += dem
            surplus = gen - dem

            # Charge battery
            charge_possible = min(surplus, battery.max_charge_rate_kw)
            space_in_battery = battery.capacity_kwh - soc
            actual_charge = min(charge_possible, space_in_battery / battery.roundtrip_efficiency)
            soc += actual_charge * battery.roundtrip_efficiency
            grid_export += surplus - actual_charge
        else:
            direct_self_consumed += gen
            deficit = dem - gen

            # Discharge battery
            discharge_possible = min(deficit, battery.max_discharge_rate_kw)
            actual_discharge = min(discharge_possible, soc)
            soc -= actual_discharge
            grid_import += deficit - actual_discharge

        soc_history.append(round(soc, 2))

    total_local_used = direct_self_consumed + (
        battery.initial_charge_kwh - soc + sum(hourly_pv) - grid_export - direct_self_consumed
    )
    total_local_used = min(total_daily_demand_kwh, max(0.0, total_daily_demand_kwh - grid_import))

    ssr = (total_local_used / max(1.0, total_daily_demand_kwh)) * 100.0
    scr = (total_local_used / max(1.0, pv_total)) * 100.0

    return SolarMicrogridResult(
        total_pv_generation_kwh=pv_total,
        total_building_demand_kwh=total_daily_demand_kwh,
        grid_import_kwh=grid_import,
        grid_export_kwh=grid_export,
        battery_energy_stored_kwh=soc,
        solar_self_consumption_pct=min(100.0, scr),
        self_sufficiency_ratio_pct=min(100.0, ssr),
        hourly_state_of_charge=soc_history,
    )
