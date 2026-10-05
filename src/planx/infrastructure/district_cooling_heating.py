# -*- coding: utf-8 -*-
"""District Heating & Cooling (DHC) 4th Generation Thermal Energy Network Balancer for PlanX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class BuildingThermalDemandProfile:
    building_id: str
    peak_heating_demand_kw: float
    peak_cooling_demand_kw: float
    pipe_distance_from_plant_m: float
    floor_area_m2: float = 2500.0


@dataclass
class DHCNetworkResult:
    total_thermal_delivered_mwh_yr: float
    network_distribution_heat_loss_pct: float
    annual_co2_avoided_tonnes: float
    pumping_electrical_energy_kwh_yr: float
    overall_system_cop: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "thermal_delivered_mwh_yr": round(self.total_thermal_delivered_mwh_yr, 1),
            "distribution_loss_pct": round(self.network_distribution_heat_loss_pct, 1),
            "co2_avoided_tonnes": round(self.annual_co2_avoided_tonnes, 1),
            "pumping_kwh_yr": round(self.pumping_electrical_energy_kwh_yr, 1),
            "system_cop": round(self.overall_system_cop, 2),
        }


def simulate_dhc_thermal_network(
    consumer_buildings: Sequence[BuildingThermalDemandProfile | dict[str, Any]],
    supply_water_temp_c: float = 65.0,  # 4GDH Low-Temp supply
    return_water_temp_c: float = 35.0,
    ambient_ground_temp_c: float = 12.0,
    pipe_insulation_u_value_w_mk: float = 0.22,
) -> DHCNetworkResult:
    """Simulate low-temperature district energy network thermal losses,
    mass flow rates, and decarbonization yield.
    """
    buildings: list[BuildingThermalDemandProfile] = []
    for b in consumer_buildings:
        if isinstance(b, dict):
            buildings.append(
                BuildingThermalDemandProfile(
                    building_id=str(b.get("id", "BLDG")),
                    peak_heating_demand_kw=float(b.get("heating_kw", 100.0)),
                    peak_cooling_demand_kw=float(b.get("cooling_kw", 80.0)),
                    pipe_distance_from_plant_m=float(b.get("distance_m", 250.0)),
                    floor_area_m2=float(b.get("area_m2", 2000.0)),
                )
            )
        else:
            buildings.append(b)

    tot_heat_kw = sum(b.peak_heating_demand_kw for b in buildings)
    tot_cooling_kw = sum(b.peak_cooling_demand_kw for b in buildings)

    # Annual full-load equivalent hours (~1800h heating, ~1200h cooling)
    heat_mwh = (tot_heat_kw * 1800.0) / 1000.0
    cooling_mwh = (tot_cooling_kw * 1200.0) / 1000.0
    tot_thermal_delivered = heat_mwh + cooling_mwh

    # Thermal distribution losses along twin pre-insulated pipes
    delta_t_ground = ((supply_water_temp_c + return_water_temp_c) / 2.0) - ambient_ground_temp_c
    tot_pipe_len_m = sum(b.pipe_distance_from_plant_m for b in buildings) * 2.0  # supply + return
    pipe_loss_kw = (pipe_insulation_u_value_w_mk * tot_pipe_len_m * delta_t_ground) / 1000.0
    annual_loss_mwh = (pipe_loss_kw * 8760.0) / 1000.0

    loss_pct = (annual_loss_mwh / max(1.0, tot_thermal_delivered + annual_loss_mwh)) * 100.0

    # Pumping power: P_pump = (Q * rho * g * H) / eta
    delta_t_supply_return = max(5.0, supply_water_temp_c - return_water_temp_c)
    mass_flow_kg_s = (tot_heat_kw / (4.184 * delta_t_supply_return))
    pumping_kwh = mass_flow_kg_s * 0.08 * 3000.0  # Empirical pumping power

    # Decarbonization vs distributed gas boilers (emission factor difference ~0.18 kg CO2/kWh)
    co2_tonnes = (tot_thermal_delivered * 1000.0 * 0.18) / 1000.0

    system_cop = 4.2  # High efficiency heat pump / waste heat COP

    return DHCNetworkResult(
        total_thermal_delivered_mwh_yr=tot_thermal_delivered,
        network_distribution_heat_loss_pct=loss_pct,
        annual_co2_avoided_tonnes=co2_tonnes,
        pumping_electrical_energy_kwh_yr=pumping_kwh,
        overall_system_cop=system_cop,
    )
