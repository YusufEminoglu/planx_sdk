# -*- coding: utf-8 -*-
"""Urban Heat Island (UHI) Mitigation & Microclimate Albedo/Shade Simulator for PlanX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class UHIMitigationResult:
    baseline_surface_temp_c: float
    mitigated_surface_temp_c: float
    surface_temp_reduction_c: float
    baseline_air_temp_c: float
    mitigated_air_temp_c: float
    air_temp_reduction_c: float
    cool_roof_area_m2: float
    canopy_shade_area_m2: float
    permeable_pavement_area_m2: float
    annual_energy_savings_kwh: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_surface_temp_c": round(self.baseline_surface_temp_c, 2),
            "mitigated_surface_temp_c": round(self.mitigated_surface_temp_c, 2),
            "surface_temp_reduction_c": round(self.surface_temp_reduction_c, 2),
            "air_temp_reduction_c": round(self.air_temp_reduction_c, 2),
            "annual_energy_savings_kwh": round(self.annual_energy_savings_kwh, 1),
        }


def simulate_uhi_temperature_reduction(
    urban_area_m2: float = 100000.0,
    roof_area_m2: float = 30000.0,
    pavement_area_m2: float = 40000.0,
    cool_roof_fraction: float = 0.50,  # 50% high-albedo cool roofs
    tree_canopy_fraction: float = 0.25,  # 25% tree canopy shade
    permeable_pavement_fraction: float = 0.30,  # 30% permeable evaporative pavement
    ambient_air_temp_c: float = 34.0,  # Peak summer heatwave
    solar_irradiance_w_m2: float = 850.0,
) -> UHIMitigationResult:
    """Simulate surface and 2m ambient air temperature reduction from urban
    cooling interventions.
    """
    # Baseline surface temperatures: T_surf = T_air + (1 - alpha) * I / h_c
    h_c = 20.0  # Convective heat transfer coefficient (W/m2.K)
    base_alpha_roof = 0.15
    base_alpha_pave = 0.10
    cool_alpha_roof = 0.70

    base_t_roof = ambient_air_temp_c + ((1.0 - base_alpha_roof) * solar_irradiance_w_m2) / h_c
    base_t_pave = ambient_air_temp_c + ((1.0 - base_alpha_pave) * solar_irradiance_w_m2) / h_c

    # Weighted baseline surface temp
    base_surf_temp = (base_t_roof * roof_area_m2 + base_t_pave * pavement_area_m2) / max(
        1.0, (roof_area_m2 + pavement_area_m2)
    )

    # Mitigated surface temperatures
    cool_roof_area = roof_area_m2 * cool_roof_fraction
    mit_t_cool_roof = ambient_air_temp_c + ((1.0 - cool_alpha_roof) * solar_irradiance_w_m2) / h_c
    mit_t_std_roof = base_t_roof

    shaded_pave_area = pavement_area_m2 * tree_canopy_fraction
    perm_pave_area = pavement_area_m2 * permeable_pavement_fraction * (1.0 - tree_canopy_fraction)
    unmit_pave_area = max(0.0, pavement_area_m2 - shaded_pave_area - perm_pave_area)

    mit_t_shaded_pave = ambient_air_temp_c + 2.5  # Heavy shade drops surface temp near air temp
    mit_t_perm_pave = base_t_pave - 6.0  # Evaporative cooling effect
    mit_t_std_pave = base_t_pave

    total_mit_heat = (
        cool_roof_area * mit_t_cool_roof
        + (roof_area_m2 - cool_roof_area) * mit_t_std_roof
        + shaded_pave_area * mit_t_shaded_pave
        + perm_pave_area * mit_t_perm_pave
        + unmit_pave_area * mit_t_std_pave
    )
    mit_surf_temp = total_mit_heat / max(1.0, (roof_area_m2 + pavement_area_m2))

    surf_reduction = max(0.0, base_surf_temp - mit_surf_temp)

    # 2m Air Temperature Reduction: empirical Akbari & Pomerantz rule,
    # ~0.18 °C air drop per 1.0 °C surface drop
    air_reduction = surf_reduction * 0.18
    base_air = ambient_air_temp_c
    mit_air = base_air - air_reduction

    # Estimated annual cooling energy savings (~4.2 kWh per m2 cool roof + 2.5 kWh per m2 shade)
    energy_savings = cool_roof_area * 4.2 + shaded_pave_area * 2.5

    return UHIMitigationResult(
        baseline_surface_temp_c=base_surf_temp,
        mitigated_surface_temp_c=mit_surf_temp,
        surface_temp_reduction_c=surf_reduction,
        baseline_air_temp_c=base_air,
        mitigated_air_temp_c=mit_air,
        air_temp_reduction_c=air_reduction,
        cool_roof_area_m2=cool_roof_area,
        canopy_shade_area_m2=shaded_pave_area,
        permeable_pavement_area_m2=perm_pave_area,
        annual_energy_savings_kwh=energy_savings,
    )
