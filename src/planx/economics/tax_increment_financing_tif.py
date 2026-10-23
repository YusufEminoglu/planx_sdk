# -*- coding: utf-8 -*-
"""Tax Increment Financing (TIF) & Municipal Bond Yield Engine for PlanX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class PropertyTaxAssessment:
    property_id: str
    base_assessed_value_usd: float
    current_assessed_value_usd: float
    annual_appreciation_rate_pct: float = 4.5
    tax_rate_pct: float = 1.8  # 1.8% annual property tax


@dataclass
class TIFDistrictResult:
    total_base_tax_revenue_usd: float
    total_incremental_tax_captured_usd: float
    net_present_value_usd: float
    bond_debt_service_coverage_ratio: float
    annual_cashflows: list[dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result to a JSON-ready dictionary."""
        return {
            "base_revenue_usd": round(self.total_base_tax_revenue_usd, 2),
            "incremental_captured_usd": round(self.total_incremental_tax_captured_usd, 2),
            "npv_usd": round(self.net_present_value_usd, 2),
            "dscr": round(self.bond_debt_service_coverage_ratio, 2),
        }


def simulate_tif_district_cashflows(
    properties: Sequence[PropertyTaxAssessment | dict[str, Any]],
    lifespan_years: int = 20,
    annual_bond_debt_service_usd: float = 250000.0,
    discount_rate_pct: float = 5.0,
) -> TIFDistrictResult:
    """Simulate 20-30 year municipal TIF district incremental property tax capture
    and debt service coverage.
    """
    props: list[PropertyTaxAssessment] = []
    for p in properties:
        if isinstance(p, dict):
            props.append(
                PropertyTaxAssessment(
                    property_id=str(p.get("id", "PROP")),
                    base_assessed_value_usd=float(p.get("base_value", 1000000.0)),
                    current_assessed_value_usd=float(p.get("current_value", 1000000.0)),
                    annual_appreciation_rate_pct=float(p.get("appreciation_pct", 4.5)),
                    tax_rate_pct=float(p.get("tax_rate_pct", 1.8)),
                )
            )
        else:
            props.append(p)

    tot_base_val = sum(pr.base_assessed_value_usd for pr in props)
    avg_tax_rate = sum(pr.tax_rate_pct for pr in props) / max(1, len(props)) / 100.0

    tot_base_rev = 0.0
    tot_inc_captured = 0.0
    npv = 0.0
    cashflows = []

    cum_dscr_sum = 0.0

    for yr in range(1, lifespan_years + 1):
        # Escalate assessed value
        yr_val = sum(
            pr.current_assessed_value_usd * ((1.0 + pr.annual_appreciation_rate_pct / 100.0) ** yr)
            for pr in props
        )
        base_rev = tot_base_val * avg_tax_rate
        total_rev = yr_val * avg_tax_rate
        inc_rev = max(0.0, total_rev - base_rev)

        tot_base_rev += base_rev
        tot_inc_captured += inc_rev

        # Discounted Cashflow (DCF)
        pv = inc_rev / ((1.0 + discount_rate_pct / 100.0) ** yr)
        npv += pv

        dscr_yr = inc_rev / max(1.0, annual_bond_debt_service_usd)
        cum_dscr_sum += dscr_yr

        cashflows.append(
            {
                "year": yr,
                "assessed_value": round(yr_val, 2),
                "base_tax": round(base_rev, 2),
                "incremental_tax": round(inc_rev, 2),
                "dscr": round(dscr_yr, 2),
            }
        )

    avg_dscr = cum_dscr_sum / lifespan_years if lifespan_years > 0 else 0.0

    return TIFDistrictResult(
        total_base_tax_revenue_usd=tot_base_rev,
        total_incremental_tax_captured_usd=tot_inc_captured,
        net_present_value_usd=npv,
        bond_debt_service_coverage_ratio=avg_dscr,
        annual_cashflows=cashflows,
    )
