# -*- coding: utf-8 -*-
"""Multi-Server EV Fast-Charging Station Queue & Waiting Time Simulator
(M/M/c/K Markov Chain) for PlanX.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class EVStationQueueConfig:
    num_fast_chargers_c: int = 6
    max_queue_capacity_k: int = 15  # Total station capacity (c in service + k-c waiting)
    mean_arrival_rate_evs_per_hour_lambda: float = 14.0  # lambda
    mean_charging_time_minutes_mu: float = 20.0  # ~3.0 EVs/hr service rate per charger mu


@dataclass
class EVChargingQueueResult:
    num_chargers: int
    charger_utilization_factor_rho: float
    blocking_rejection_probability_pk: float
    average_evs_in_queue_lq: float
    average_waiting_time_minutes_wq: float
    effective_throughput_evs_per_hour: float
    grid_peak_power_demand_kw: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "chargers": self.num_chargers,
            "utilization_rho": round(self.charger_utilization_factor_rho, 3),
            "blocking_prob_pk": round(self.blocking_rejection_probability_pk, 4),
            "queue_length_lq": round(self.average_evs_in_queue_lq, 2),
            "waiting_time_min": round(self.average_waiting_time_minutes_wq, 1),
            "throughput_evs_hr": round(self.effective_throughput_evs_per_hour, 1),
            "peak_grid_kw": round(self.grid_peak_power_demand_kw, 1),
        }


def simulate_ev_charging_station_queues(
    config: EVStationQueueConfig | None = None,
    charger_rated_power_kw: float = 150.0,  # 150 kW DC Fast Charger
) -> EVChargingQueueResult:
    """Compute steady-state M/M/c/K Markov birth-death queuing probabilities
    and waiting times for EV charging hubs.
    """
    cfg = config or EVStationQueueConfig()
    c = max(1, cfg.num_fast_chargers_c)
    k = max(c, cfg.max_queue_capacity_k)
    lam = max(0.1, cfg.mean_arrival_rate_evs_per_hour_lambda)
    mu = 60.0 / max(1.0, cfg.mean_charging_time_minutes_mu)  # Service rate per charger

    r = lam / mu  # Offered load
    rho = r / float(c)  # Traffic intensity

    # Compute p0 (idle probability) for M/M/c/K
    sum_terms = 0.0
    for n in range(c):
        sum_terms += (r ** n) / math.factorial(n)

    sum_queue_terms = 0.0
    for n in range(c, k + 1):
        sum_queue_terms += ((r ** n) / (math.factorial(c) * (c ** (n - c))))

    p0 = 1.0 / (sum_terms + sum_queue_terms)

    # State probability p_n
    p_k = (r ** k) / (math.factorial(c) * (c ** (k - c))) * p0

    # Average queue length Lq
    lq = 0.0
    for n in range(c + 1, k + 1):
        pn = (r ** n) / (math.factorial(c) * (c ** (n - c))) * p0
        lq += (n - c) * pn

    # Effective arrival rate lambda_eff = lambda * (1 - p_K)
    lam_eff = lam * (1.0 - p_k)
    # Little's Law: Wq = Lq / lambda_eff (hours -> minutes)
    wq_min = (lq / max(0.01, lam_eff)) * 60.0

    peak_kw = c * charger_rated_power_kw * min(1.0, rho * 1.15)

    return EVChargingQueueResult(
        num_chargers=c,
        charger_utilization_factor_rho=min(1.0, rho),
        blocking_rejection_probability_pk=p_k,
        average_evs_in_queue_lq=lq,
        average_waiting_time_minutes_wq=wq_min,
        effective_throughput_evs_per_hour=lam_eff,
        grid_peak_power_demand_kw=peak_kw,
    )
