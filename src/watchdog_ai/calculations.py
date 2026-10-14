"""Deterministic weekly totals, calendar baselines, and numeric anomaly detection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta

import pandas as pd


@dataclass(frozen=True)
class Thresholds:
    history_pct: float = 20.0
    peer_pct: float = 20.0

    def __post_init__(self) -> None:
        if any(not math.isfinite(v) or v <= 0 for v in (self.history_pct, self.peer_pct)):
            raise ValueError("Anomaly thresholds must be finite positive percentages")


def weekly_costs(shipments: pd.DataFrame) -> pd.DataFrame:
    work = shipments.sort_values("shipment_id").copy()
    work["week_of"] = work["shipment_date"].map(lambda d: d - timedelta(days=d.weekday()))
    work["tonne_km"] = work["quantity_tonnes"] * work["distance_km"]
    results = []
    for key, group in work.groupby(["origin", "destination", "route_type", "week_of"], sort=True):
        cost = math.fsum(group["freight_cost_inr"])
        tonne_km = math.fsum(group["tonne_km"])
        if not math.isfinite(cost) or not math.isfinite(tonne_km) or tonne_km <= 0:
            raise ValueError(f"Invalid financial totals for {key}")
        rate = cost / tonne_km
        if not math.isfinite(rate):
            raise ValueError(f"Nonfinite cost per tonne-km for {key}")
        results.append(dict(origin=key[0], destination=key[1], route_type=key[2], week_of=key[3],
                            route=f"{key[0]}-{key[1]}", freight_cost_inr=cost, tonne_km=tonne_km,
                            cost_per_tonne_km=rate, shipment_count=len(group),
                            shipment_dates=tuple(sorted(set(group["shipment_date"])))))
    return pd.DataFrame(results).sort_values(["route", "week_of"]).reset_index(drop=True)


def _mean(values: list[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def deviation(current: float, baseline: float | None) -> float | None:
    # A zero baseline has no defined percentage change; it is not 0% change.
    if baseline is None or not math.isfinite(baseline) or baseline <= 0:
        return None
    value = (current / baseline - 1) * 100
    if not math.isfinite(value):
        raise ValueError("Percentage comparison overflowed")
    return value


def add_baselines(weekly: pd.DataFrame) -> pd.DataFrame:
    rows = weekly.to_dict("records")
    by_route: dict[tuple, list[dict]] = {}
    by_type_week: dict[tuple, list[dict]] = {}
    for row in rows:
        by_route.setdefault((row["origin"], row["destination"], row["route_type"]), []).append(row)
        by_type_week.setdefault((row["route_type"], row["week_of"]), []).append(row)
    for row in rows:
        route = (row["origin"], row["destination"], row["route_type"])
        start = row["week_of"] - timedelta(weeks=8)
        history = [r["cost_per_tonne_km"] for r in by_route[route] if start <= r["week_of"] < row["week_of"]]
        peers = [r["cost_per_tonne_km"] for r in by_type_week[(row["route_type"], row["week_of"]) ]
                 if (r["origin"], r["destination"]) != (row["origin"], row["destination"])]
        row.update(history_baseline=_mean(history), peer_baseline=_mean(peers),
                   history_observations=len(history), peer_observations=len(peers))
        row["history_pct"] = deviation(row["cost_per_tonne_km"], row["history_baseline"])
        row["peer_pct"] = deviation(row["cost_per_tonne_km"], row["peer_baseline"])
    return pd.DataFrame(rows)


def detect(analysis: pd.DataFrame, thresholds: Thresholds = Thresholds()) -> pd.DataFrame:
    result = analysis.copy()
    result["is_rising"] = result["history_pct"].gt(0)
    result["unusual_history"] = result["history_pct"].ge(thresholds.history_pct)
    result["unusual_peers"] = result["peer_pct"].ge(thresholds.peer_pct)
    result["trend_unavailable"] = result["history_pct"].isna()
    result["is_anomaly"] = (result["is_rising"] | result["trend_unavailable"]) & (
        result["unusual_history"] | result["unusual_peers"])
    return result
