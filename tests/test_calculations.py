from datetime import date, timedelta
from decimal import Decimal

import pandas as pd
import pytest

from watchdog_ai.calculations import Thresholds, add_baselines, detect, deviation, weekly_costs


def small_weekly(points):
    return pd.DataFrame([dict(origin=r, destination="X", route=r+"-X", route_type=t,
                              week_of=date.fromisoformat(w), cost_per_tonne_km=v)
                         for r, t, w, v in points])


@pytest.mark.parametrize("route,week,cost,hist,peer", [
    ("Delhi-Jaipur", "2024-11-11", "4.17", "35.5", "21.0"),
    ("Ahmedabad-Mumbai", "2025-01-20", "3.29", "29.5", "22.5"),
    ("Mumbai-Pune", "2025-09-15", "3.98", "9.2", "23.6"),
])
def test_official_regressions(shipments, route, week, cost, hist, peer):
    df = add_baselines(weekly_costs(shipments))
    row = df[(df.route == route) & (df.week_of == date.fromisoformat(week))].iloc[0]
    assert f"{row.cost_per_tonne_km:.2f}" == cost
    assert f"{row.history_pct:.1f}" == hist
    assert f"{row.peer_pct:.1f}" == peer


def test_weighted_totals_monday_sunday_and_input_order(shipments):
    frame = shipments.iloc[:3].copy()
    frame["origin"], frame["destination"], frame["route_type"] = "A", "B", "Short"
    frame["shipment_date"] = [date(2024,1,1), date(2024,1,7), date(2024,1,8)]
    frame["freight_cost_inr"], frame["quantity_tonnes"], frame["distance_km"] = [100,900,10], [1,3,1], [10,10,1]
    w = weekly_costs(frame)
    assert w.cost_per_tonne_km.tolist() == [25,10]
    assert w.shipment_count.tolist() == [2,1]
    pd.testing.assert_frame_equal(w, weekly_costs(frame.iloc[::-1]))


def test_eight_calendar_weeks_gaps_and_no_leakage():
    frame = small_weekly([("A","Short","2024-01-01",100), ("A","Short","2024-01-08",2),
                         ("A","Short","2024-02-26",4), ("A","Short","2024-03-04",9),
                         ("A","Short","2024-03-11",999)])
    row = add_baselines(frame).iloc[3]
    assert row.history_baseline == 3
    assert row.history_observations == 2
    frame.loc[3,"cost_per_tonne_km"] = 9999
    assert add_baselines(frame).iloc[3].history_baseline == 3
    assert add_baselines(frame).iloc[0].history_observations == 0


def test_peers_exclude_self_other_types_and_dates():
    frame = small_weekly([("A","Short","2024-01-01",100), ("B","Short","2024-01-01",2),
                         ("C","Short","2024-01-01",4), ("D","Long","2024-01-01",900),
                         ("B","Short","2024-01-08",800)])
    got = add_baselines(frame)
    assert got.iloc[0].peer_baseline == 3
    assert got.iloc[0].peer_observations == 2
    assert got.iloc[3].peer_observations == 0
    assert pd.isna(got.iloc[3].peer_pct)


def test_independent_decimal_reference_for_every_week(shipments):
    reference = {}
    for r in shipments.to_dict("records"):
        w = r["shipment_date"] - timedelta(days=r["shipment_date"].weekday())
        key = (r["origin"]+"-"+r["destination"], w)
        c, t = reference.get(key, (Decimal(0), Decimal(0)))
        reference[key] = c + Decimal(str(r["freight_cost_inr"])), t + Decimal(str(r["quantity_tonnes"])) * Decimal(str(r["distance_km"]))
    for r in weekly_costs(shipments).itertuples():
        c,t = reference[(r.route,r.week_of)]
        assert r.cost_per_tonne_km == pytest.approx(float(c/t), rel=1e-14)


def test_thresholds_and_missing_baselines():
    frame = pd.DataFrame({"history_pct":[20,19.99,-1,0,None,None,5],"peer_pct":[0,0,90,90,21,None,21]})
    assert detect(frame).is_anomaly.tolist() == [True,False,False,False,True,False,True]
    assert detect(frame, Thresholds(25,25)).is_anomaly.tolist() == [False]*7
    pd.testing.assert_frame_equal(detect(frame), detect(frame))
    assert deviation(1,0) is None
    for bad in [0, -1, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            Thresholds(bad,20)
