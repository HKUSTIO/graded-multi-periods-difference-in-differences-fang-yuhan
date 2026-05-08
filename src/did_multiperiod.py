from __future__ import annotations

import math

import numpy as np
import pandas as pd


def logistic(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-float(x))))


def generate_panel_data(config: dict, heterogeneous_trend: bool) -> pd.DataFrame:
    rng = np.random.default_rng(int(config["seed_population"]))
    n_units = int(config["n_units"])
    t0 = int(config["pre_periods"])
    t_total = t0 + int(config["post_periods"])
    times = np.arange(1, t_total + 1)

    x_draw = rng.uniform(0.0, 1.0, size=n_units)
    x1 = (x_draw >= 0.3).astype(int)
    x2 = (x_draw >= 0.7).astype(int)
    u = rng.uniform(0.0, 1.0, size=n_units)

    alpha0 = float(config["adoption_intercept"])
    alpha1 = float(config["adoption_slope_middle"])
    alpha2 = float(config["adoption_slope_late"])
    p1 = logistic(alpha0)
    p2 = np.array([logistic(alpha0 + alpha1 * value) for value in x1])
    p3 = np.array([logistic(alpha0 + alpha1 * value) for value in x1 + x2])
    p4 = np.array([logistic(alpha0 + alpha2 * value) for value in x1 + x2])

    cohort = np.zeros(n_units, dtype=int)
    cohort[u <= p1] = t0 + 1
    cohort[(u > p1) & (u <= p2)] = t0 + 2
    cohort[(u > p2) & (u <= p3)] = t0 + 3
    cohort[(u > p3) & (u <= p4)] = t0 + 4

    individual_effect = rng.normal(0.0, float(config["individual_sd"]), size=n_units)
    tau_i = rng.normal(float(config["tau_mean"]), float(config["tau_sd"]), size=n_units)
    time_shocks = rng.uniform(float(config["tau_time_low"]), float(config["tau_time_high"]), size=t_total + 1)

    rows = []
    for time in times:
        if heterogeneous_trend:
            trend_cfg = config["heterogeneous_trend"]
            trend = (
                (time / t_total) * float(trend_cfg["baseline_slope"]) * (1 - x1 - x2)
                + (time / t_total) * float(trend_cfg["x1_slope"]) * x1
                + (time / t_total) * float(trend_cfg["x2_slope"]) * x2
            )
        else:
            trend = np.full(n_units, time / t_total)

        error = rng.normal(0.0, float(config["error_sd"]), size=n_units)
        treated_ever = (cohort > 0).astype(int)
        y0 = float(config["base_level"]) + treated_ever * (-individual_effect) + (1 - treated_ever) * individual_effect + trend + error

        d = ((cohort > 0) & (time >= cohort)).astype(int)
        multiplier = np.zeros(n_units)
        multiplier[cohort == t0 + 1] = 1.0
        multiplier[cohort == t0 + 2] = -2.5
        multiplier[cohort == t0 + 3] = -1.75
        multiplier[cohort == t0 + 4] = -1.0
        tau_it = time_shocks[time] * np.abs(tau_i) * multiplier
        y = y0 + d * tau_it
        relative_time = np.where(cohort > 0, time - cohort, 0)

        for idx in range(n_units):
            rows.append(
                {
                    "id": idx + 1,
                    "x1": int(x1[idx]),
                    "x2": int(x2[idx]),
                    "cohort": int(cohort[idx]),
                    "time": int(time),
                    "relative_time": int(relative_time[idx]),
                    "d": int(d[idx]),
                    "y0": float(y0[idx]),
                    "tau_it": float(tau_it[idx]),
                    "y": float(y[idx]),
                }
            )

    return pd.DataFrame(rows, columns=["id", "x1", "x2", "cohort", "time", "relative_time", "d", "y0", "tau_it", "y"])


def summarize_group_shares_and_att(data: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per treated cohort and one row for all treated observations.
    """
    n_units = data["id"].nunique()
    treated_unit_cohorts = sorted(int(c) for c in data.loc[data["cohort"] > 0, "cohort"].unique())

    rows: list[dict[str, float | str]] = []
    for cohort in treated_unit_cohorts:
        cohort_mask = data["cohort"] == cohort
        cohort_share = float(cohort_mask.mean())

        cohort_treated = data.loc[cohort_mask & (data["d"] == 1), "tau_it"]
        cohort_att = float(cohort_treated.mean()) if not cohort_treated.empty else float("nan")

        rows.append({"group": f"cohort_{cohort}", "fraction": cohort_share, "att": cohort_att})

    treated_rows = data.loc[data["d"] == 1, "tau_it"]
    all_treated_fraction = float(data["d"].mean())
    all_treated_att = float(treated_rows.mean()) if not treated_rows.empty else float("nan")
    rows.append({"group": "all_treated", "fraction": all_treated_fraction, "att": all_treated_att})

    summary = pd.DataFrame(rows, columns=["group", "fraction", "att"])
    if n_units == 0:
        return pd.DataFrame(columns=["group", "fraction", "att"])
    return summary


def estimate_cohort_did(data: pd.DataFrame, cohort: int, event_time: int, control_group: str) -> float:
    """
    Return a two-period DID estimate for one treatment cohort and event time.
    """
    if control_group not in {"never", "notyet"}:
        raise ValueError("control_group must be either 'never' or 'notyet'.")

    target_time = int(cohort + event_time)
    baseline_time = int(cohort - 1)

    treated_base = data.loc[(data["cohort"] == cohort) & (data["time"] == baseline_time), "y"]
    treated_target = data.loc[(data["cohort"] == cohort) & (data["time"] == target_time), "y"]

    if treated_base.empty or treated_target.empty:
        raise ValueError("Missing treated-group observations for requested cohort/event_time.")

    if control_group == "never":
        control_mask = data["cohort"] == 0
    else:
        control_mask = (data["cohort"] == 0) | (data["cohort"] > target_time)

    control_base = data.loc[control_mask & (data["time"] == baseline_time), "y"]
    control_target = data.loc[control_mask & (data["time"] == target_time), "y"]

    if control_base.empty or control_target.empty:
        raise ValueError("Missing control-group observations for requested cohort/event_time.")

    treated_change = float(treated_target.mean() - treated_base.mean())
    control_change = float(control_target.mean() - control_base.mean())
    return treated_change - control_change


def estimate_event_study(data: pd.DataFrame, event_times: list[int], control_group: str) -> pd.DataFrame:
    """
    Return cohort-event DID estimates.
    """
    min_time = int(data["time"].min())
    max_time = int(data["time"].max())
    cohorts = sorted(int(c) for c in data.loc[data["cohort"] > 0, "cohort"].unique())

    rows = []
    for cohort in cohorts:
        for event_time in event_times:
            target_time = cohort + int(event_time)
            if target_time < min_time or target_time > max_time:
                continue
            estimate = estimate_cohort_did(data, cohort=cohort, event_time=int(event_time), control_group=control_group)
            rows.append({"cohort": int(cohort), "event_time": int(event_time), "estimate": float(estimate)})

    out = pd.DataFrame(rows, columns=["cohort", "event_time", "estimate"])
    if out.empty:
        return out
    return out.sort_values(["cohort", "event_time"]).reset_index(drop=True)


def aggregate_post_treatment_effects(event_study: pd.DataFrame) -> float:
    """
    Return the average estimate over post-treatment event times.
    """
    post = event_study.loc[event_study["event_time"] >= 0, "estimate"]
    if post.empty:
        return float("nan")
    return float(post.mean())


def estimate_twfe_coefficient(data: pd.DataFrame) -> float:
    """
    Return the coefficient from a residualized two-way fixed effects regression of y on d.
    """
    unit_mean = data.groupby("id")[["y", "d"]].transform("mean")
    time_mean = data.groupby("time")[["y", "d"]].transform("mean")
    overall_mean = data[["y", "d"]].mean()

    y_tilde = data["y"] - unit_mean["y"] - time_mean["y"] + float(overall_mean["y"])
    d_tilde = data["d"] - unit_mean["d"] - time_mean["d"] + float(overall_mean["d"])

    denominator = float((d_tilde**2).sum())
    if np.isclose(denominator, 0.0):
        return float("nan")
    numerator = float((d_tilde * y_tilde).sum())
    return numerator / denominator
