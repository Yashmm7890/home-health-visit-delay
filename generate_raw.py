from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(4127)
    n = 7200

    regions = np.array(["north", "south", "east", "west", "central", "rural"])
    service_types = np.array(["assessment", "wound_care", "infusion", "therapy", "medication", "lab_draw"])
    clinician_types = np.array(["RN", "LPN", "PT", "OT", "aide"])
    zip_clusters = np.array([f"zc_{i:02d}" for i in range(18)])
    route_zones = np.array(["A", "B", "C", "D", "E"])
    note_topics = np.array(["gate_code", "mobility", "pets", "translator", "parking", "supplies", "family_present"])

    start = np.datetime64("2025-01-01")
    days = rng.integers(0, 365, size=n)
    dates = start + days.astype("timedelta64[D]")
    weekday = pd.to_datetime(dates).weekday.to_numpy()

    hour = rng.choice(np.arange(7, 19), size=n, p=np.array([4, 5, 7, 8, 9, 10, 10, 11, 10, 10, 9, 7]) / 100)
    minute = rng.choice([0, 15, 30, 45], size=n)
    scheduled_time = np.array([f"{h:02d}:{m:02d}" for h, m in zip(hour, minute)])

    region = rng.choice(regions, size=n, p=[0.18, 0.17, 0.18, 0.16, 0.19, 0.12])
    service_type = rng.choice(service_types, size=n, p=[0.25, 0.16, 0.10, 0.18, 0.20, 0.11])
    clinician_type = rng.choice(clinician_types, size=n, p=[0.34, 0.24, 0.16, 0.11, 0.15])
    zip_cluster = rng.choice(zip_clusters, size=n)
    route_zone = rng.choice(route_zones, size=n, p=[0.24, 0.22, 0.20, 0.19, 0.15])

    patient_acuity = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.19, 0.28, 0.27, 0.18, 0.08])
    miles_from_branch = np.round(rng.gamma(2.4, 4.8, size=n) + (region == "rural") * rng.gamma(2.2, 4.0, size=n), 2)
    prior_visit_count = rng.poisson(5.5, size=n)
    prior_late_rate = np.clip(rng.beta(1.7, 5.2, size=n) + (region == "rural") * 0.05, 0, 0.95)
    prior_late_rate[prior_visit_count == 0] = np.nan
    clinician_daily_load = np.clip(rng.poisson(5.3, size=n) + (weekday >= 4), 1, 11)
    previous_visit_gap_minutes = rng.normal(42, 18, size=n)
    previous_visit_gap_minutes[hour <= 8] = np.nan
    previous_visit_gap_minutes = np.round(np.clip(previous_visit_gap_minutes, 5, 120), 1)

    avg_route_speed_mph = np.round(
        np.clip(rng.normal(29, 6, size=n) - (region == "central") * 4 - (region == "rural") * 3, 9, 52),
        1,
    )
    rush = ((hour >= 8) & (hour <= 10)) | ((hour >= 15) & (hour <= 17))
    traffic_index = np.clip(rng.normal(38, 14, size=n) + rush * 18 + (region == "central") * 10, 0, 100)
    precipitation_mm = np.round(rng.gamma(0.6, 3.0, size=n) * (rng.random(n) < 0.38), 2)
    temperature_f = np.round(58 + 22 * np.sin((days - 80) / 365 * 2 * np.pi) + rng.normal(0, 8, size=n), 1)
    is_holiday_week = np.isin(days, np.r_[0:7, 48:55, 178:185, 327:334, 356:365]).astype(int)
    authorization_required = rng.binomial(1, 0.18 + 0.05 * (service_type == "infusion"), size=n)
    supply_complexity = np.clip(rng.poisson(0.8, size=n) + (service_type == "wound_care") + (service_type == "infusion"), 0, 3)

    topic_strings = []
    for i in range(n):
        probs = np.array([0.12, 0.18, 0.09, 0.08, 0.20, 0.13, 0.10])
        probs += (service_type[i] == "infusion") * np.array([0, 0, 0, 0, 0.20, 0.25, 0])
        probs += (patient_acuity[i] >= 4) * np.array([0, 0.10, 0, 0.06, 0, 0.05, 0.12])
        picked = note_topics[rng.random(len(note_topics)) < probs]
        topic_strings.append("|".join(picked) if len(picked) else "none")

    region_effect = pd.Series(region).map({"north": 2, "south": -1, "east": 1, "west": 0, "central": 4, "rural": 6}).to_numpy()
    service_effect = pd.Series(service_type).map(
        {"assessment": 0, "wound_care": 5, "infusion": 8, "therapy": 3, "medication": 1, "lab_draw": -1}
    ).to_numpy()
    zone_effect = pd.Series(route_zone).map({"A": -2, "B": 0, "C": 2, "D": 3, "E": 5}).to_numpy()

    gap_penalty = np.nan_to_num((35 - previous_visit_gap_minutes) / 3.5, nan=0)
    prior_effect = np.nan_to_num(prior_late_rate * 18, nan=3.5)
    weather_effect = np.maximum(0, precipitation_mm - 2) * 1.2 + (temperature_f < 25) * 4 + (temperature_f > 92) * 2
    workload_effect = np.maximum(0, clinician_daily_load - 5) * 3.2
    travel_effect = miles_from_branch / np.maximum(avg_route_speed_mph, 1) * 18 + traffic_index * 0.11
    topic_effect = np.array([
        2.5 * ("parking" in s) + 2.0 * ("gate_code" in s) + 3.0 * ("supplies" in s) + 2.0 * ("translator" in s)
        for s in topic_strings
    ])

    latent = (
        -8
        + region_effect
        + service_effect
        + zone_effect
        + patient_acuity * 1.8
        + workload_effect
        + travel_effect
        + weather_effect
        + prior_effect
        + gap_penalty
        + topic_effect
        + rush * 4.5
        + (weekday >= 5) * 3.5
        + is_holiday_week * 5.0
        + authorization_required * 3.0
        + supply_complexity * 2.5
        + rng.normal(0, 8, size=n)
    )
    delay_minutes = np.round(np.clip(latent + rng.gamma(1.3, 3.0, size=n), 0, 180), 1)

    arrival_minutes = hour * 60 + minute + delay_minutes
    actual_arrival_time = np.array([f"{int(x // 60) % 24:02d}:{int(x % 60):02d}" for x in arrival_minutes])
    dispatcher_adjustment_minutes = np.round(delay_minutes * rng.uniform(0.2, 0.55, size=n) + rng.normal(0, 2, size=n), 1)
    same_day_complaint_flag = (delay_minutes > rng.normal(36, 8, size=n)).astype(int)

    data = pd.DataFrame(
        {
            "visit_id": [f"visit_{i:05d}" for i in range(n)],
            "scheduled_date": pd.to_datetime(dates).strftime("%Y-%m-%d"),
            "scheduled_time": scheduled_time,
            "weekday": weekday,
            "region": region,
            "service_type": service_type,
            "clinician_type": clinician_type,
            "patient_acuity": patient_acuity,
            "miles_from_branch": miles_from_branch,
            "prior_visit_count": prior_visit_count,
            "prior_late_rate": np.round(prior_late_rate, 3),
            "clinician_daily_load": clinician_daily_load,
            "previous_visit_gap_minutes": previous_visit_gap_minutes,
            "avg_route_speed_mph": avg_route_speed_mph,
            "traffic_index": np.round(traffic_index, 1),
            "precipitation_mm": precipitation_mm,
            "temperature_f": temperature_f,
            "is_holiday_week": is_holiday_week,
            "authorization_required": authorization_required,
            "supply_complexity": supply_complexity,
            "zip_cluster": zip_cluster,
            "route_zone": route_zone,
            "notes_topic": topic_strings,
            "delay_minutes": delay_minutes,
            "actual_arrival_time": actual_arrival_time,
            "dispatcher_adjustment_minutes": dispatcher_adjustment_minutes,
            "same_day_complaint_flag": same_day_complaint_flag,
        }
    )

    out = Path(__file__).resolve().parent / "data.csv"
    data.to_csv(out, index=False)
    print(f"wrote {out} with {len(data)} rows")


if __name__ == "__main__":
    main()

