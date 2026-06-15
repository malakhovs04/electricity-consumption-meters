import pandas as pd
import numpy as np


class ConsumptionProfileBuilder:
    def build(self, df):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        rows = []
        for meter_id, group in df.groupby("meter_id"):
            group = group.sort_values("timestamp")
            consumer_class = group["consumer_class"].iloc[0]
            timestamps = group["timestamp"]
            gaps = (timestamps.diff().dt.total_seconds().div(3600))

            mean_gap = gaps.mean()
            max_gap = gaps.max()
            valid_gaps = gaps.dropna()
            large_gap_ratio = ((valid_gaps > 4).sum()/ max(len(valid_gaps), 1))

            group["hour"] = group["timestamp"].dt.hour
            night = group[group["hour"].between(0, 5)]["A+"]
            day = group[group["hour"].between(8, 22)]["A+"]
            night_mean = night.mean()
            day_mean = day.mean()
            night_mean = 0.0 if pd.isna(night_mean) else night_mean
            day_mean = 0.0 if pd.isna(day_mean) else day_mean
            night_ratio = night_mean / (day_mean + 1e-6)

            group["weekday"] = (group["timestamp"].dt.weekday)
            row = {"meter_id": meter_id,
                "consumer_class": consumer_class,
                "n_measurements": len(group),
                "mean_gap_hours": mean_gap if not pd.isna(mean_gap) else 0.0,
                "max_gap_hours": max_gap if not pd.isna(max_gap) else 0.0,
                "large_gap_ratio": large_gap_ratio,
                "night_day_ratio": night_ratio}

            hourly_profile = (group.groupby("hour")["A+"].mean())
            for h in range(24):
                row[f"hour_{h}"] = (hourly_profile.get(h, np.nan))

            weekday_profile = (group.groupby("weekday")["A+"].mean())

            for d in range(7):
                row[f"weekday_{d}"] = (weekday_profile.get(d, np.nan))
            rows.append(row)
        profiles_df = pd.DataFrame(rows)
        return profiles_df