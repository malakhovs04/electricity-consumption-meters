import pandas as pd
import numpy as np
from scipy.stats import entropy


class PeriodFeatureExtractor:
    """
    Делит временной ряд каждого счётчика на N равных периодов по времени
    и считает тот же набор признаков, что AnomalyFeatureExtractor +
    профильные метрики (night_day_ratio, max_gap_hours, large_gap_ratio)
    для каждого периода отдельно.

    По умолчанию n_periods=2 — первая и вторая половина наблюдений.
    Это позволяет сравнить "раньше" и "сейчас" для каждого счётчика
    и выявить, у каких объектов профиль потребления изменился
    (потенциальный ранний сигнал аномалии).

    Параметры
    ----------
    n_periods : int
        Число равных периодов. 2 = первая/вторая половина.
    min_measurements : int
        Минимум измерений в периоде. Периоды с меньшим числом
        отбрасываются (частые разрывы данных или короткий ряд).
    """

    def __init__(self, n_periods: int = 2, min_measurements: int = 10):
        self.n_periods = n_periods
        self.min_measurements = min_measurements

    def _extract_period_features(self, group: pd.DataFrame) -> dict:
        signal = group["A+"].values
        signal = signal[~np.isnan(signal)]

        if len(signal) == 0:
            return None

        mean_load = np.mean(signal)
        std_load = np.std(signal)
        max_load = np.max(signal)

        features = {
            "mean_load":   mean_load,
            "std_load":    std_load,
            "peak_factor": max_load / (mean_load + 1e-6),
            "load_factor": mean_load / (max_load + 1e-6),
            "cv":          std_load / (mean_load + 1e-6),
            "zero_ratio":  float(np.mean(signal == 0)),
            "entropy":     float(entropy(np.histogram(signal, bins=15)[0] + 1)),
            "n_measurements": len(group),
        }

        # ---- ночное/дневное соотношение ----
        hours = group["timestamp"].dt.hour
        night_mean = group.loc[hours.between(0, 5), "A+"].mean()
        day_mean   = group.loc[hours.between(8, 22), "A+"].mean()
        night_mean = 0.0 if pd.isna(night_mean) else float(night_mean)
        day_mean   = 0.0 if pd.isna(day_mean)   else float(day_mean)
        features["night_day_ratio"] = night_mean / (day_mean + 1e-6)

        # ---- разрывы во времени ----
        gaps = (
            group["timestamp"]
            .diff()
            .dt.total_seconds()
            .div(3600)
        )
        valid_gaps = gaps.dropna()
        max_gap = valid_gaps.max() if len(valid_gaps) else 0.0
        features["max_gap_hours"]    = 0.0 if pd.isna(max_gap) else float(max_gap)
        features["large_gap_ratio"]  = float(
            (valid_gaps > 4).sum() / max(len(valid_gaps), 1)
        )

        return features

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        rows = []

        for meter_id, group in df.groupby("meter_id"):

            group = group.sort_values("timestamp").reset_index(drop=True)

            consumer_class = (
                group["consumer_class"].iloc[0]
                if "consumer_class" in group.columns else -1
            )

            total_duration = (
                group["timestamp"].iloc[-1] - group["timestamp"].iloc[0]
            )
            period_duration = total_duration / self.n_periods

            t_start = group["timestamp"].iloc[0]

            for p in range(self.n_periods):

                p_start = t_start + p * period_duration
                p_end   = t_start + (p + 1) * period_duration

                # последний период включает правую границу
                if p < self.n_periods - 1:
                    mask = (group["timestamp"] >= p_start) & (group["timestamp"] < p_end)
                else:
                    mask = (group["timestamp"] >= p_start) & (group["timestamp"] <= p_end)

                period_group = group[mask]

                if len(period_group) < self.min_measurements:
                    continue

                features = self._extract_period_features(period_group)

                if features is None:
                    continue

                features["meter_id"]       = meter_id
                features["consumer_class"] = consumer_class
                features["period_idx"]     = p
                features["period_label"]   = f"period_{p}"
                features["period_start"]   = p_start
                features["period_end"]     = p_end

                rows.append(features)

        col_order = [
            "meter_id", "consumer_class", "period_idx", "period_label",
            "period_start", "period_end", "n_measurements",
            "mean_load", "std_load", "peak_factor", "load_factor",
            "cv", "zero_ratio", "entropy",
            "night_day_ratio", "max_gap_hours", "large_gap_ratio",
        ]

        result = pd.DataFrame(rows)
        existing_cols = [c for c in col_order if c in result.columns]
        return result[existing_cols]