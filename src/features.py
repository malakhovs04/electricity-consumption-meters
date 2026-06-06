import pandas as pd
import numpy as np


class FeatureExtractor:

    def __init__(self):
        pass

    def extract_statistical_features(self, signal, prefix):
        features = {}

        signal = np.asarray(signal, dtype=float)
        signal = signal[~np.isnan(signal)]

        if len(signal) == 0:
            return {f"{prefix}_mean": 0,
                f"{prefix}_std": 0,
                f"{prefix}_min": 0,
                f"{prefix}_max": 0,
                f"{prefix}_median": 0,
                f"{prefix}_p95": 0,
                f"{prefix}_p5": 0,
                f"{prefix}_cv": 0,}

        mean = np.mean(signal)
        std = np.std(signal)

        features[f"{prefix}_mean"] = mean
        features[f"{prefix}_std"] = std
        features[f"{prefix}_min"] = np.min(signal)
        features[f"{prefix}_max"] = np.max(signal)
        features[f"{prefix}_median"] = np.median(signal)
        features[f"{prefix}_p95"] = np.percentile(signal, 95)
        features[f"{prefix}_p5"] = np.percentile(signal, 5)
        features[f"{prefix}_cv"] = std / (mean + 1e-6)

        return features

    def extract_frequency_features(self, signal, prefix=""):
        features = {}

        signal = np.asarray(signal, dtype=float)
        signal = signal[~np.isnan(signal)]

        if len(signal) < 2:
            return {f"{prefix}_fft_max": 0,
                f"{prefix}_fft_mean": 0,
                f"{prefix}_fft_energy": 0,}

        spectrum = np.fft.fft(signal)
        spectrum = np.abs(spectrum)

        features[f"{prefix}_fft_max"] = np.max(spectrum)
        features[f"{prefix}_fft_mean"] = np.mean(spectrum)
        features[f"{prefix}_fft_energy"] = np.sum(spectrum ** 2)

        return features

    def extract_electrical_features(self, active_signal, reactive_signal):
        features = {}

        active_signal = np.asarray(active_signal, dtype=float)
        reactive_signal = np.asarray(reactive_signal, dtype=float)

        active_signal = active_signal[~np.isnan(active_signal)]
        reactive_signal = reactive_signal[~np.isnan(reactive_signal)]

        min_len = min(len(active_signal), len(reactive_signal))

        if min_len == 0:
            return {"apparent_power_mean": 0,
                "apparent_power_std": 0,
                "phase_angle_mean": 0,
                "phase_angle_std": 0,}

        active_signal = active_signal[:min_len]
        reactive_signal = reactive_signal[:min_len]

        complex_power = active_signal + 1j * reactive_signal

        apparent_power = np.abs(complex_power)
        phase_angle = np.angle(complex_power)

        features["apparent_power_mean"] = np.mean(apparent_power)
        features["apparent_power_std"] = np.std(apparent_power)
        features["phase_angle_mean"] = np.mean(phase_angle)
        features["phase_angle_std"] = np.std(phase_angle)

        return features

    def extract(self, df):
        rows = []

        grouped = df.groupby("meter_id")

        for meter_id, group in grouped:

            features = {"meter_id": meter_id,
                "consumer_class": group["consumer_class"].iloc[0]
                if "consumer_class" in group.columns else -1}

            for col in ["A+", "A-", "R+", "R-"]:
                signal = group[col].values

                features.update(
                    self.extract_statistical_features(signal, col))
                features.update(
                    self.extract_frequency_features(signal, col))

            features.update(
                self.extract_electrical_features(
                    group["A+"].values,
                    group["R+"].values))
            rows.append(features)

        return pd.DataFrame(rows)