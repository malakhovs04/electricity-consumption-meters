import pandas as pd
import numpy as np
from scipy.stats import entropy


class AnomalyFeatureExtractor:
    def extract(self, df):
        rows = []
        for meter_id, group in df.groupby("meter_id"):
            signal = group["A+"].values
            signal = signal[~np.isnan(signal)]
            if len(signal) == 0:
                continue
            mean_load = np.mean(signal)
            std_load = np.std(signal)
            max_load = np.max(signal)
            peak_factor = (max_load /(mean_load + 1e-6))
            load_factor = (mean_load /(max_load + 1e-6))
            cv = (std_load /(mean_load + 1e-6))
            zero_ratio = np.mean(signal == 0)
            hist, _ = np.histogram(signal, bins=30)
            signal_entropy = entropy(hist + 1)
            row = {"meter_id": meter_id,
                "mean_load": mean_load,
                "std_load": std_load,
                "peak_factor": peak_factor,
                "load_factor": load_factor,
                "cv": cv,
                "zero_ratio": zero_ratio,
                "entropy": signal_entropy}
            rows.append(row)
        return pd.DataFrame(rows)