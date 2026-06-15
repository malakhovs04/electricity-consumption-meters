import matplotlib.pyplot as plt
import pandas as pd


class AnomalyAnalyzer:
    def __init__(self, output_dir):

        self.output_dir = output_dir

    def plot_anomaly_vs_normal(
        self,
        raw_df,
        meter_profiles_df,
        target_meter_id,
        consumer_class):

        target = raw_df[
            raw_df["meter_id"]
            == target_meter_id].copy()

        normal_ids = raw_df[
            raw_df["consumer_class"]
            == consumer_class
        ]["meter_id"].unique()

        normal_ids = [
            x for x in normal_ids
            if x != target_meter_id
        ]

        if len(normal_ids) == 0:
            return

        reference_id = normal_ids[0]

        reference = raw_df[
            raw_df["meter_id"]
            == reference_id
        ].copy()

        target["timestamp"] = pd.to_datetime(
            target["timestamp"]
        )

        reference["timestamp"] = pd.to_datetime(
            reference["timestamp"]
        )

        plt.figure(figsize=(15, 6))

        plt.plot(
            target["timestamp"],
            target["A+"],
            label=f"Anomaly {target_meter_id}",
            linewidth=2
        )

        plt.plot(
            reference["timestamp"],
            reference["A+"],
            alpha=0.7,
            label="Normal reference"
        )

        plt.legend()

        plt.title(
            f"Consumption comparison ({consumer_class})"
        )

        plt.tight_layout()

        plt.savefig(
            f"{self.output_dir}/{target_meter_id}.png",
            dpi=300
        )

        plt.close()