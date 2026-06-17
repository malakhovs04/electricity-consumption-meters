import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class AnomalyTypeVisualizer:
    """
    Визуализация результатов AnomalyTypeClassifier:
      - PCA-проекция аномалий с раскраской по типу
      - boxplot ключевых признаков по типам аномалий
    """

    def __init__(self, output_dir: str = "image"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_pca_scatter(self, clustered_anomalies: pd.DataFrame):
        if clustered_anomalies.empty or "pca_x" not in clustered_anomalies.columns:
            print("Нет данных для PCA-визуализации типов аномалий.")
            return

        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=clustered_anomalies,
            x="pca_x",
            y="pca_y",
            hue="anomaly_type",
            s=60,
            alpha=0.8,
        )
        plt.title("Типы аномалий (PCA-проекция признаков)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "anomaly_types_pca.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    def plot_feature_distributions(self, clustered_anomalies: pd.DataFrame, features=None):
        if clustered_anomalies.empty:
            print("Нет данных для построения распределений по типам аномалий.")
            return

        if features is None:
            features = [
                "cv", "zero_ratio", "entropy",
                "night_day_ratio", "large_gap_ratio", "max_gap_hours",
            ]
        features = [f for f in features if f in clustered_anomalies.columns]

        n = len(features)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 6), sharex=False)
        if n == 1:
            axes = [axes]

        for ax, feature in zip(axes, features):
            sns.boxplot(
                data=clustered_anomalies,
                x="anomaly_type",
                y=feature,
                ax=ax,
            )
            ax.set_title(feature)
            ax.tick_params(axis="x", rotation=90, labelsize=7)
            ax.set_xlabel("")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "anomaly_types_features.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()