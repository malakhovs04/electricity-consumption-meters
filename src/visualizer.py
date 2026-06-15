import os
import matplotlib.pyplot as plt


class AnomalyVisualizer:
    def __init__(self, output_dir: str = "image"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def plot(self, embedding_df):
        plt.figure(figsize=(14, 10))
        normal = embedding_df[embedding_df["anomaly"] == 1]
        anomaly = embedding_df[embedding_df["anomaly"] == -1]
        plt.scatter(normal["x"],
            normal["y"],
            s=8,
            alpha=0.4,
            label="Normal")
        plt.scatter(anomaly["x"],
            anomaly["y"],
            s=20,
            label="Anomaly")
        plt.legend()
        plt.title("UMAP Anomaly Detection")
        plt.savefig(os.path.join(self.output_dir, "anomaly_umap.png"), dpi=300, bbox_inches="tight")
        plt.show()