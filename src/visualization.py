import os
import matplotlib.pyplot as plt


class Visualizer:

    def __init__(self, output_dir: str = "image"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_umap(self, embedding_df):
        plt.figure(figsize=(16, 10))
        for cls in embedding_df["consumer_class"].unique():
            subset = embedding_df[
                embedding_df["consumer_class"] == cls]

            plt.scatter(subset["x"], subset["y"], s=8, alpha=0.7, label=cls)

        plt.title("UMAP Projection")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

        plt.subplots_adjust(right=0.75)
        plt.savefig(os.path.join(self.output_dir, "umap_projection.png"), dpi=300, bbox_inches="tight")
        plt.show()