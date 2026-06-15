import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap


class AnomalyUMAP:
    def fit_transform(self, df):

        X = df.drop(columns=["meter_id", "anomaly","anomaly_score"])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42)
        embedding = reducer.fit_transform(X_scaled)
        result = pd.DataFrame({"x": embedding[:, 0],
            "y": embedding[:, 1],
            "anomaly": df["anomaly"]})

        return result