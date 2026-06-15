from sklearn.ensemble import IsolationForest

class IsolationForestDetector:
    def detect(self, features_df):
        X = features_df.drop(columns=["meter_id"], errors="ignore")
        model = IsolationForest(n_estimators=500, contamination=0.03, random_state=42)
        labels = model.fit_predict(X)
        scores = model.score_samples(X)
        result = features_df.copy()
        result["anomaly"] = labels
        result["anomaly_score"] = scores

        return result