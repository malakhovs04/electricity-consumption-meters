from src.dataset_loader import DatasetLoader
from src.features import FeatureExtractor
from src.eda import EDAAnalyzer
from src.embedding import UMAPEmbedding
from src.visualization import Visualizer
from src.classifier import DataSplitter
from src.classifier import ConsumerClassifier
from src.classification_analysis import ClassificationAnalyzer
from src.consumption_profiles import ConsumptionProfileBuilder
from src.anomaly_features import AnomalyFeatureExtractor
from src.isolation_forest_detector import IsolationForestDetector
from src.visualizer import AnomalyVisualizer
from src.anomaly_embedding import AnomalyUMAP
from src.risk_scorer import RiskScorer
from src.anomaly_analizer import AnomalyAnalyzer


def main():

    print("\n========== STEP 1: LOAD DATA ==========")

    loader = DatasetLoader("data/raw")
    df = loader.load_all()
    print("Loaded shape:", df.shape)

    # =====================================================
    # FEATURE ENGINEERING FOR CLASSIFICATION
    # =====================================================

    print("\n========== STEP 2: FEATURE EXTRACTION ==========")

    extractor = FeatureExtractor()
    features_df = extractor.extract(df)
    features_df.to_csv("data/processed/features.csv",index=False)
    print("Features saved")

    # =====================================================
    # EDA
    # =====================================================

    print("\n========== STEP 3: EDA ==========")

    eda = EDAAnalyzer()
    eda.basic_info(features_df)
    eda.class_distribution(features_df)
    summary = eda.numerical_summary(features_df)
    missing = eda.missing_report(features_df)
    constants = eda.constant_features(features_df)
    print("\nConstant features:")
    print(constants)

    # =====================================================
    # UMAP FOR CONSUMER CLASSES
    # =====================================================

    print("\n========== STEP 4: UMAP ==========")

    embedder = UMAPEmbedding()
    embedding_df = embedder.fit_transform(features_df)
    visualizer = Visualizer()
    visualizer.plot_umap(embedding_df)

    # =====================================================
    # RANDOM FOREST CLASSIFICATION
    # =====================================================

    print("\n========== STEP 5: CLASSIFICATION ==========")

    splitter = DataSplitter()
    X_train, X_test, y_train, y_test = (splitter.split(features_df))
    classifier = ConsumerClassifier()
    model, y_pred = classifier.train(X_train, X_test, y_train, y_test)

    # =====================================================
    # CLASSIFICATION ANALYSIS
    # =====================================================

    print("\n========== STEP 6: MODEL ANALYSIS ==========")

    analyzer = ClassificationAnalyzer()
    analyzer.plot_confusion_matrix(y_test,y_pred)

    analyzer.feature_importance(model,features_df.drop(columns=["meter_id","consumer_class"]).columns)
    top_features = ["A+_fft_max",
        "A+_fft_energy",
        "A+_mean",
        "apparent_power_mean",
        "A+_p95"]
    for feature in top_features:
        analyzer.boxplot_features( features_df, feature, "consumer_class")

    # =====================================================
    # CONSUMPTION PROFILES
    # =====================================================

    print("\n========== STEP 7: CONSUMPTION PROFILES ==========")

    profile_builder = ConsumptionProfileBuilder()
    profiles = profile_builder.build(df)
    profiles.to_csv("data/processed/meter_profiles.csv",index=False)
    print("Meter profiles saved")
    print("Profiles shape:", profiles.shape)

    # =====================================================
    # ANOMALY FEATURES
    # =====================================================

    print("\n========== STEP 8: ANOMALY FEATURES ==========")

    anomaly_feature_extractor = (AnomalyFeatureExtractor())
    anomaly_features = (anomaly_feature_extractor.extract(df))
    anomaly_features.to_csv( "data/processed/anomaly_features.csv",index=False)
    print("Anomaly features saved")

    # =====================================================
    # ISOLATION FOREST
    # =====================================================

    print("\n========== STEP 9: ISOLATION FOREST ==========")

    detector = IsolationForestDetector()
    anomaly_scores = detector.detect(anomaly_features)
    anomaly_scores.to_csv("data/processed/anomaly_scores.csv",index=False)
    top_anomalies = (anomaly_scores.sort_values("anomaly_score").head(50))
    print("Top anomalies:")
    print(top_anomalies[["meter_id", "anomaly_score"]])
    top_anomalies.to_csv("data/processed/top_anomalies.csv", index=False)
    print("Top anomalies saved")

    # =====================================================
    # UMAP FOR ANOMALIES
    # =====================================================

    print("\n========== STEP 10: ANOMALY UMAP ==========")

    anomaly_embedder = (AnomalyUMAP())
    anomaly_embedding = (anomaly_embedder.fit_transform(anomaly_scores))
    anomaly_embedding["anomaly"] = (anomaly_scores["anomaly"].values)
    anomaly_visualizer = (AnomalyVisualizer())
    anomaly_visualizer.plot( anomaly_embedding)

    # =====================================================
    # RISK SCORING
    # =====================================================

    print("\n========== STEP 11: RISK SCORING ==========")

    scorer = RiskScorer()
    risk_report = scorer.calculate_risk(anomaly_scores_df=anomaly_scores,
        meter_profiles_df=profiles,
        anomaly_features_df=anomaly_features)
    
    if "consumer_class" not in risk_report.columns:
        meter_classes = (df[["meter_id", "consumer_class"]].drop_duplicates())
        risk_report["meter_id"] = risk_report["meter_id"].astype(str).str.strip()
        meter_classes["meter_id"] = meter_classes["meter_id"].astype(str).str.strip()
        risk_report = risk_report.merge(meter_classes, on="meter_id", how="left")
    risk_report.to_csv("data/processed/final_risk_report.csv", index=False)

    print(risk_report[["meter_id", "risk_score", "consumer_class"]].head(10))
    # =====================================================
    # REPORTS FOR TOP ANOMALIES
    # =====================================================

    print("\n========== STEP 12: GENERATE REPORTS ==========")

    anomaly_analyzer = AnomalyAnalyzer(output_dir="image/reports")
    top_risk_objects = (risk_report.head(5))
    for _, row in top_risk_objects.iterrows():
        anomaly_analyzer.plot_anomaly_vs_normal(raw_df=df,
            meter_profiles_df=profiles,
            target_meter_id=row["meter_id"],
            consumer_class=row["consumer_class"])

    print("\nPipeline finished successfully!")


if __name__ == "__main__":
    main()