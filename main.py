from src.dataset_loader import DatasetLoader
from src.features import FeatureExtractor
from src.eda import EDAAnalyzer
from src.embedding import UMAPEmbedding
from src.visualization import Visualizer
from src.classifier import DataSplitter
from src.classifier import ConsumerClassifier
from src.classification_analysis import ClassificationAnalyzer

def main():
    loader = DatasetLoader("data/raw")
    df = loader.load_all()
    print("loaded shape:", df.shape)

    extractor = FeatureExtractor()
    features_df = extractor.extract(df)
    print(features_df.head())

    features_df.to_csv("data/processed/features.csv", index=False)
    print("Features saved")

    eda = EDAAnalyzer()
    eda.basic_info(features_df)
    eda.class_distribution(features_df)
    summary = eda.numerical_summary(features_df)
    missing = eda.missing_report(features_df)
    constants = eda.constant_features(features_df)
    print("\nConstant features:")
    print(constants)

    embedder = UMAPEmbedding()
    embedding_df = embedder.fit_transform(features_df)

    visualizer = Visualizer()
    visualizer.plot_umap(embedding_df)

    splitter = DataSplitter()
    X_train, X_test, y_train, y_test = splitter.split(features_df)

    classifier = ConsumerClassifier()
    model, y_pred = classifier.train(X_train, X_test, y_train, y_test)
    analyzer = ClassificationAnalyzer()
    analyzer.plot_confusion_matrix(y_test, y_pred)

if __name__ == "__main__":
    main()