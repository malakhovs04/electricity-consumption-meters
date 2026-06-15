import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

class ClassificationAnalyzer:

    def __init__(self, output_dir: str = "image"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred, normalize="true")
        fig, ax = plt.subplots(figsize=(22, 18))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(y_test.unique()))
        disp.plot(ax=ax, cmap="Blues", xticks_rotation=90, values_format=".2f")
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        for text in ax.texts:
            text.set_fontsize(8)
        plt.title("Normalized Confusion Matrix", fontsize=18, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "confusion_matrix.png"), dpi=400, bbox_inches="tight")
        plt.show()

    def feature_importance(self, model, feature_names, top_n=20):
        importances = model.feature_importances_
        indices = importances.argsort()[::-1][:top_n]
        print("\nTop Feature Importances:")
        for i in indices:
            print(f"{feature_names[i]}: {importances[i]:.4f}")

        plt.figure(figsize=(10, 6))
        plt.title("Top Feature Importances")
        plt.bar(range(top_n), importances[indices], align="center")
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "feature_importance.png"), dpi=300)
        plt.show()

    def boxplot_features(self, df, feature, target):
        plt.figure(figsize=(16, 6))
        sns.boxplot(data=df, x=target, y=feature)
        plt.yscale("log")
        plt.xticks(rotation=90)
        plt.title(f"{feature} by {target}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"boxplot_{feature}.png"), dpi=300)
        plt.show()