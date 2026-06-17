import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve


class TransitionRiskVisualizer:
    """
    Визуализация результатов TransitionRiskModel:
      - ROC-кривая на тестовых счетчиках
      - важность признаков
      - распределение текущей вероятности перехода в аномалию
    """

    def __init__(self, output_dir: str = "image"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_roc_curve(self, y_test, y_proba):
        if len(set(y_test)) < 2:
            print("Недостаточно классов в тестовой выборке для ROC-кривой.")
            return

        fpr, tpr, _ = roc_curve(y_test, y_proba)

        plt.figure(figsize=(7, 7))
        plt.plot(fpr, tpr, linewidth=2, label="ROC")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Случайный классификатор")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC: прогноз перехода 'норма -> аномалия'")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "transition_risk_roc.png"),
            dpi=300, bbox_inches="tight"
        )
        plt.show()

    def plot_feature_importance(self, model, feature_names, top_n=10):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(10, 6))
        plt.title("Важность признаков: прогноз перехода в аномалию")
        plt.bar(range(len(indices)), importances[indices], align="center")
        plt.xticks(
            range(len(indices)),
            [feature_names[i] for i in indices],
            rotation=45, ha="right"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "transition_risk_feature_importance.png"),
            dpi=300
        )
        plt.show()

    def plot_risk_distribution(self, current_risk_df):
        if current_risk_df.empty:
            print("Нет данных для распределения риска переходов.")
            return

        plt.figure(figsize=(10, 6))
        sns.histplot(current_risk_df["transition_risk_proba"], bins=30, kde=True)
        plt.title("Текущая вероятность перехода в аномалию (следующее окно)")
        plt.xlabel("Вероятность")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "transition_risk_distribution.png"),
            dpi=300
        )
        plt.show()