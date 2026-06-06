import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


class ClassificationAnalyzer:

    def plot_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(14, 12))

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(y_test.unique()))

        disp.plot(xticks_rotation=90, cmap="Blues")

        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("image/confusion_matrix.png", dpi=300)
        plt.show()