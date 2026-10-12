import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)


def evaluate_model(y_true, y_pred):
    """
    Beräknar modellens utvärderingsmått.

    Args:
        y_true: Facit (verkliga 0 och 1).
        y_pred: Modellens förutsägelser (0 och 1).

    Returns:
        dict: Precision, Recall och F1-score.
    """

    metrics = {
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred)
    }

    return metrics



# Plottar enbart en CM
def plot_confusion_matrix(y_true, y_pred):
    """
    Visualiserar Confusion Matrix för en modell.

    Args:
        y_true: Facit (verkliga 0 och 1).
        y_pred: Modellens förutsägelser (0 och 1).
    """

    # Beräknar Confusion Matrix.
    # labels=[0, 1] säkerställer samma klassordning.
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Skapar visualiseringen.
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Inte avbokad", "Avbokad"]
    )

    # Ritar matrisen.
    disp.plot(values_format="d")

    plt.title("Confusion Matrix")
    plt.show()



# Plottar flera CM för enklare jämförelse av flera modeller
def plot_multiple_confusion_matrices(y_true, predictions):
    """
    Visualiserar Confusion Matrix för flera modeller.

    Args:
        y_true: Facit (verkliga 0 och 1).
        predictions: Dictionary med modellnamn och förutsägelser.
    """

    # Räknar antalet modeller som ska jämföras.
    n_models = len(predictions)

    # Skapar en figur med en Confusion Matrix per modell.
    fig, axes = plt.subplots(
        1, n_models,
        figsize=(6 * n_models, 5),
        squeeze=False
    )

    # Går igenom varje modell och dess förutsägelser.
    for i, (name, y_pred) in enumerate(predictions.items()):

        # Beräknar Confusion Matrix.
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        # Skapar den visuella matrisen.
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Inte avbokad", "Avbokad"]
        )

        # Ritar matrisen på modellens plats i figuren.
        disp.plot(
            ax=axes[0, i],
            values_format="d"
        )

        axes[0, i].set_title(name)

    plt.tight_layout()
    plt.show()



def plot_threshold_metrics(y_true, probabilities):
    """
    Visualiserar Precision, Recall och F1 vid olika thresholds.

    Args:
        y_true: Facit (0 eller 1).
        probabilities: Modellens sannolikheter för klass 1.
    """

    # Skapar 99 thresholds från 0.01 till 0.99.
    thresholds = np.linspace(0.01, 0.99, 99)

    precision_values = []
    recall_values = []
    f1_values = []

    # Testar varje threshold på samma valideringsdata.
    for threshold in thresholds:

        # Omvandlar sannolikheterna till 0 eller 1.
        y_pred = (probabilities >= threshold).astype(int)

        # Sparar utvärderingsmåtten för denna threshold.
        precision_values.append(precision_score(y_true, y_pred, zero_division=0))
        recall_values.append(recall_score(y_true, y_pred, zero_division=0))
        f1_values.append(f1_score(y_true, y_pred, zero_division=0))

    # Ritar de tre måtten mot threshold.
    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, precision_values, label="Precision")
    plt.plot(thresholds, recall_values, label="Recall")
    plt.plot(thresholds, f1_values, label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision, Recall och F1 vid olika thresholds")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()



def plot_roc_curve(y_true, probabilities):
    """
    Visualiserar ROC-kurvan och beräknar ROC-AUC.

    Args:
        y_true: Facit (0 = inte avbokad, 1 = avbokad).
        probabilities: Modellens sannolikheter för avbokning.

    Returns:
        dict: ROC-AUC och threshold enligt Youdens J.
    """

    # Beräknar False Positive Rate, True Positive Rate
    # och motsvarande thresholds.
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)

    # Hittar den punkt som maximerar skillnaden TPR - FPR.
    # Detta kallas Youdens J-statistik.
    best_idx = np.argmax(tpr - fpr)

    best_threshold = thresholds[best_idx]
    best_fpr = fpr[best_idx]
    best_tpr = tpr[best_idx]

    # Beräknar arean under ROC-kurvan.
    auc = roc_auc_score(y_true, probabilities)

    # Skriver ut resultaten.
    print(f"ROC-AUC: {auc:.3f}")
    print(f"Threshold (Youdens J): {best_threshold:.4f}")

    # Skapar ROC-diagrammet.
    plt.figure(figsize=(7, 5))

    plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.3f})")

    # Diagonal linje motsvarar slumpmässig rangordning.
    plt.plot([0, 1], [0, 1], "--", label="Slumpnivå")

    # Markerar threshold som maximerar Youdens J.
    plt.scatter(
        best_fpr,
        best_tpr,
        color="black",
        label=f"Youden threshold = {best_threshold:.2f}"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC-kurva – Logistic Regression")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    return {
        "roc_auc": auc,
        "youden_threshold": best_threshold
    }