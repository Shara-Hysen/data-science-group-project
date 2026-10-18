import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
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

    return plt.gcf()



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

    return plt.gcf()



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

    return plt.gcf()



def plot_roc_curve(y_true, predictions):
    """
    Visualiserar ROC-kurvor för en eller flera modeller.

    Args:
        y_true: Facit (0 = inte avbokad, 1 = avbokad).
        predictions: Dictionary med modellnamn och
                     sannolikheter för klass 1.

    Returns:
        dict: ROC-AUC och Youdens threshold per modell.
    """

    results = {}

    plt.figure(figsize=(8, 6))

    # Går igenom varje modell och dess sannolikheter.
    for name, probabilities in predictions.items():

        # Beräknar ROC-kurvan.
        fpr, tpr, thresholds = roc_curve(y_true, probabilities)

        # Hittar threshold som maximerar Youdens J.
        best_idx = np.argmax(tpr - fpr)
        best_threshold = thresholds[best_idx]

        # Beräknar ROC-AUC.
        auc = roc_auc_score(y_true, probabilities)

        # Sparar resultatet för aktuell modell.
        results[name] = {
            "roc_auc": auc,
            "youden_threshold": best_threshold
        }

        # Ritar modellens ROC-kurva.
        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {auc:.3f})"
        )

        # Markerar modellens Youden-punkt.
        plt.scatter(
            fpr[best_idx],
            tpr[best_idx],
            label=f"{name} – Youdens J (threshold = {best_threshold:.2f})"
        )

    # Referenslinje för slumpmässig rangordning.
    plt.plot([0, 1], [0, 1], "--", label="Slumpnivå")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC-kurvor – Modelljämförelse")
    plt.legend()
    plt.grid(alpha=0.3)

    return results


def plot_pr_curve(y_true, predictions):
    """
    Visualiserar Precision-Recall-kurvor för en eller flera modeller.

    Args:
        y_true: Facit (0 = inte avbokad, 1 = avbokad).
        predictions: Dictionary med modellnamn och sannolikheter för klass 1.

    Returns:
        dict: PR-AUC (Average Precision) per modell.
    """
    results = {}
    fig = plt.figure(figsize=(8, 6))

    # Slumpnivå: andelen faktiska avbokningar i datan
    baseline = np.mean(y_true)
    plt.plot([0, 1], [baseline, baseline], "--", label=f"Slumpnivå ({baseline:.2f})")

    # Plotta PR-kurvan för varje modell
    for name, probabilities in predictions.items():
        precision, recall, _ = precision_recall_curve(y_true, probabilities)
        ap = average_precision_score(y_true, probabilities)

        results[name] = {"pr_auc": ap}
        plt.plot(recall, precision, label=f"{name} (PR-AUC = {ap:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall-kurvor – Modelljämförelse")
    plt.legend()
    plt.grid(alpha=0.3)

    return results


def flag_top_share(scores: pd.Series | np.ndarray, share: float = 0.20) -> np.ndarray:
    """
    Flaggar de bokningar som har högst riskpoäng.

    Funktionen kan användas för både baseline och ML-modeller.

    Args:
        scores: Riskpoäng för varje bokning, t.ex. lead_time
                från baseline eller förutsagda sannolikheter
                från en ML-modell.
        share: Andel bokningar som ska flaggas (standard 20 %).

    Returns:
        np.ndarray: En array med 1 för flaggade bokningar
                    och 0 för övriga, i samma radordning som indata.
    """
    if not 0 <= share <= 1:
        raise ValueError("share måste vara mellan 0 och 1.")

    # Omvandla riskpoängen till en NumPy-array.
    scores = np.asarray(scores)

    # Beräkna hur många bokningar som ska flaggas.
    n_flagged = int(np.ceil(len(scores) * share))

    # Skapa en array med 0 för alla bokningar.
    flags = np.zeros(len(scores), dtype=int)

    # Hitta positionerna för bokningarna med högst riskpoäng.
    top_positions = np.argsort(-scores, kind="stable")[:n_flagged]

    # Markera de utvalda bokningarna med 1.
    flags[top_positions] = 1

    return flags


def evaluate_top_share(
    y_true: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
    share: float = 0.20
) -> dict:
    """
    Utvärderar de bokningar som flaggas med högst riskpoäng.

    Args:
        y_true: Verkliga avbokningar (0 eller 1).
        scores: Riskpoäng från baseline eller en ML-modell.
        share: Andel bokningar som ska flaggas (standard 20 %).

    Returns:
        dict: Antal bokningar, antal flaggade, korrekt flaggade
              samt precision, recall och F1-score.
    """

    # Flagga de bokningar som har högst riskpoäng.
    y_pred = flag_top_share(scores, share)

    # Jämför y_pred med y_true och beräkna precision, recall och F1-score.
    metrics = evaluate_model(y_true, y_pred)

    # Sammanställ antalet bokningar och utvärderingsmåtten.
    results = {
        "share": share,
        "antal_bokningar": len(y_true),
        "antal_flaggade": int(y_pred.sum()),
        "korrekt_flaggade": int(np.sum((np.asarray(y_true) == 1) & (y_pred == 1))),
        "precision": metrics["Precision"],
        "recall": metrics["Recall"],
        "f1_score": metrics["F1-score"],
    }

    return results

