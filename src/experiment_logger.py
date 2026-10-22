"""
src/experiment_logger.py

Hjälpmodul för att bokföra modellkörningar och spara figurer.
Gör att alla i gruppen kan experimentera i sina notebooks och 
automatiskt samla resultatet i en gemensam CSV-fil.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt 
from src.config import get_logger


logger = get_logger(__name__)

DEFAULT_CSV_PATH = Path("outputs/experiment_results.csv")
DEFAULT_FIG_DIR = Path("outputs/figures")


def log_experiment(
    model_name: str,
    feature_set: str,
    metrics: dict,
    dataset: str = "val",
    notes: str = "",
    filepath: Path | str = DEFAULT_CSV_PATH
) -> pd.DataFrame:
    """
    Sparar en körning i en central CSV-fil (standard: outputs/experiment_results.csv)

    Args:
        model_name: Namn på modellen, t.ex. 'Random Forest', 'Logistic Regression'.
        feature_set: Beskrivning av features, t.ex. 'full', 'utan deposit_type'.
        metrics: Dict med utvärderingsmått (precision, recall, f1_score etc.).
        dataset: Vilket dataset som utvärderats ('val' eller 'test').
        notes: Valfria anteckningar (t.ex. hyperparametrar eller vem som körde).
        filepath: Sökväg till resultatfilen.

    Returns:
        pd.DataFrame: den sparade raden.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    pr_auc = metrics.get("pr_auc")
    # Bygg rad för CSV: .get() förhindrar KeyError om modellen saknar vissa mätvärden
    row = {
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "model": model_name,
        "feature_set": feature_set,
        "dataset": dataset,
        "precision_at_20": round(float(metrics.get("precision", 0.0)), 4),
        "recall_at_20": round(float(metrics.get("recall", 0.0)), 4),
        "f1_score": round(float(metrics.get("f1_score", 0.0)), 4),
        "pr_auc": round(float(pr_auc), 4) if pr_auc is not None else None,
        "antal_flaggade": metrics.get("antal_flaggade", None),
        "korrekt_flaggade": metrics.get("korrekt_flaggade", None),
        "notes": notes
    }

    df_row = pd.DataFrame([row])

    if path.exists():
        previous_results = pd.read_csv(path)
        results_to_save = pd.concat(
            [previous_results, df_row],
            ignore_index=True,
        )
    else:
        results_to_save = df_row

    results_to_save.to_csv(path, index=False)
    logger.info("Registrerade experiment i: %s", path)

    return df_row


def save_plot(
    filename: str,
    fig=None,
    output_dir: Path | str = DEFAULT_FIG_DIR,
    dpi: int = 300,
    close_fig: bool = False
) -> Path:
    """
    Sparar en graf till mappen för figurer (standard: outputs/figures/).

    Hanterar både Figure-objekt, Axes-objekt och den aktiva figuren (plt.gcf()).

    Args:
        filename: Filnamn, t.ex. 'rf_val_roc_curve.png'.
        fig: Matplotlib Figure- eller Axes-objekt. Om None används plt.gcf().
        output_dir: Mapp där bilderna sparas.
        dpi: Upplösning (standard 300 dpi för presentation och rapport).
        close_fig: Om figuren ska stängas efter sparande (standard False).

    Returns:
        Path: Sökvägen till den sparade filen.
    """
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if not filename.endswith(".png"):
        filename += ".png"

    filepath = target_dir / filename

    # Om ett Axes-objekt skickas in, hämta dess tillhörande figur
    if hasattr(fig, "figure"):
        target_fig = fig.figure
    elif fig is not None:
        target_fig = fig
    else:
        # Kontrollera att en aktiv figur finns innan vi sparar.
        if not plt.get_fignums():
            raise ValueError("Ingen aktiv figur att spara.")

        target_fig = plt.gcf()

    target_fig.savefig(filepath, bbox_inches="tight", dpi=dpi)
    logger.info("Figur sparad: %s", filepath)

    if close_fig:
        plt.close(target_fig)

    return filepath


def get_leaderboard(filepath: Path | str = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """
    Läser in och returnerar alla sparade experiment sorterade efter recall_at_20.

    Args:
        filepath: Sökväg till resultatfilen.

    Returns:
        pd.DataFrame: Sammanställd tabell över genomförda körningar.
    """
    path = Path(filepath)
    if not path.exists():
        logger.warning("Inga loggade experiment hittades på %s", path)
        return pd.DataFrame()

    df = pd.read_csv(path)
    return df.sort_values(by="recall_at_20", ascending=False).reset_index(drop=True)