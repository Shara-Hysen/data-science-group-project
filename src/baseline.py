
"""
src/baseline.py

Regelbaserad baseline för att identifiera bokningar med hög avbokningsrisk.

Baselinen använder lead_time som riskmått och flaggar de 20 % av
bokningarna som har längst tid mellan bokning och ankomst.

Ingen modell tränas och inga verkliga avbokningsutfall används
för att skapa förutsägelserna.
"""

import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score
from src.config import get_logger

logger = get_logger(__name__)


def predict_baseline(
    df: pd.DataFrame,
    top_fraction: float = 0.20
) -> pd.Series:
    """
    Flaggar bokningarna med längst lead_time.

    Args:
        df: DataFrame som innehåller kolumnen lead_time.
        top_fraction: Andel bokningar som ska flaggas (standard 0.20).

    Returns:
        pd.Series: Förutsägelser med 1 för flaggad bokning och 0 för övriga.
        Resultatet har samma index och radordning som indata.
    """
    if not 0 <= top_fraction <= 1:
        raise ValueError("top_fraction måste vara mellan 0 och 1.")

    if "lead_time" not in df.columns:
        raise ValueError("Kolumnen 'lead_time' saknas.")

    if df["lead_time"].isna().any():
        raise ValueError("Kolumnen 'lead_time' innehåller saknade värden.")

    # Beräkna hur många bokningar som ska flaggas.
    n_flagged = int(np.ceil(len(df) * top_fraction))

    # Skapa förutsägelser med 0 för samtliga bokningar.
    predictions = np.zeros(len(df), dtype=int)

    # Hitta radpositionerna för bokningarna med längst lead_time.
    top_positions = df["lead_time"].reset_index(drop=True).nlargest(n_flagged).index

    # Markera de utvalda bokningarna med 1.
    predictions[top_positions] = 1

    logger.info(
        "Baseline: %d av %d bokningar flaggades (%.1f %%).",
        n_flagged,
        len(df),
        100 * n_flagged / len(df) if len(df) else 0
    )

    return pd.Series(predictions, index=df.index, name="baseline_prediction")



def evaluate_baseline(df: pd.DataFrame) -> dict:
    """
    Utvärderar baselinen på hotellbokningar.

    Flaggar de 20 % med längst lead_time och jämför
    förutsägelserna med de verkliga avbokningarna.

    Args:
        df: DataFrame med lead_time och is_canceled.

    Returns:
        dict: Antal bokningar, antal flaggade,
        korrekt flaggade samt precision och recall.
    """
    # Verkliga avbokningar (facit).
    y_true = df["is_canceled"]

    # Baselinen flaggar de 20 % med längst lead_time.
    y_pred = predict_baseline(df)

    # Sammanställ resultaten.
    results = {
        "antal_bokningar": len(df),
        "antal_flaggade": int(y_pred.sum()),
        "korrekt_flaggade": int(((y_true == 1) & (y_pred == 1)).sum()),
        "precision_at_20": precision_score(y_true, y_pred, zero_division=0),
        "recall_at_20": recall_score(y_true, y_pred, zero_division=0),
    }

    logger.info(
        "Baseline utvärderad: Precision@20%% = %.2f, Recall@20%% = %.2f",
        results["precision_at_20"],
        results["recall_at_20"]
    )

    return results
