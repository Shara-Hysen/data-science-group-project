"""
src/prediction.py

Gemensamma funktioner för att göra förutsägelser med tränade ML-modeller.

Modulen ansvarar för att:
1. Beräkna sannolikheten för avbokning.
2. Omvandla sannolikheter till klassificeringar (0 eller 1).

Funktionerna kan användas för både Logistic Regression
och Random Forest.
"""


def predict_probabilities(model, X_val):
    """
    Beräknar sannolikheten för avbokning.

    Args:
        model: Den tränade ML-modellen.
        X_val: Features från valideringsdatan.

    Returns:
        Sannolikheten för avbokning (klass 1).
    """

    # Beräknar sannolikheten för avbokning för varje bokning.
    probabilities = model.predict_proba(X_val)[:, 1]

    # Returnerar sannolikheterna.
    return probabilities


def predict_with_threshold(probabilities, threshold=0.5):
    """
    Omvandlar sannolikheter till klassificeringar (0 eller 1).

    Args:
        probabilities: Modellens sannolikheter för klass 1.
        threshold: Gränsvärde för klassificering (standard 0.5).

    Returns:
        NumPy-array med förutsägelser (0 eller 1).
    """

    # Jämför varje sannolikhet med vår threshold.
    # True om sannolikheten >= threshold, annars False.
    # astype(int) omvandlar True till 1 och False till 0.
    predictions = (probabilities >= threshold).astype(int)

    return predictions