from sklearn.linear_model import LogisticRegression


def train_logistic_regression(X_train, y_train):
    """
    Tränar en logistisk regressionsmodell.

    Args:
        X_train: Features (oberoende variabler).
        y_train: Facit (målvariabeln, 0 eller 1).

    Returns:
        LogisticRegression: Den tränade modellen.
    """

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model



def predict_probabilities(model, X_val):
    """
    Beräknar sannolikheten för avbokning.

    Args:
        model: Den tränade Logistic Regression-modellen.
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

    # Jämför varje sannolikhet med vår threshold. True om sannolikheten >= threshold, annars False.
    # astype(int) omvandlar True till 1 och False till 0.
    predictions = (probabilities >= threshold).astype(int)

    return predictions