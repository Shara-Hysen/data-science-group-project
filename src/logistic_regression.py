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