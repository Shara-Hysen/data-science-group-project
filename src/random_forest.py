"""
src/random_forest.py

Funktioner för att träna och prediktera med Random Forest.
"""

from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def train_random_forest(
    X_train,
    y_train,
    n_estimators: int = 100,
    max_depth: int | None = 15,
    min_samples_split: int = 2,
    min_samples_leaf: int = 2,
    random_state: int = 42,
    n_jobs: int = -1,
    **kwargs
) -> RandomForestClassifier:
    """
    Tränar en Random Forest-modell.

    Args:
        X_train: Förbehandlade features (matris).
        y_train: Facit (0 eller 1).
        n_estimators: Antal träd i skogen.
        max_depth: Maxdjup per träd.
        min_samples_split: Minsta antal samples för att dela en nod.
        min_samples_leaf: Minsta antal samples i ett löv.
        random_state: Slumptalsfrö.
        n_jobs: Antal CPU-kärnor (-1 för alla tillgängliga).

    Returns:
        RandomForestClassifier: Den tränade modellen.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs
    )
    model.fit(X_train, y_train)
    return model


def predict_probabilities(model: RandomForestClassifier, X_val):
    """
    Beräknar sannolikheten för avbokning (klass 1).

    Args:
        model: Den tränade Random Forest-modellen.
        X_val: Förbehandlade features från valideringsdatan.

    Returns:
        NumPy-array med sannolikheter för klass 1.
    """
    return model.predict_proba(X_val)[:, 1]


def predict_with_threshold(probabilities, threshold: float = 0.5):
    """
    Omvandlar sannolikheter till klassificeringar (0 eller 1) utifrån threshold.
    """
    return (probabilities >= threshold).astype(int)


def get_feature_importances(model: RandomForestClassifier, feature_names: list) -> pd.DataFrame:
    """
    Hämtar variablernas betydelse (feature importance) i fallande ordning.

    Args:
        model: Tränad RandomForestClassifier.
        feature_names: Kolumnnamn från preprocessorn (t.ex. preprocessor.get_feature_names_out()).

    Returns:
        pd.DataFrame med featurenamn och deras importances.
    """
    df_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    })
    return df_importance.sort_values(by="importance", ascending=False).reset_index(drop=True)