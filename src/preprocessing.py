
"""
src/preprocessing.py

Modul för preprocessing av hotellbokningsdata inför maskininlärning.

Modulen ansvarar för att:
1. Välja features som är tillgängliga vid bokningstillfället.
2. Hantera saknade värden.
3. Skala numeriska variabler.
4. Koda kategoriska variabler med OneHotEncoder.
5. Separera features (X) och målvariabel (y).

Preprocessorn anpassas endast på träningsdata
för att undvika dataläckage.
"""

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# Målvariabel
# --------------------------------------------------

TARGET = "is_canceled"


# --------------------------------------------------
# Feature-urval enligt gruppens gemensamma beslut
# --------------------------------------------------

NUMERICAL_FEATURES = [
    "lead_time",
    "previous_cancellations",
    "previous_bookings_not_canceled",
]

CATEGORICAL_FEATURES = [
    "hotel",
    "arrival_date_month",
    "market_segment",
    "distribution_channel",
    "customer_type",
    "is_repeated_guest",
    "agent",
    "company",
    "reserved_room_type",
]

FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# --------------------------------------------------
# Separera features och målvariabel
# --------------------------------------------------

def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Delar upp datasetet i features (X) och målvariabel (y).

    Endast de features som gruppen har valt inkluderas.
    Övriga kolumner exkluderas för att minska risken
    för dataläckage.
    """

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # Agent och company är ID-nummer och behandlas
    # därför som kategoriska variabler.
    # Saknade ID-värden representeras av 0.

    X["agent"] = X["agent"].fillna(0).astype(str)
    X["company"] = X["company"].fillna(0).astype(str)

    return X, y


# --------------------------------------------------
# Skapa preprocessing-pipeline
# --------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """
    Skapar en preprocessing-pipeline.

    Numeriska features:
    - Ersätter saknade värden med medianen.
    - Standardiserar värden med StandardScaler.

    Kategoriska features:
    - Ersätter saknade värden med vanligaste värdet.
    - Omvandlar kategorier med OneHotEncoder.

    Preprocessorn ska endast anpassas på träningsdata.
    """

    # Pipeline för numeriska variabler
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    # Pipeline för kategoriska variabler
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            ),
        ]
    )

    # Kombinerar numerisk och kategorisk preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                NUMERICAL_FEATURES
            ),
            (
                "cat",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            ),
        ],

        # Exkluderar kolumner som inte ingår i feature-urvalet
        remainder="drop"
    )

    return preprocessor


# --------------------------------------------------
# Testa preprocessing med gruppens datasplit
# --------------------------------------------------

if __name__ == "__main__":

    from src.data_split import load_cleaned_data, get_splits

    # Läs in städad data och hämta kronologiska uppdelningar
    splits = get_splits(load_cleaned_data())

    train = splits["train_val"]
    val = splits["val"]

    # Separera features och målvariabel
    X_train, y_train = split_features_target(train)
    X_val, y_val = split_features_target(val)

    # Skapa preprocessorn
    preprocessor = build_preprocessor()

    # Anpassa preprocessorn endast på träningsdata
    X_train_processed = preprocessor.fit_transform(X_train)

    # Omvandla valideringsdata med samma preprocessor
    X_val_processed = preprocessor.transform(X_val)

    # Kontrollera dimensionerna före och efter preprocessing
    print("Träningsdata före:", X_train.shape)
    print("Träningsdata efter:", X_train_processed.shape)

    print("Valideringsdata före:", X_val.shape)
    print("Valideringsdata efter:", X_val_processed.shape)

    # Kontrollera att antalet rader inte har förändrats
    assert X_train_processed.shape[0] == len(X_train)
    assert X_val_processed.shape[0] == len(X_val)

    # Kontrollera att båda dataseten har samma antal kolumner
    assert X_train_processed.shape[1] == X_val_processed.shape[1]

    print("Preprocessing genomförd utan fel!")
