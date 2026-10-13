
"""
src/preprocessing.py

Modul för preprocessing av hotellbokningsdata inför maskininlärning.

Modulen ansvarar för att:
1. Välja features som är tillgängliga vid bokningstillfället.
2. Göra om agent till kategori och company till 0/1.
3. Skala numeriska variabler.
4. Koda kategoriska variabler med OneHotEncoder. För agent slås
   ovanliga kategorier ihop till en gemensam kategori.
5. Separera features (X) och målvariabel (y).

Saknade värden hanteras i städningen (src/data_cleaning.py). Imputerarna
finns kvar som skydd om nya saknade värden skulle uppstå.

Preprocessorn anpassas endast på träningsdata för att undvika dataläckage.
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
    "company",
    "reserved_room_type",
]

AGENT_FEATURE = ["agent"]

# Agenter med färre bokningar än detta i träningsdatan slås ihop.
# Gränsen 50 minskar antalet agentkategorier från 210 till 72,
# medan cirka 5 % av bokningarna hamnar i den gemensamma kategorin.
MIN_AGENT_FREQUENCY = 50

def get_categorical_features(include_deposit_type: bool = True) -> list[str]:
    """
    Returnerar de kategoriska variablerna, utom agent.

    deposit_type ingår i huvudmodellen. Med include_deposit_type=False
    kan modellen köras utan variabeln som jämförelse.
    """
    if include_deposit_type:
        return CATEGORICAL_FEATURES + ["deposit_type"]
    return CATEGORICAL_FEATURES


# --------------------------------------------------
# Separera features och målvariabel
# --------------------------------------------------

def split_features_target(
    df: pd.DataFrame,
    include_deposit_type: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Delar upp datasetet i features (X) och målvariabel (y).

    Endast de features som gruppen har valt inkluderas.
    Övriga kolumner exkluderas för att minska risken för dataläckage.
    """
    features = (
        NUMERICAL_FEATURES
        + AGENT_FEATURE
        + get_categorical_features(include_deposit_type)
    )


    X = df[features].copy()
    y = df[TARGET].copy()

    # Agent och company är ID-nummer och behandlas
    # därför som kategoriska variabler.
    # Saknade ID-värden representeras av 0.

    # Agent är ett ID och behandlas som kategori.
    # 0 betyder ingen resebyrå (satt i städningen).
    X["agent"] = X["agent"].astype(int).astype(str)

    # Company görs om till 0/1: bokad via företag eller inte.
    # 0 betyder inget företag (satt i städningen).
    X["company"] = (X["company"] != 0).astype(int)

    return X, y


# --------------------------------------------------
# Skapa preprocessing-pipeline
# --------------------------------------------------

def build_preprocessor(include_deposit_type: bool = True) -> ColumnTransformer:
    """
    Skapar en preprocessing-pipeline.

    Numeriska features:
    - Ersätter saknade värden med medianen.
    - Standardiserar värden med StandardScaler.

    Agent:
    - Omvandlas med OneHotEncoder. Agenter med färre än MIN_AGENT_FREQUENCY
      bokningar i träningsdatan, och agenter som inte finns i träningsdatan
      alls, slås ihop till en gemensam kategori.

    Övriga kategoriska features:
    - Ersätter eventuella saknade värden med vanligaste värdet.
    - Omvandlas med OneHotEncoder. Kategorier som inte finns i
      träningsdatan får nollor i alla kolumner..

    Samma värde på include_deposit_type måste användas som i split_features_target
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

    # Agenter med färre än 50 bokningar i träningsdatan, och agenter som inte
    # finns i träningsdatan alls, slås ihop till en gemensam kategori.
    agent_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=MIN_AGENT_FREQUENCY,
                ),
            ),
        ]
    )

    # Pipeline för övriga kategoriska variabler
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
    return ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                NUMERICAL_FEATURES
            ),
            (
                "agent",
                agent_pipeline,
                AGENT_FEATURE
            ),
            (
                "cat",
                categorical_pipeline,
                get_categorical_features(include_deposit_type)
            ),

        ],

        # Exkluderar kolumner som inte ingår i feature-urvalet
        remainder="drop"
    )


# --------------------------------------------------
# Testa preprocessing med gruppens datasplit
# --------------------------------------------------

if __name__ == "__main__":

    from src.data_split import load_cleaned_data, get_splits

    # Läs in städad data och hämta kronologiska uppdelningar
    splits = get_splits(load_cleaned_data())

    train = splits["train_val"]
    val = splits["val"]

    for include_deposit_type in [True, False]:
    # Separera features och målvariabel
        X_train, y_train = split_features_target(train, include_deposit_type)
        X_val, y_val = split_features_target(val, include_deposit_type)

        # Skapa preprocessorn
        preprocessor = build_preprocessor(include_deposit_type)

        # Anpassa preprocessorn endast på träningsdata
        X_train_processed = preprocessor.fit_transform(X_train)

        # Omvandla valideringsdata med samma preprocessor
        X_val_processed = preprocessor.transform(X_val)

        # Kontrollera dimensionerna före och efter preprocessing
        print(f"\nMed deposit_type: {include_deposit_type}")
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
