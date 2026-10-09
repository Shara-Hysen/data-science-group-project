"""
src/data_split.py
Kronologisk uppdelning av städad data i träning, validering och test.

Prognosen görs vid bokningstillfället. Vid en prognostidpunkt (split_date) får
modellen bara lära sig av bokningar vars utfall redan var känt. Utfallet
(avbokning eller no-show) är känt senast vid ankomsten, så:
- Träning: bokningar med ankomst före split_date
- Utvärdering: bokningar gjorda från split_date till eval_end

Delmängder (städad data innehåller bokningar gjorda från 2015-07-01):
- train_val:  ankomst före 2016-07-01, för att träna modeller som jämförs på val
- val:        bokningar gjorda 2016-07-01 - 2016-09-30
- train_full: ankomst före 2016-10-01, för att träna om de valda modellerna inför test
- test:       bokningar gjorda 2016-10-01 - 2016-12-31

Bokningar gjorda från 2017 används inte, eftersom bokningar med lång ledtid
saknas i datan (den slutar med ankomster i augusti 2017)
"""

from pathlib import Path
import pandas as pd

from src.config import get_logger

logger = get_logger(__name__)

CLEANED_PATH = Path("data/processed/hotel_bookings_cleaned.csv")

VAL_SPLIT_DATE, VAL_END = "2016-07-01", "2016-10-01"
TEST_SPLIT_DATE, TEST_END = "2016-10-01", "2017-01-01"


def load_cleaned_data(path: Path = CLEANED_PATH) -> pd.DataFrame:
    """Läser in städad data med datumkolumnerna som datum."""
    logger.info("Läser in städad data från %s", path)
    return pd.read_csv(path, parse_dates=["arrival_date", "booking_date"])

def split_at(df: pd.DataFrame, split_date: str, eval_end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Delar upp data vid en prognostidpunkt."""
    split_date, eval_end = pd.Timestamp(split_date), pd.Timestamp(eval_end)

    train = df[df["arrival_date"] < split_date]
    evaluation = df[(df["booking_date"] >= split_date) & (df["booking_date"] < eval_end)]

    logger.info(
        "Split vid %s: %d träning (avbokade %.3f), %d utvärdering (avbokade %.3f)",
        split_date.date(), len(train), train["is_canceled"].mean(),
        len(evaluation), evaluation["is_canceled"].mean(),
    )
    return train, evaluation


def get_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Returnerar alla fyra delmängder."""
    train_val, val = split_at(df, VAL_SPLIT_DATE, VAL_END)
    train_full, test = split_at(df, TEST_SPLIT_DATE, TEST_END)
    return {"train_val": train_val, "val": val, "train_full": train_full, "test": test}


if __name__ == "__main__":
    df = load_cleaned_data()
    splits = get_splits(df)
    for name, data in splits.items():
        print(f"{name:11} {len(data):6} rader, bokade {data.booking_date.min().date()} "
              f"till {data.booking_date.max().date()}, avbokade {data.is_canceled.mean():.3f}")