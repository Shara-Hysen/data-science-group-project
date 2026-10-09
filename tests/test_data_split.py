"""
tests/test_data_split.py
Tester för den kronologiska uppdelningen i src/data_split.py.

Testerna använder ett litet påhittat dataset där varje bokning är vald för att
pröva en viss gräns i uppdelningen.
"""

import pandas as pd
import pytest

from src.data_split import get_splits, load_cleaned_data


@pytest.fixture
def bookings() -> pd.DataFrame:
    data = [
        # id, booking_date, arrival_date, is_canceled
        ("A", "2016-01-10", "2016-03-01", 0),  # ankomst långt före första split
        ("B", "2016-03-01", "2016-06-30", 1),  # ankomst dagen före första split
        ("C", "2016-05-01", "2016-07-01", 0),  # ankomst exakt på första split, ej i train_val
        ("D", "2016-07-01", "2016-08-15", 1),  # bokad första dagen i val
        ("E", "2016-09-30", "2017-01-10", 0),  # bokad sista dagen i val, ej i train_full
        ("F", "2016-10-01", "2016-12-01", 1),  # bokad första dagen i test
        ("G", "2016-12-31", "2017-03-01", 0),  # bokad sista dagen i test
        ("H", "2017-01-01", "2017-02-01", 1),  # efter testperioden, används inte
    ]
    df = pd.DataFrame(data, columns=["id", "booking_date", "arrival_date", "is_canceled"])
    df["booking_date"] = pd.to_datetime(df["booking_date"])
    df["arrival_date"] = pd.to_datetime(df["arrival_date"])
    return df


def test_get_splits_puts_each_booking_in_the_right_subsets(bookings):
    splits = {name: set(d["id"]) for name, d in get_splits(bookings).items()}
    assert splits == {
        "train_val": {"A", "B"},
        "val": {"D", "E"},
        "train_full": {"A", "B", "C", "D"},
        "test": {"F", "G"},
    }


def test_load_cleaned_data_parses_dates(tmp_path, bookings):
    path = tmp_path / "cleaned.csv"
    bookings.to_csv(path, index=False)
    df = load_cleaned_data(path)
    assert pd.api.types.is_datetime64_any_dtype(df["arrival_date"])
    assert pd.api.types.is_datetime64_any_dtype(df["booking_date"])