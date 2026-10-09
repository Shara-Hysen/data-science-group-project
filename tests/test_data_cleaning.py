"""
tests/test_data_cleaning.py
Enhetstester för datastädningen och Pandera-valideringen. 
"""

import pandas as pd
import pandera.pandas as pa
import pytest

from src.data_cleaning import (
    cleaned_data_schema,
    derive_dates,
    filter_incomplete_date_range,
    filter_invalid_rows,
    handle_missing_values
)


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Skapar en liten representativ DataFrame för testning."""
    return pd.DataFrame(
        {
            "adults": [2, 0, 1, 1],
            "children": [1.0, 0.0, None, 0.0],
            "babies": [0, 0, 0, 0],
            "country": ["PRT", None, "FRA", "DEU"],
            "agent": [9.0, None, 1.0, 0.0],
            "company": [None, None, 40.0, 0.0],
            "adr": [100.0, 50.0, -5.0, 6000.0],
            "lead_time": [30, 10, 5, 20],
            "arrival_date_year": [2016, 2015, 2016, 2015],
            "arrival_date_month": ["July", "August", "May", "January"],
            "arrival_date_day_of_month": [15, 1, 10, 5]
        }
    )


def test_handle_missing_values(sample_raw_data: pd.DataFrame) -> None:
    """Säkerställer att saknade värden fylls med förväntade standardvärden."""
    cleaned = handle_missing_values(sample_raw_data)

    assert cleaned["children"].isna().sum() == 0
    assert cleaned.loc[2, "children"] == 0.0
    assert cleaned["country"].isna().sum() == 0
    assert cleaned.loc[1, "country"] == "Unknown"
    assert cleaned.loc[1, "agent"] == 0.0
    assert cleaned.loc[0, "company"] == 0.0


def test_filter_invalid_rows(sample_raw_data: pd.DataFrame) -> None:
    """Verifierar att rader med 0 gäster och orimliga adr-värden rensas bort."""
    cleaned = filter_invalid_rows(sample_raw_data)

    # Rad 1 (0 gäster), rad 2 (adr < 0) och rad 3 (adr > 5000) ska tas bort 
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["adults"] == 2
    assert cleaned.iloc[0]["adr"] == 100.0


def test_derive_dates(sample_raw_data: pd.DataFrame) -> None:
    """Verifierar korrekt datumkonvertering och härledning av booking_date."""
    dated = derive_dates(sample_raw_data)

    assert pd.api.types.is_datetime64_any_dtype(dated["arrival_date"])
    assert pd.api.types.is_datetime64_any_dtype(dated["booking_date"])

    # 2016-07-15 minus 30 dagar lead_time = 2016-06-15
    assert dated.loc[0, "arrival_date"] == pd.Timestamp("2016-07-15")
    assert dated.loc[0, "booking_date"] == pd.Timestamp("2016-06-15")


def test_filter_incomplete_date_range() -> None:
    """Kontrollerar att rader med booking_date före 2015-07-01 rensas ut."""
    df = pd.DataFrame(
        {
            "booking_date": [
                pd.Timestamp("2015-06-30"),
                pd.Timestamp("2015-07-01"),
                pd.Timestamp("2016-01-01")
            ]
        }
    )
    filtered = filter_incomplete_date_range(df)

    assert len(filtered) == 2
    assert (filtered["booking_date"] >= pd.Timestamp("2015-07-01")).all()


def test_cleaned_data_schema_valid() -> None: 
    """Verifierar att godkänd städad data passerar Pandera-schemat."""
    valid_df = pd.DataFrame(
        {
            "adults": [2],
            "children": [0.0],
            "babies": [0],
            "country": ["PRT"],
            "agent": [1.0],
            "company": [0.0],
            "adr": [120.5],
            "arrival_date": [pd.Timestamp("2016-08-10")],
            "booking_date": [pd.Timestamp("2016-08-01")]
        }
    )
    validated = cleaned_data_schema.validate(valid_df)
    assert isinstance(validated, pd.DataFrame)


def test_cleaned_data_schema_catches_zero_guests() -> None: 
    """Verifierar att Schemat fångar upp och kastar fel om 0 gäster förekommer."""
    invalid_df = pd.DataFrame(
        {
            "adults": [0],
            "children": [0.0],
            "babies": [0],
            "country": ["PRT"],
            "agent": [1.0],
            "company": [0.0],
            "adr": [120.5],
            "arrival_date": [pd.Timestamp("2016-08-10")],
            "booking_date": [pd.Timestamp("2016-08-01")]
        }
    )
    with pytest.raises(pa.errors.SchemaError):
        cleaned_data_schema.validate(invalid_df)