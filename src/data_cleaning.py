"""
src/data_cleaning.py
Modul och körbart skript för datastädning och validering.

Modulen ansvarar för att förbereda rådata inför kronologisk train/test-split:
1. Hantera basala saknade värden på radnivå (children, country, agent, company).
2. Filtrera bort orimliga observationer (noll gäster och ogiltiga adr-värden).
3. Härleda arrival_date och booking_date (arrival_date - lead_time).
4. Filtrera bort ofullständiga bokningar före 2015-07-01.
5. Validera städresultatet mot Pandera-schemat cleaned_data_schema.
6. Spara bearbetad data till disk.

Samtliga originalkolumner behålls för vidare hantering i preprocessing-steget.
"""

from pathlib import Path
import pandas as pd
import pandera.pandas as pa

from src.config import get_logger
from src.data_loader import load_raw_data

logger = get_logger(__name__)

OUTPUT_PATH = Path("data/processed/hotel_bookings_cleaned.csv")


cleaned_data_schema = pa.DataFrameSchema(
    columns={
        "children": pa.Column(float, nullable=False),
        "country": pa.Column(str, nullable=False),
        "agent": pa.Column(float, nullable=False),
        "company": pa.Column(float, nullable=False),
        "adr": pa.Column(float, checks=[pa.Check.ge(0), pa.Check.lt(5000)], nullable=False),
        "arrival_date": pa.Column(pa.DateTime, nullable=False),
        "booking_date": pa.Column(
            pa.DateTime,
            checks=pa.Check.ge(
                pd.Timestamp("2015-07-01"),
                error="Det finns bokningar daterade före 2015-07-01 kvar i datasetet"
            ),
            nullable=False
        )
    },
    checks=[
        pa.Check(
            lambda df: (df["adults"] + df["children"] + df["babies"]) > 0,
            name="check_at_least_one_guest",
            error="Det finns bokningar med 0 gäster kvar i datan",
        ),
        pa.Check(
            lambda df: df["booking_date"] <= df["arrival_date"],
            name="check_booking_date_chronology",
            error="booking_date kan inte inträffa efter arrival_date"
        )
    ],
    strict=False,
    coerce=True,
    name="CleanedHotelDataSchema"
)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Hanterar basala saknade värden för children, country, agent och company."""
    logger.info("Hanterar saknade värden...")
    data = df.copy()

    data["children"] = data["children"].fillna(0.0)
    data["country"] = data["country"].fillna("Unknown")
    data["agent"] = data["agent"].fillna(0.0)
    data["company"] = data["company"].fillna(0.0)

    return data


def filter_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Tar bort rader med 0 gäster samt orimliga dagspriser."""
    logger.info("Filtrerar bort orimliga rader...")
    initial_rows = len(df)

    has_guests = (df["adults"] + df["children"].fillna(0.0) + df["babies"]) > 0
    valid_adr = (df["adr"] >= 0.0) & (df["adr"] < 5000.0)

    filtered_df = df[has_guests & valid_adr].copy()
    removed = initial_rows - len(filtered_df)
    logger.info("Tog bort %d rader. Kvarvarande: %d.", removed, len(filtered_df))

    return filtered_df


def derive_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Skapar arrival_date och härleder booking_date (arrival_date - lead_time)."""
    logger.info("Härleder arrival_date och booking_date...")
    data = df.copy()

    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12,
    }
    # Steg 1: Mappa månadsnamn till siffror
    month_num = data["arrival_date_month"].map(month_map)

    # Steg 2: Bygg arrival_date
    data["arrival_date"] = pd.to_datetime(
        data["arrival_date_year"].astype(str)
        + "-"
        + month_num.astype(str)
        + "-"
        + data["arrival_date_day_of_month"].astype(str)
    )

    data["booking_date"] = data["arrival_date"] - pd.to_timedelta(
        data["lead_time"], unit="D"
    )
    return data


def filter_incomplete_date_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtrerar bort bokningar gjorda före 2015-07-01.

    Observationer före detta datum är ofullständiga (endast långa ledtider
    finns representerade) och innehåller skeva blockbokningar från 2014.
    """
    logger.info("Filtrerar bort ofullständiga bokningar före 2015-07-01...")
    initial_rows = len(df)

    cutoff_date = pd.Timestamp("2015-07-01")
    filtered_df = df[df["booking_date"] >= cutoff_date].copy()

    removed = initial_rows - len(filtered_df)
    logger.info(
        "Tog bort %d observationer före %s. Kvarvarande: %d",
        removed,
        cutoff_date.date(),
        len(filtered_df)
    )
    return filtered_df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validerar att städningen uppfyller kraven i cleaned_data_schema."""
    logger.info("Validerar städad data med Pandera...")
    try:
        validated_df = cleaned_data_schema.validate(df, lazy=True)
        logger.info("Datavalidering godkänd utan anmärkningar.")
        return validated_df
    except pa.errors.SchemaErrors as exc:
        logger.error("Valideringen misslyckades!\n%s", exc.failure_cases)
        raise exc


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Kör alla städningssteg i sekvens och validerar resultatet."""
    logger.info("Startar städningspipeline...")
    return (
        df.pipe(handle_missing_values)
        .pipe(filter_invalid_rows)
        .pipe(derive_dates)
        .pipe(filter_incomplete_date_range)
        .pipe(validate_data)
    )


def save_cleaned_data(df: pd.DataFrame, output_path: Path = OUTPUT_PATH) -> None:
    """Sparar städad och validerad data till fil."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Sparar städad data till %s", output_path)

    df.to_csv(output_path, index=False)
    logger.info("Data sparades framgångsrikt.")


if __name__ == "__main__":
    raw_df = load_raw_data()
    cleaned_df = clean_raw_data(raw_df)
    save_cleaned_data(cleaned_df)