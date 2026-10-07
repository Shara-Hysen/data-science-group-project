"""
tests/test_data_loader.py
Enhetstester för modulens datainläsning.
"""

import pandas as pd
import pytest
from src.data_loader import load_raw_data


def test_load_raw_data_returns_dataframe():
    """Testar att rådatan läses in som en icke-tom pandas DataFrame
    och innehåller förväntade kolummer.
    """
    df = load_raw_data()

    # Kontrollera typ
    assert isinstance(df, pd.DataFrame), "Returvärdet ska vara en pandas DataFrame"

    # Kontrollera att datan inte är tom
    assert not df.empty, "Datasetet får inte vara tomt"

    # Kontrollera att målvariabeln och centrala kolumner finns med
    assert "is_canceled" in df.columns, "Kolumnen 'is_canceled' saknas i rådatan"
    assert "hotel" in df.columns, "Kolumnen 'hotel' saknas i rådatan"


def test_load_raw_data_raises_when_nocsv(monkeypatch):
    """Säkerställer att FileNotFoundError reses om katalogen saknar CSV-filer."""
    # Simulera att mappen inte innehåller någon .csv-fil utan att faktiskt röra filsystemet
    monkeypatch.setattr("os.listdir", lambda path: ["other_file.txt"])

    with pytest.raises(FileNotFoundError):
        load_raw_data()