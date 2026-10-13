
"""Tester för den regelbaserade baselinen."""

import pandas as pd
import pytest 

from src.baseline import predict_baseline


def test_predict_baseline_flags_top_20_percent():
    """Kontrollerar att de 20 % med längst lead_time flaggas."""

    # Arrange: Skapa fem testbokningar.
    df = pd.DataFrame({
        "lead_time": [12, 150, 45, 80, 20]
    })

    # Act: Kör baselinen.
    predictions = predict_baseline(df)

    # Assert: Endast bokningen med längst lead_time ska flaggas.
    expected = [0, 1, 0, 0, 0]

    assert predictions.tolist() == expected



def test_predict_baseline_with_equal_lead_times():
    """Kontrollerar att lika ledtider inte ger för många flaggningar."""

    # Arrange: Två bokningar har samma högsta lead_time.
    df = pd.DataFrame({
        "lead_time": [30, 150, 45, 150, 20]
    })

    # Act: Flagga de 20 % med längst lead_time.
    predictions = predict_baseline(df)

    # Assert: Exakt en bokning ska flaggas.
    assert predictions.sum() == 1

    # Vid lika lead_time väljs den som kommer först.
    assert predictions.tolist() == [0, 1, 0, 0, 0]



def test_predict_baseline_missing_lead_time():
    """Kontrollerar att ett tydligt fel ges om lead_time saknas."""

    # Arrange: Skapa data utan kolumnen lead_time.
    df = pd.DataFrame({
        "adr": [100, 150, 200]
    })

    # Act & Assert: Funktionen ska ge ett ValueError.
    with pytest.raises(ValueError, match="lead_time"):
        predict_baseline(df)



def test_predict_baseline_missing_values():
    """Kontrollerar att saknade lead_time-värden ger ett tydligt fel."""

    # Arrange: En bokning saknar lead_time.
    df = pd.DataFrame({
        "lead_time": [30, 150, None, 80, 20]
    })

    # Act & Assert: Funktionen ska ge ett ValueError.
    with pytest.raises(ValueError, match="saknade värden"):
        predict_baseline(df)



def test_predict_baseline_empty_dataframe():
    """Kontrollerar att en tom DataFrame ger ett tomt resultat."""

    # Arrange: Skapa en tom DataFrame med rätt kolumn.
    df = pd.DataFrame({"lead_time": []})

    # Act: Kör baselinen.
    predictions = predict_baseline(df)

    # Assert: Inga bokningar ska flaggas.
    assert predictions.empty

