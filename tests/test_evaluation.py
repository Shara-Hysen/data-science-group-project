"""
tests/test_evaluation.py
Enhetstester för modulens gemensamma flaggnings- och utvärderingsfunktioner.
"""

import numpy as np
import pytest

from src.evaluation import flag_top_share, evaluate_top_share


def test_flag_top_share_flags_top_20_percent():
    """Kontrollerar att de 20 % med högst riskpoäng flaggas."""

    # Arrange: Skapa fem bokningar med olika riskpoäng.
    scores = np.array([12, 150, 45, 80, 20])

    # Act: Flagga de 20 % med högst riskpoäng.
    predictions = flag_top_share(scores)

    # Assert: Endast bokningen med högst riskpoäng ska flaggas.
    expected = [0, 1, 0, 0, 0]

    assert predictions.tolist() == expected


def test_flag_top_share_with_equal_scores():
    """Kontrollerar att lika riskpoäng inte ger för många flaggningar."""

    # Arrange: Två bokningar har samma högsta riskpoäng.
    scores = np.array([30, 150, 45, 150, 20])

    # Act: Flagga de 20 % med högst riskpoäng.
    predictions = flag_top_share(scores)

    # Assert: Exakt en bokning ska flaggas.
    assert predictions.sum() == 1

    # Vid lika riskpoäng väljs den som kommer först.
    assert predictions.tolist() == [0, 1, 0, 0, 0]


def test_evaluate_top_share():
    """Kontrollerar att precision och recall beräknas korrekt."""

    # Arrange: Fem bokningar med riskpoäng och verkliga utfall.
    scores = np.array([20, 150, 60, 200, 90])
    y_true = np.array([0, 1, 1, 1, 0])

    # Act: Utvärdera de 20 % med högst riskpoäng.
    results = evaluate_top_share(y_true, scores)

    # Assert: En bokning flaggas och den är korrekt.
    assert results["antal_bokningar"] == 5
    assert results["antal_flaggade"] == 1
    assert results["korrekt_flaggade"] == 1
    assert results["precision"] == 1.0
    assert results["recall"] == pytest.approx(1 / 3)