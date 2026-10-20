"""
tests/test_evaluation.py
Enhetstester för modulens gemensamma flaggnings- och utvärderingsfunktioner.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")   # Förhindrar Tkinter/GUI-fel under tester
import matplotlib.pyplot as plt

from src.evaluation import flag_top_share, evaluate_top_share, plot_pr_curve


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
    assert results["pr_auc"] == pytest.approx(11 / 12)


def test_plot_pr_curve():
    """Verifierar att plot_pr_curve beräknar PR-AUC och returnerar rätt struktur."""
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    predictions = {
        # Modell A: Bra separation (höga värden på ettor, låga på nollor)
        "Modell A": np.array([0.1, 0.9, 0.2, 0.8, 0.85, 0.15, 0.75, 0.1]),
        # Modell B: Dålig separation (sätter t.ex. hög sannolikhet på en nolla och låg på en etta)
        "Modell B": np.array([0.8, 0.2, 0.7, 0.3, 0.4, 0.9, 0.2, 0.6])
    }

    results = plot_pr_curve(y_true, predictions)

    # Stäng figuren
    plt.close("all")

    # Kontrollera att båda modellerna finns i resultatet
    assert "Modell A" in results
    assert "Modell B" in results

    # Kontrollera att PR-AUC beräknas och ligger i ett rimligt intervall [0, 1]
    for model_name, metrics in results.items():
        assert "pr_auc" in metrics
        assert 0.0 <= metrics["pr_auc"] <= 1.0

    # Modell A har nu garanterat bättre separation och högre PR-AUC än Modell B
    assert results["Modell A"]["pr_auc"] > results["Modell B"]["pr_auc"]



@pytest.mark.filterwarnings("error")
def test_evaluate_top_share_without_flags():
    y_true = np.array([0, 1, 1])
    scores = np.array([10, 20, 30])

    results = evaluate_top_share(y_true, scores, share=0)

    assert results["antal_flaggade"] == 0
    assert results["precision"] == 0.0
    assert results["recall"] == 0.0
    assert results["f1_score"] == 0.0