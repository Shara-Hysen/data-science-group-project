"""
tests/test_experiment_logger.py

Enhetstester för experiment_logger.
Verifierar CSV-loggning, tillägg av nya rader och sparande av figurer
utan att påverka projektets riktiga output-kataloger.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.experiment_logger import get_leaderboard, log_experiment, save_plot


@pytest.fixture
def temp_output_dir(tmp_path: Path):
    """Skapar en temporär arbetskatalog för testerna."""
    return tmp_path


def test_log_experiment_creates_file_and_logs_row(temp_output_dir: Path):
    """Testar att en ny CSV-fil skapas och att värden sparas korrekt."""
    csv_file = temp_output_dir / "experiment_results.csv"
    metrics = {
        "precision": 0.75123,
        "recall": 0.60456,
        "f1_score": 0.67001,
        "antal_flaggade": 100,
        "korrekt_flaggade": 75,
    }

    df_result = log_experiment(
        model_name="TestModel",
        feature_set="test_features",
        metrics=metrics,
        dataset="val",
        notes="Första testet",
        filepath=csv_file,
    )

    assert csv_file.exists()
    assert isinstance(df_result, pd.DataFrame)
    assert df_result.iloc[0]["model"] == "TestModel"

    df_read = pd.read_csv(csv_file)
    assert len(df_read) == 1
    assert df_read.iloc[0]["model"] == "TestModel"
    assert df_read.iloc[0]["precision_at_20"] == 0.7512  
    assert df_read.iloc[0]["recall_at_20"] == 0.6046


def test_log_experiment_appends_row_without_overwriting(temp_output_dir: Path):
    """Testar att flera körningar läggs till efter varandra (mode='a')."""
    csv_file = temp_output_dir / "experiment_results.csv"

    log_experiment(
        model_name="Model_1",
        feature_set="f1",
        metrics={"recall": 0.50},
        filepath=csv_file,
    )
    log_experiment(
        model_name="Model_2",
        feature_set="f2",
        metrics={"recall": 0.85},
        filepath=csv_file,
    )

    df_read = pd.read_csv(csv_file)
    assert len(df_read) == 2
    assert df_read.iloc[0]["model"] == "Model_1"
    assert df_read.iloc[1]["model"] == "Model_2"


def test_get_leaderboard_sorting(temp_output_dir: Path):
    """Testar att get_leaderboard sorterar efter fallande recall_at_20."""
    csv_file = temp_output_dir / "experiment_results.csv"

    log_experiment(
        model_name="LowRecall",
        feature_set="f1",
        metrics={"recall": 0.40},
        filepath=csv_file,
    )
    log_experiment(
        model_name="HighRecall",
        feature_set="f2",
        metrics={"recall": 0.90},
        filepath=csv_file,
    )

    leaderboard = get_leaderboard(filepath=csv_file)
    assert leaderboard.iloc[0]["model"] == "HighRecall"
    assert leaderboard.iloc[1]["model"] == "LowRecall"


def test_save_plot_creates_file(temp_output_dir: Path):
    """Testar att save_plot sparar bildfilen med korrekt format och stänger den."""
    figures_dir = temp_output_dir / "figures"

    # Skapa en enkel testfigur
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])
    ax.set_title("Test Plot")

    saved_path = save_plot(
        filename="test_curve",  # Utan .png för att testa att filtillägget läggs till
        fig=fig,
        output_dir=figures_dir,
        close_fig=True,
    )

    assert saved_path.exists()
    assert saved_path.name == "test_curve.png"
    assert figures_dir.exists()