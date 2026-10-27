"""
src/model_io.py

Hanterar sparning och inläsning av modellartefakter (tränade modeller och
anpassade preprocessorer) med joblib.
"""

from pathlib import Path
from typing import Any
import joblib

from src.config import get_logger

logger = get_logger(__name__)

# Standardkatalog och filnamn under outputs/
DEFAULT_MODEL_DIR = Path("outputs") / "models"
DEFAULT_MODEL_NAME = "final_model_rf.joblib"
DEFAULT_PREPROCESSOR_NAME = "preprocessor.joblib"


def save_model_artifacts(
    model: Any,
    preprocessor: Any,
    output_dir: Path | str = DEFAULT_MODEL_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    preprocessor_name: str = DEFAULT_PREPROCESSOR_NAME,
) -> None:
    """
    Sparar tränad modell och anpassad preprocessor till disk med joblib.

    Tar emot generiska Python-objekt (Any) för att inte låsa modellen
    till specifika scikit-learn-klasser. Skapar målmappen om den inte redan finns.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Sparar modellartefakter till %s...", out_path)
    joblib.dump(model, out_path / model_name)
    joblib.dump(preprocessor, out_path / preprocessor_name)
    logger.info("Modell (%s) och preprocessor (%s) sparades framgångsrikt.", model_name, preprocessor_name)


def load_model_artifacts(
    input_dir: Path | str = DEFAULT_MODEL_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    preprocessor_name: str = DEFAULT_PREPROCESSOR_NAME,
) -> tuple[Any, Any]:
    """
    Laddar sparad modell och anpassad preprocessor från disk med joblib.

    Returnerar:
        tuple[model, preprocessor]
    """
    in_path = Path(input_dir)
    model_path = in_path / model_name
    preprocessor_path = in_path / preprocessor_name

    if not model_path.exists() or not preprocessor_path.exists():
        logger.error("Kunde inte hitta modellartefakter i %s", in_path.resolve())
        raise FileNotFoundError(
            f"Kunde inte hitta modellartefakter i {in_path.resolve()}. "
            "Kör 02_analysis_and_results.ipynb först för att träna och spara dem."
        )

    logger.info("Läser in modellartefakter från %s...", in_path)
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    logger.info("Artefakter inlästa framgångsrikt.")

    return model, preprocessor