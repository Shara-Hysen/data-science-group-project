"""
src/data_loader.py
Modul för hämtning och inläsning av rådata från Kaggle.
Dataset: Hotel Booking Demand (jessemostipak/hotel-booking-demand)
"""

import os
import kagglehub
import pandas as pd
from src.config import get_logger

logger = get_logger(__name__)


def load_raw_data() -> pd.DataFrame:
    """
    Laddar ner datasetet via kagglehub (om det inte redan finns cachat)
    och returnerar datan som en pandas DataFrame.
    """
    logger.info("Kontrollerar/hämtar dataset via kagglehub...")
    try:
        # kagglehub sparar datasetet i en lokal cachemapp och returnerar mappens sökväg
        dataset_path = kagglehub.dataset_download(
            "jessemostipak/hotel-booking-demand"
        )
    except Exception as exc:
        logger.error("Fel vid nedladdning via kagglehub: %s", exc)
        raise exc

    # Leta upp CSV-filen dynamiskt i cachen för att undvika hårdkodade lokala sökvägar
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
    if not csv_files:
        msg = f"Ingen CSV-fil hittades i katalogen: {dataset_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    # Bygg ihop fullständig sökväg med rätt separator oberoende av operativsystem
    full_csv_path = os.path.join(dataset_path, csv_files[0])
    logger.info("Läser in rådata från: %s", full_csv_path)

    df = pd.read_csv(full_csv_path)
    logger.info("Data inläst framgångsrikt. Dimensioner: %s", df.shape)

    return df

if __name__ == "__main__":
    load_raw_data()