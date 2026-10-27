"""
scripts/export_demo_booking.py

Engångsskript för att deterministiskt identifiera och exportera en representativ
högriskbokning (~70 % predicerad avbokningsrisk) ur testmängden till en fristående
CSV-fil för 06_demonstration.ipynb.
"""

from pathlib import Path

from src.data_split import get_splits, load_cleaned_data
from src.model_io import load_model_artifacts
from src.preprocessing import split_features_target

OUTPUT_PATH = Path("data/demo_booking.csv")


def export_representative_booking(
    target_risk_min: float = 0.69,
    target_risk_max: float = 0.71,
    output_path: Path = OUTPUT_PATH,
) -> None:
    # 1. Ladda artefakter och testdata
    rf_model, preprocessor = load_model_artifacts()
    test_data = get_splits(load_cleaned_data())["test"]

    # 2. Transformera och beräkna sannolikheter
    X_test, _ = split_features_target(test_data, include_deposit_type=False)
    X_processed = preprocessor.transform(X_test)
    probas = rf_model.predict_proba(X_processed)[:, 1]

    # 3. Hämta första bästa kandidat i önskat intervall
    mask = (probas >= target_risk_min) & (probas <= target_risk_max)
    candidate_idx = test_data.index[mask][0]

    demo_row = test_data.loc[[candidate_idx]].copy()

    # 4. Spara till CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    demo_row.to_csv(output_path, index=True)

    actual_risk = probas[mask][0]
    print(f"Sparade bokning #{candidate_idx} med {actual_risk:.1%} risk till {output_path}")


if __name__ == "__main__":
    export_representative_booking()