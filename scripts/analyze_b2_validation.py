from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = (
    PROJECT_DIR / "reports" / "encoding_comparison_v2" / "time_5m_centered"
)
PREDICTIONS_PATH = MODEL_DIR / "validation_predictions.csv"

SCENARIO_LABELS = {
    "moving": 0,
    "stop_and_resume": 0,
    "position_jump": 1,
    "gradual_offset": 1,
    "position_freeze": 1,
}


def main():
    results = pd.read_csv(PREDICTIONS_PATH)

    if (
        len(results) != 100
        or not results["sample_id"].is_unique
        or results["session_id"].nunique() != 20
        or not results["split"].eq("validation").all()
    ):
        raise ValueError("Diperlukan 100 sampel validation dari 20 sesi.")

    expected_labels = results["scenario"].map(SCENARIO_LABELS)
    if not expected_labels.eq(results["label_id"]).all():
        raise ValueError("Skenario dan label asli tidak sesuai.")

    if not results["predicted_label_id"].isin([0, 1]).all():
        raise ValueError("Ada prediksi di luar kelas 0 dan 1.")

    # Hitung ulang benar atau salah dari label dan prediksi tersimpan.
    results["correct"] = results["label_id"].eq(
        results["predicted_label_id"]
    )

    summary = results.groupby("scenario")["correct"].agg(
        samples="size",
        correct="sum",
    )
    summary["errors"] = summary["samples"] - summary["correct"]
    summary["accuracy_pct"] = 100 * summary["correct"] / summary["samples"]

    # Bandingkan normal bergerak dan drift dari sesi asal yang sama.
    selected = results.loc[
        results["scenario"].isin(["moving", "gradual_offset"])
    ]
    pairs = selected.pivot(
        index="session_id",
        columns="scenario",
        values="predicted_label_id",
    ).rename(columns={
        "moving": "moving_prediction",
        "gradual_offset": "drift_prediction",
    })

    if len(pairs) != 20 or pairs.isna().any().any():
        raise ValueError("Diperlukan satu pasangan normal–drift per sesi.")

    both_correct = (
        pairs["moving_prediction"].eq(0)
        & pairs["drift_prediction"].eq(1)
    )
    same_prediction = pairs["moving_prediction"].eq(
        pairs["drift_prediction"]
    )

    pairs["both_correct"] = both_correct
    pairs["same_prediction"] = same_prediction

    summary.to_csv(MODEL_DIR / "validation_by_scenario.csv")
    pairs.to_csv(MODEL_DIR / "validation_moving_vs_drift.csv")
    results.loc[~results["correct"]].to_csv(
        MODEL_DIR / "validation_errors.csv",
        index=False,
    )

    print("Hasil validation per skenario")
    print(summary.round(2).to_string())

    print("\nPerbandingan normal bergerak dan drift dari sesi yang sama")
    print(f"Kedua prediksi benar: {int(both_correct.sum())}/20 sesi")
    print(f"Kedua prediksi sama: {int(same_prediction.sum())}/20 sesi")
    print(f"\nFolder laporan: {MODEL_DIR}")


if __name__ == "__main__":
    main()