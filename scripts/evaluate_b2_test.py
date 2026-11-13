from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch import nn

from train_cnn import SmallCNN


PROJECT_DIR = Path(__file__).resolve().parent.parent
SPLIT_DIR = PROJECT_DIR / "dataset_time_5m_centered_v1" / "splits"
OUTPUT_DIR = PROJECT_DIR / "reports" / "encoding_comparison_v2" / "time_5m_centered"
CHECKPOINT_PATH = OUTPUT_DIR / "best_model.pth"

ENCODING_ID = "last_visit_time_5m_bbox_center_v1"
CLASS_NAMES = ["normal", "spoofed"]


def main():
    test = pd.read_csv(SPLIT_DIR / "test.csv").sort_values("sample_id")
    train = pd.read_csv(SPLIT_DIR / "train.csv")
    validation = pd.read_csv(SPLIT_DIR / "validation.csv")

    if len(test) != 100 or test["session_id"].nunique() != 20:
        raise ValueError("Test harus berisi 100 sampel dari 20 sesi.")

    if not test["sample_id"].is_unique or test["session_id"].isna().any():
        raise ValueError("ID sampel duplikat atau ID sesi kosong.")

    if not test["split"].eq("test").all():
        raise ValueError("Ada sampel yang tidak ditandai sebagai test.")

    if not test["encoding"].eq(ENCODING_ID).all():
        raise ValueError("Encoding pada metadata test tidak sesuai.")

    used_sessions = set(train["session_id"]) | set(validation["session_id"])
    if set(test["session_id"]) & used_sessions:
        raise ValueError("Ada sesi test yang juga digunakan pada train atau validation.")

    expected_labels = test["label"].map({"normal": 0, "spoofed": 1})
    if not expected_labels.eq(test["label_id"]).all():
        raise ValueError("Label dan label_id tidak sesuai.")

    if test["label"].value_counts().to_dict() != {"normal": 40, "spoofed": 60}:
        raise ValueError("Test harus berisi 40 normal dan 60 spoofed.")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    expected_checkpoint = {
        "architecture": "SmallCNN_1_4_8_v1",
        "encoding": ENCODING_ID,
        "input_normalization": "uint8 divided by 255",
        "epoch": 26,
    }
    for key, expected in expected_checkpoint.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"Checkpoint tidak sesuai pada bagian {key}.")

    images = []
    for row in test.itertuples(index=False):
        image = np.load(PROJECT_DIR / row.npy_path, allow_pickle=False)
        if image.shape != (32, 32) or image.dtype != np.uint8:
            raise ValueError(f"Format gambar tidak sesuai: {row.sample_id}")
        images.append(image)

    inputs = torch.from_numpy(np.stack(images)[:, None]).float() / 255.0
    actual = test["label_id"].to_numpy(dtype=np.int64, copy=True)

    model = SmallCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.inference_mode():
        logits = model(inputs)
        predicted = logits.argmax(dim=1).numpy()
        test_loss = nn.CrossEntropyLoss()(
            logits, torch.from_numpy(actual)
        ).item()

    results = test.copy()
    results["predicted_label"] = [CLASS_NAMES[index] for index in predicted]
    results["correct"] = actual == predicted

    by_scenario = results.groupby("scenario")["correct"].agg(
        samples="size", correct="sum"
    )
    by_scenario["errors"] = by_scenario["samples"] - by_scenario["correct"]
    by_scenario["accuracy_pct"] = (
        100 * by_scenario["correct"] / by_scenario["samples"]
    )

    matrix = pd.DataFrame(
        confusion_matrix(actual, predicted, labels=[0, 1]),
        index=["true_normal", "true_spoofed"],
        columns=["pred_normal", "pred_spoofed"],
    )
    report = classification_report(
        actual,
        predicted,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    results.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    matrix.to_csv(OUTPUT_DIR / "test_confusion_matrix.csv")
    by_scenario.to_csv(OUTPUT_DIR / "test_by_scenario.csv")

    summary = (
        f"Checkpoint epoch: {checkpoint['epoch']}\n"
        f"Test loss: {test_loss:.4f}\n"
        f"Test accuracy: {results['correct'].mean():.1%}\n"
        f"Balanced accuracy: {balanced_accuracy_score(actual, predicted):.2%}\n"
        f"\nClassification report\n{report}"
        f"\nConfusion matrix\n{matrix.to_string()}\n"
        f"\nHasil per skenario\n{by_scenario.round(2).to_string()}\n"
    )
    (OUTPUT_DIR / "test_report.txt").write_text(summary, encoding="utf-8")

    print(summary)
    print(f"Folder hasil: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()