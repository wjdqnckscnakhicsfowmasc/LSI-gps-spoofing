from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_cnn import SmallCNN


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "reports" / "encoding_comparison_v2"

DATASETS = {
    "density_10m": (
        "dataset_sessions_v2",
        "fixed_density_10m_center16_v1",
    ),
    "time_5m_centered": (
        "dataset_time_5m_centered_v1",
        "last_visit_time_5m_bbox_center_v1",
    ),
}

SEED = 2032
EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 0.001
MATCH_COLUMNS = ["sample_id", "session_id", "label_id"]


def load_split(dataset_dir, encoding_id, split_name):
    metadata = pd.read_csv(
        PROJECT_DIR / dataset_dir / "splits" / f"{split_name}.csv"
    ).sort_values("sample_id").reset_index(drop=True)

    expected_count = {"train": 300, "validation": 100}[split_name]

    if len(metadata) != expected_count or not metadata["sample_id"].is_unique:
        raise ValueError(f"Jumlah atau ID sampel tidak sesuai: {dataset_dir}/{split_name}")

    if not metadata["encoding"].eq(encoding_id).all():
        raise ValueError(f"Encoding tidak sesuai: {dataset_dir}/{split_name}")

    images = []

    for row in metadata.itertuples(index=False):
        image = np.load(PROJECT_DIR / row.npy_path, allow_pickle=False)

        if image.shape != (32, 32) or image.dtype != np.uint8:
            raise ValueError(f"Format gambar tidak sesuai: {row.sample_id}")

        images.append(image)

    inputs = torch.from_numpy(np.stack(images)[:, None]).float() / 255.0
    labels = torch.tensor(metadata["label_id"].to_numpy(), dtype=torch.long)
    return metadata, inputs, labels


def train_one(name, encoding_id, train, validation):
    _, train_images, train_labels = train
    val_metadata, val_images, val_labels = validation

    torch.manual_seed(SEED)
    model = SmallCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loader = DataLoader(
        TensorDataset(train_images, train_labels),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0,
    )

    model_dir = OUTPUT_DIR / name
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "best_model.pth"

    best_loss = float("inf")
    history = []

    print(f"\nTraining: {name}")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        for images, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_logits = model(train_images)
            val_logits = model(val_images)
            train_loss = criterion(train_logits, train_labels).item()
            val_loss = criterion(val_logits, val_labels).item()
            val_accuracy = (
                val_logits.argmax(dim=1) == val_labels
            ).float().mean().item()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
        })

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "architecture": "SmallCNN_1_4_8_v1",
                "encoding": encoding_id,
                "input_normalization": "uint8 divided by 255",
                "epoch": epoch,
                "validation_loss": val_loss,
                "seed": SEED,
            }, checkpoint_path)

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"Epoch {epoch:3d}/{EPOCHS} | train loss={train_loss:.4f} | "
                f"val loss={val_loss:.4f} | val acc={val_accuracy:.1%}"
            )

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        predicted = model(val_images).argmax(dim=1).numpy()

    actual = val_labels.numpy()
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    true_normal, false_spoofed, false_normal, true_spoofed = matrix.ravel()

    predictions = val_metadata.copy()
    predictions["predicted_label_id"] = predicted
    predictions["correct"] = actual == predicted
    predictions.to_csv(model_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(model_dir / "training_history.csv", index=False)

    return {
        "encoding": name,
        "best_epoch": checkpoint["epoch"],
        "validation_loss": checkpoint["validation_loss"],
        "accuracy": float(np.mean(actual == predicted)),
        "balanced_accuracy": balanced_accuracy_score(actual, predicted),
        "macro_f1": f1_score(actual, predicted, average="macro", zero_division=0),
        "true_normal": int(true_normal),
        "false_spoofed": int(false_spoofed),
        "false_normal": int(false_normal),
        "true_spoofed": int(true_spoofed),
    }


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

    prepared = {}

    for name, (dataset_dir, encoding_id) in DATASETS.items():
        train = load_split(dataset_dir, encoding_id, "train")
        validation = load_split(dataset_dir, encoding_id, "validation")

        if set(train[0]["session_id"]) & set(validation[0]["session_id"]):
            raise ValueError(f"Sesi train dan validation bertumpang tindih: {name}")

        prepared[name] = (encoding_id, train, validation)

    _, density_train, density_val = prepared["density_10m"]
    _, time_train, time_val = prepared["time_5m_centered"]

    for density, time in ((density_train, time_train), (density_val, time_val)):
        if not density[0][MATCH_COLUMNS].equals(time[0][MATCH_COLUMNS]):
            raise ValueError("Sampel, sesi, atau label berbeda antar-encoding.")

    results = [
        train_one(name, encoding_id, train, validation)
        for name, (encoding_id, train, validation) in prepared.items()
    ]

    summary = pd.DataFrame(results)
    summary_path = OUTPUT_DIR / "validation_comparison.csv"
    summary.to_csv(summary_path, index=False)

    print("\nPerbandingan validation")
    print(summary.round(4).to_string(index=False))
    print(f"\nLaporan: {summary_path}")


if __name__ == "__main__":
    main()