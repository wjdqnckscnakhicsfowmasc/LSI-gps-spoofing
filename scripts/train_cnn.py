from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gps_encoder import ENCODING_ID


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset_sessions_v1"
OUTPUT_DIR = PROJECT_DIR / "reports" / "cnn_baseline_v1"

SEED = 2032
EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 0.001


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(8 * 8 * 8, 2)

    def forward(self, images):
        features = self.features(images)
        return self.classifier(torch.flatten(features, start_dim=1))


def load_split(name):
    metadata = pd.read_csv(
        DATASET_DIR / "splits" / f"{name}.csv"
    )

    if metadata.empty or not metadata["sample_id"].is_unique:
        raise ValueError(f"Metadata {name} kosong atau duplikat.")

    if not metadata["encoding"].eq(ENCODING_ID).all():
        raise ValueError(f"Encoding {name} tidak sesuai.")

    images = []
    labels = []

    for _, record in metadata.iterrows():
        image = np.load(
            PROJECT_DIR / record["npy_path"],
            allow_pickle=False,
        )

        if image.shape != (32, 32) or image.dtype != np.uint8:
            raise ValueError(f"Format salah: {record['sample_id']}")

        label = int(record["label_id"])
        expected_label = {"normal": 0, "spoofed": 1}.get(record["label"])

        if label != expected_label:
            raise ValueError("Label tidak sesuai.")

        images.append(image.astype(np.float32) / 255.0)
        labels.append(label)

    image_tensor = torch.from_numpy(
        np.stack(images)[:, None, :, :]
    )
    label_tensor = torch.tensor(labels, dtype=torch.long)

    return metadata, image_tensor, label_tensor


def evaluate(model, images, labels, criterion):
    model.eval()

    with torch.no_grad():
        logits = model(images)
        loss = criterion(logits, labels).item()
        predictions = logits.argmax(dim=1)
        accuracy = (
            predictions == labels
        ).float().mean().item()

    return loss, accuracy, predictions.numpy()


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

    train_meta, train_images, train_labels = load_split("train")
    val_meta, val_images, val_labels = load_split("validation")

    if set(train_meta["session_id"]) & set(val_meta["session_id"]):
        raise ValueError("Sesi train dan validation bertumpang tindih.")

    if set(train_meta["sample_id"]) & set(val_meta["sample_id"]):
        raise ValueError("Sampel train dan validation bertumpang tindih.")

    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        TensorDataset(train_images, train_labels),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    model = SmallCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / "best_model.pth"

    best_val_loss = float("inf")
    best_epoch = 0
    history = []

    print("Training menggunakan CPU.")
    print(f"Train: {len(train_meta)}, validation: {len(val_meta)}")
    print(f"Jumlah parameter: {sum(p.numel() for p in model.parameters())}")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        for images, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        train_loss, train_accuracy, _ = evaluate(
            model, train_images, train_labels, criterion
        )
        val_loss, val_accuracy, _ = evaluate(
            model, val_images, val_labels, criterion
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "train_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
        })

        # Pilih model berdasarkan loss validation terendah.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            torch.save({
                "model_state_dict": model.state_dict(),
                "architecture": "SmallCNN_1_4_8_v1",
                "encoding": ENCODING_ID,
                "input_normalization": "uint8 divided by 255",
                "class_names": ["normal", "spoofed"],
                "seed": SEED,
                "epoch": epoch,
                "validation_loss": val_loss,
            }, checkpoint_path)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d}/{EPOCHS} | "
                f"train loss={train_loss:.4f}, "
                f"acc={train_accuracy:.1%} | "
                f"val loss={val_loss:.4f}, "
                f"acc={val_accuracy:.1%}"
            )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    _, val_accuracy, predictions = evaluate(
        model, val_images, val_labels, criterion
    )

    report = classification_report(
        val_labels.numpy(),
        predictions,
        labels=[0, 1],
        target_names=["normal", "spoofed"],
        zero_division=0,
        digits=3,
    )

    matrix = confusion_matrix(
        val_labels.numpy(),
        predictions,
        labels=[0, 1],
    )

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(OUTPUT_DIR / "training_history.csv", index=False)

    val_results = val_meta.copy()
    val_results["predicted_label_id"] = predictions
    val_results["correct"] = predictions == val_labels.numpy()
    val_results.to_csv(
        OUTPUT_DIR / "validation_predictions.csv",
        index=False,
    )

    pd.DataFrame(
        matrix,
        index=["true_normal", "true_spoofed"],
        columns=["pred_normal", "pred_spoofed"],
    ).to_csv(OUTPUT_DIR / "validation_confusion_matrix.csv")

    (OUTPUT_DIR / "validation_report.txt").write_text(
        f"Best epoch: {best_epoch}\n\n{report}",
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history_frame["epoch"], history_frame["train_loss"],
                 label="Train")
    axes[0].plot(history_frame["epoch"],
                 history_frame["validation_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history_frame["epoch"],
                 history_frame["train_accuracy"], label="Train")
    axes[1].plot(history_frame["epoch"],
                 history_frame["validation_accuracy"], label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
    plt.close(figure)

    print()
    print(f"Training selesai. Model terpilih dari epoch {best_epoch}.")
    print(f"Validation accuracy: {val_accuracy:.1%}")
    print(report)
    print(f"Folder hasil: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()