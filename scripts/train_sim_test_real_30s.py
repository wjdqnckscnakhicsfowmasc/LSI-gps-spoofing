from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_DIR = Path(__file__).resolve().parent.parent
SIM_DIR = PROJECT_DIR / "dataset_sim_time_5m_centered_30s_v1"
REAL_DIR = PROJECT_DIR / "dataset_real_time_5m_centered_30s_v1"
OUTPUT_DIR = PROJECT_DIR / "reports" / "sim_real_30s_v1"

ENCODING = "last_visit_time_5m_bbox_center_v1"

SPLIT_SEED = 2031
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


def split_sessions(metadata):
    # Pembagian per sesi 60/20/20, semua skenario satu sesi ikut bersama.
    sessions = np.array(sorted(metadata["session_id"].unique()))
    shuffled = np.random.default_rng(SPLIT_SEED).permutation(sessions)

    n_train = round(len(shuffled) * 0.6)
    n_val = round(len(shuffled) * 0.2)

    groups = {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }

    session_to_split = {
        session_id: name
        for name, group in groups.items()
        for session_id in group
    }

    combined = metadata.copy()
    combined["split"] = combined["session_id"].map(session_to_split)

    if combined["split"].isna().any():
        raise ValueError("Ada sampel simulasi tanpa kelompok.")

    parts = {
        name: combined.loc[combined["split"] == name]
        .sort_values(["session_id", "sample_id"])
        .reset_index(drop=True)
        for name in groups
    }

    for a in parts:
        for b in parts:
            if a < b and set(parts[a]["session_id"]) & set(parts[b]["session_id"]):
                raise ValueError(f"Sesi {a} dan {b} bertumpang tindih.")

    return parts


def load_images(metadata, name):
    if metadata.empty or not metadata["sample_id"].is_unique:
        raise ValueError(f"Metadata {name} kosong atau duplikat.")

    if not metadata["encoding"].eq(ENCODING).all():
        raise ValueError(f"Encoding {name} tidak sesuai.")

    images = []
    labels = []

    for _, record in metadata.iterrows():
        image = np.load(PROJECT_DIR / record["npy_path"], allow_pickle=False)

        if image.shape != (32, 32) or image.dtype != np.uint8:
            raise ValueError(f"Format salah: {record['sample_id']}")

        label = int(record["label_id"])
        if label != {"normal": 0, "spoofed": 1}.get(record["label"]):
            raise ValueError(f"Label tidak sesuai: {record['sample_id']}")

        images.append(image.astype(np.float32) / 255.0)
        labels.append(label)

    image_tensor = torch.from_numpy(np.stack(images)[:, None, :, :])
    label_tensor = torch.tensor(labels, dtype=torch.long)

    return image_tensor, label_tensor


def evaluate(model, images, labels, criterion):
    model.eval()

    with torch.no_grad():
        logits = model(images)
        loss = criterion(logits, labels).item()
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == labels).float().mean().item()

    return loss, accuracy, predictions.numpy(), probabilities.numpy()


def per_scenario(results):
    summary = (
        results.groupby("scenario")
        .agg(samples=("correct", "size"), correct=("correct", "sum"))
        .reset_index()
    )
    summary["accuracy"] = summary["correct"] / summary["samples"]
    return summary


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

    sim_meta = pd.read_csv(SIM_DIR / "dataset_metadata.csv")
    real_meta = pd.read_csv(REAL_DIR / "dataset_metadata.csv")

    parts = split_sessions(sim_meta)

    train_images, train_labels = load_images(parts["train"], "train")
    val_images, val_labels = load_images(parts["validation"], "validation")
    test_images, test_labels = load_images(parts["test"], "test")
    real_images, real_labels = load_images(real_meta, "real")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, part in parts.items():
        part.to_csv(OUTPUT_DIR / f"split_{name}.csv", index=False)

    loader = DataLoader(
        TensorDataset(train_images, train_labels),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0,
    )

    model = SmallCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    checkpoint_path = OUTPUT_DIR / "best_model.pth"
    best_val_loss = float("inf")
    best_epoch = 0
    history = []

    print(f"PyTorch {torch.__version__}, CPU")
    for name, part in parts.items():
        print(
            f"{name}: {len(part)} sampel, {part['session_id'].nunique()} sesi "
            f"(normal {(part['label'] == 'normal').sum()}, "
            f"spoofed {(part['label'] == 'spoofed').sum()})"
        )
    print(f"GPS asli: {len(real_meta)} sampel")
    print()

    for epoch in range(1, EPOCHS + 1):
        model.train()

        for images, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        train_loss, train_acc, _, _ = evaluate(
            model, train_images, train_labels, criterion
        )
        val_loss, val_acc, _, _ = evaluate(
            model, val_images, val_labels, criterion
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "train_accuracy": train_acc,
            "validation_accuracy": val_acc,
        })

        # Pilih model berdasarkan loss validation terendah.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "architecture": "SmallCNN_1_4_8_v1",
                "encoding": ENCODING,
                "window_s": 30,
                "input_normalization": "uint8 divided by 255",
                "class_names": ["normal", "spoofed"],
                "split_seed": SPLIT_SEED,
                "seed": SEED,
                "epoch": epoch,
                "validation_loss": val_loss,
            }, checkpoint_path)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d}/{EPOCHS} | "
                f"train loss={train_loss:.4f}, acc={train_acc:.1%} | "
                f"val loss={val_loss:.4f}, acc={val_acc:.1%}"
            )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Uji 1: test simulasi.
    _, test_acc, test_pred, test_prob = evaluate(
        model, test_images, test_labels, criterion
    )
    test_results = parts["test"].copy()
    test_results["predicted_label_id"] = test_pred
    test_results["prob_spoofed"] = test_prob
    test_results["correct"] = test_pred == test_labels.numpy()
    test_results.to_csv(OUTPUT_DIR / "sim_test_predictions.csv", index=False)

    test_scenarios = per_scenario(test_results)
    test_scenarios.to_csv(OUTPUT_DIR / "sim_test_by_scenario.csv", index=False)

    test_report = classification_report(
        test_labels.numpy(), test_pred, labels=[0, 1],
        target_names=["normal", "spoofed"], zero_division=0, digits=3,
    )
    test_matrix = confusion_matrix(test_labels.numpy(), test_pred, labels=[0, 1])

    # Uji 2: GPS asli (semua normal).
    _, real_acc, real_pred, real_prob = evaluate(
        model, real_images, real_labels, criterion
    )
    real_results = real_meta.copy()
    real_results["predicted_label_id"] = real_pred
    real_results["prob_spoofed"] = real_prob
    real_results["correct"] = real_pred == real_labels.numpy()
    real_results.to_csv(OUTPUT_DIR / "real_predictions.csv", index=False)

    real_scenarios = per_scenario(real_results)
    real_scenarios.to_csv(OUTPUT_DIR / "real_by_scenario.csv", index=False)

    real_sessions = (
        real_results.groupby("session_id")
        .agg(samples=("correct", "size"), correct=("correct", "sum"))
        .reset_index()
    )
    real_sessions.to_csv(OUTPUT_DIR / "real_by_session.csv", index=False)

    false_alarms = real_results.loc[~real_results["correct"]]
    false_alarms[["sample_id", "scenario", "nonzero_pixels", "prob_spoofed"]].to_csv(
        OUTPUT_DIR / "real_false_alarms.csv", index=False
    )

    pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)

    lines = [
        f"PyTorch: {torch.__version__}",
        f"Best epoch: {best_epoch} (val loss {best_val_loss:.4f})",
        "",
        f"== Test simulasi: {test_acc:.1%} ({int(test_results['correct'].sum())}/{len(test_results)}) ==",
        test_scenarios.to_string(index=False),
        "",
        "Confusion matrix (baris=asli normal/spoofed, kolom=prediksi):",
        str(test_matrix),
        "",
        test_report,
        f"== GPS asli: {int(real_results['correct'].sum())}/{len(real_results)} "
        f"dikenali normal ({real_acc:.1%}), "
        f"{len(false_alarms)} salah dituduh spoofed ==",
        real_scenarios.to_string(index=False),
        "",
        real_sessions.to_string(index=False),
    ]
    summary = "\n".join(lines)
    (OUTPUT_DIR / "summary.txt").write_text(summary, encoding="utf-8")

    history_frame = pd.DataFrame(history)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history_frame["epoch"], history_frame["train_loss"], label="Train")
    axes[0].plot(history_frame["epoch"], history_frame["validation_loss"], label="Validation")
    axes[0].axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history_frame["epoch"], history_frame["train_accuracy"], label="Train")
    axes[1].plot(history_frame["epoch"], history_frame["validation_accuracy"], label="Validation")
    axes[1].axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
    plt.close(figure)

    print()
    print(summary)
    print()
    print(f"Folder hasil: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()