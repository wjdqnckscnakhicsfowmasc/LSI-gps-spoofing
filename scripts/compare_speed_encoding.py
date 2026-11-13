import numpy as np
import pandas as pd
import torch

from compare_encodings import OUTPUT_DIR, PROJECT_DIR, train_one
from gps_encoder import gps_to_local_meters


DATASET_DIR = PROJECT_DIR / "dataset_time_5m_centered_v1"
ENCODING_ID = "max_step_speed_5m_bbox_center_v1"
EXPERIMENT_NAME = "speed_5m_centered"

IMAGE_SIZE = 32
METERS_PER_PIXEL = 5.0
SPEED_CAP_M_S = 60.0


def encode_speed(data):
    times = data["time_s"].to_numpy(dtype=float)
    intervals = np.diff(times)

    if (
        len(times) < 2
        or not np.isfinite(times).all()
        or np.any(intervals <= 0)
    ):
        raise ValueError("Timestamp harus valid dan berurutan naik.")

    east, north = gps_to_local_meters(data)
    center_east = (east.min() + east.max()) / 2
    center_north = (north.min() + north.max()) / 2
    grid_midpoint = IMAGE_SIZE / 2

    columns = np.floor(
        grid_midpoint + (east - center_east) / METERS_PER_PIXEL
    ).astype(int)
    rows = np.floor(
        grid_midpoint - (north - center_north) / METERS_PER_PIXEL
    ).astype(int)

    if np.any(
        (columns < 0) | (columns >= IMAGE_SIZE)
        | (rows < 0) | (rows >= IMAGE_SIZE)
    ):
        raise ValueError("Ada titik keluar dari gambar 32×32.")

    # Kecepatan dari titik sebelumnya dicatat di piksel tujuan.
    step_speed = np.hypot(np.diff(east), np.diff(north)) / intervals
    clipped_speed = np.minimum(step_speed, SPEED_CAP_M_S)

    # Skala log menjaga perbedaan kecepatan rendah tetap terlihat.
    normalized = np.log1p(clipped_speed) / np.log1p(SPEED_CAP_M_S)
    intensities = np.rint(1 + 254 * normalized).astype(np.uint8)

    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    image[rows[0], columns[0]] = 1
    np.maximum.at(image, (rows[1:], columns[1:]), intensities)
    return image


def load_split(name, expected_count):
    metadata = pd.read_csv(DATASET_DIR / "splits" / f"{name}.csv")
    metadata = metadata.sort_values("sample_id").reset_index(drop=True)

    if (
        len(metadata) != expected_count
        or not metadata["sample_id"].is_unique
        or not metadata["split"].eq(name).all()
    ):
        raise ValueError(f"Pembagian {name} tidak sesuai.")

    expected_labels = metadata["label"].map({"normal": 0, "spoofed": 1})
    if not expected_labels.eq(metadata["label_id"]).all():
        raise ValueError(f"Label pada {name} tidak sesuai.")

    images = []
    for row in metadata.itertuples(index=False):
        try:
            data = pd.read_csv(PROJECT_DIR / row.raw_path)
            images.append(encode_speed(data))
        except ValueError as error:
            raise ValueError(f"{row.sample_id}: {error}") from error

    metadata = metadata.assign(encoding=ENCODING_ID)
    inputs = torch.from_numpy(np.stack(images)[:, None]).float() / 255.0
    labels = torch.tensor(metadata["label_id"].tolist(), dtype=torch.long)
    return metadata, inputs, labels


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

    train = load_split("train", 300)
    validation = load_split("validation", 100)

    if set(train[0]["session_id"]) & set(validation[0]["session_id"]):
        raise ValueError("Sesi train dan validation bertumpang tindih.")

    metrics = train_one(EXPERIMENT_NAME, ENCODING_ID, train, validation)

    experiment_dir = OUTPUT_DIR / EXPERIMENT_NAME
    predictions = pd.read_csv(experiment_dir / "validation_predictions.csv")
    predictions["correct"] = predictions["label_id"].eq(
        predictions["predicted_label_id"]
    )

    summary = predictions.groupby("scenario")["correct"].agg(
        samples="size", correct="sum"
    )
    summary["errors"] = summary["samples"] - summary["correct"]

    summary.to_csv(experiment_dir / "validation_by_scenario.csv")
    pd.DataFrame([metrics]).to_csv(
        experiment_dir / "validation_summary.csv", index=False
    )

    print("\nHasil validation per skenario")
    print(summary.to_string())
    print(f"\nBalanced accuracy: {metrics['balanced_accuracy']:.2%}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Folder hasil: {experiment_dir}")


if __name__ == "__main__":
    main()