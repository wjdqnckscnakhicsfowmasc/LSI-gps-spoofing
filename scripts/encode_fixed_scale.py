from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent.parent

IMAGE_SIZE = 32
METERS_PER_PIXEL = 10.0
ORIGIN_COLUMN = 8
ORIGIN_ROW = 16
POINTS_FOR_WHITE = 8
EARTH_RADIUS_M = 6_371_000

OUTPUT_DIR = PROJECT_DIR / "dataset_fixed_10m"
PREVIEW_DIR = PROJECT_DIR / "reports" / "figures"

SAMPLES = [
    ("normal", "synthetic_001"),
    ("spoofed", "synthetic_jump_001"),
]


def encode_heatmap(data):
    coordinates = data[["latitude", "longitude"]].to_numpy(
        dtype=np.float64
    )

    if len(coordinates) == 0:
        raise ValueError("Data GPS kosong.")

    if not np.isfinite(coordinates).all():
        raise ValueError("Data GPS mengandung koordinat tidak valid.")

    latitude = np.radians(coordinates[:, 0])
    longitude = np.radians(coordinates[:, 1])

    east_m = (
        EARTH_RADIUS_M
        * (longitude - longitude[0])
        * np.cos(latitude[0])
    )
    north_m = EARTH_RADIUS_M * (latitude - latitude[0])

    # Setiap perpindahan 10 meter dipetakan ke satu piksel.
    columns = ORIGIN_COLUMN + np.floor(
        east_m / METERS_PER_PIXEL + 0.5
    ).astype(int)

    # Arah utara menuju baris gambar yang lebih kecil.
    rows = ORIGIN_ROW - np.floor(
        north_m / METERS_PER_PIXEL + 0.5
    ).astype(int)

    inside = (
        (columns >= 0)
        & (columns < IMAGE_SIZE)
        & (rows >= 0)
        & (rows < IMAGE_SIZE)
    )

    if not inside.all():
        outside_count = int(np.count_nonzero(~inside))
        raise ValueError(
            f"{outside_count} titik keluar gambar. "
            "Skala atau panjang window perlu dievaluasi."
        )

    counts = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.uint32,
    )
    np.add.at(counts, (rows, columns), 1)

    # Aturan kecerahan sama untuk semua sampel.
    heatmap = np.rint(
        np.minimum(counts, POINTS_FOR_WHITE)
        / POINTS_FOR_WHITE
        * 255
    ).astype(np.uint8)

    return heatmap


def main():
    encoded_samples = []

    # Validasi seluruh input sebelum menyimpan gambar.
    for label, sample_name in SAMPLES:
        source = (
            PROJECT_DIR
            / "data_raw"
            / f"gps_{label}"
            / f"{sample_name}.csv"
        )

        data = pd.read_csv(source)
        heatmap = encode_heatmap(data)
        encoded_samples.append((label, sample_name, heatmap))

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))

    for axis, (label, sample_name, heatmap) in zip(
        axes, encoded_samples
    ):
        sample_dir = OUTPUT_DIR / label
        sample_dir.mkdir(parents=True, exist_ok=True)

        np.save(sample_dir / f"{sample_name}.npy", heatmap)
        Image.fromarray(heatmap).save(
            sample_dir / f"{sample_name}.png"
        )

        axis.imshow(
            heatmap,
            cmap="gray",
            vmin=0,
            vmax=255,
            interpolation="nearest",
        )
        axis.set_title(
            f"{label} — {METERS_PER_PIXEL:g} meter/piksel"
        )
        axis.set_xlabel("Kolom piksel")
        axis.set_ylabel("Baris piksel")

        print(
            f"{label}: ukuran={heatmap.shape}, "
            f"tipe={heatmap.dtype}, "
            f"piksel terisi={np.count_nonzero(heatmap)}"
        )

    figure.tight_layout()
    preview_path = PREVIEW_DIR / "fixed_scale_10m_comparison.png"
    figure.savefig(preview_path, dpi=150)
    plt.close(figure)

    print("Encoding skala tetap selesai.")
    print(f"Skala: {METERS_PER_PIXEL:g} meter/piksel")
    print(f"Titik awal: kolom {ORIGIN_COLUMN}, baris {ORIGIN_ROW}")
    print(f"Folder hasil: {OUTPUT_DIR}")
    print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main()