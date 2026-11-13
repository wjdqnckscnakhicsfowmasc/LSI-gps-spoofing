from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from encode_fixed_scale import encode_heatmap, METERS_PER_PIXEL


PROJECT_DIR = Path(__file__).resolve().parent.parent

FREEZE_START_S = 20.0
FREEZE_END_S = 40.0


def main():
    source = (
        PROJECT_DIR
        / "data_raw"
        / "gps_normal"
        / "synthetic_001.csv"
    )
    normal = pd.read_csv(source)
    frozen = normal.copy()

    time_seconds = normal["time_s"].to_numpy(dtype=float)

    if (
        len(time_seconds) < 2
        or not np.isfinite(time_seconds).all()
        or np.any(np.diff(time_seconds) <= 0)
    ):
        raise ValueError("Waktu harus valid dan berurutan naik.")

    if not (
        time_seconds[0]
        <= FREEZE_START_S
        < FREEZE_END_S
        <= time_seconds[-1]
    ):
        raise ValueError("Interval freeze harus berada dalam lintasan.")

    freeze_mask = (
        (time_seconds >= FREEZE_START_S)
        & (time_seconds < FREEZE_END_S)
    )

    freeze_indices = np.flatnonzero(freeze_mask)

    if len(freeze_indices) < 2:
        raise ValueError("Interval freeze harus mencakup minimal dua titik.")

    anchor_index = freeze_indices[0]
    coordinate_columns = ["latitude", "longitude"]

    # Selama interval freeze, koordinat tetap pada titik awal interval.
    frozen.loc[freeze_mask, coordinate_columns] = (
        normal.loc[anchor_index, coordinate_columns].to_numpy()
    )

    normal_image = encode_heatmap(normal)
    frozen_image = encode_heatmap(frozen)

    raw_dir = PROJECT_DIR / "data_raw" / "gps_spoofed"
    output_dir = PROJECT_DIR / "dataset_fixed_10m" / "spoofed"
    figure_dir = PROJECT_DIR / "reports" / "figures"

    for folder in (raw_dir, output_dir, figure_dir):
        folder.mkdir(parents=True, exist_ok=True)

    sample_name = "synthetic_frozen_001"

    frozen.to_csv(raw_dir / f"{sample_name}.csv", index=False)
    np.save(output_dir / f"{sample_name}.npy", frozen_image)
    Image.fromarray(frozen_image).save(
        output_dir / f"{sample_name}.png"
    )

    figure, axes = plt.subplots(1, 2, figsize=(9, 4))

    for axis, label, image in zip(
        axes,
        ("Normal sintetis", "Frozen sintetis"),
        (normal_image, frozen_image),
    ):
        axis.imshow(
            image,
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

    figure.tight_layout()
    preview_path = figure_dir / "frozen_comparison_10m.png"
    figure.savefig(preview_path, dpi=150)
    plt.close(figure)

    print("Contoh frozen position berhasil dibuat.")
    print(
        f"Interval freeze: {FREEZE_START_S:g} "
        f"hingga sebelum {FREEZE_END_S:g} detik"
    )
    print(f"Jumlah titik dalam interval: {len(freeze_indices)}")
    print(f"Ukuran matriks: {frozen_image.shape}")
    print(f"Tipe data: {frozen_image.dtype}")
    print(f"Piksel terisi: {np.count_nonzero(frozen_image)}")
    print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main()