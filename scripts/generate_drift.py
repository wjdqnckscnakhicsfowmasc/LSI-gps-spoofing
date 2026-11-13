from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from encode_fixed_scale import encode_heatmap, METERS_PER_PIXEL


PROJECT_DIR = Path(__file__).resolve().parent.parent
EARTH_RADIUS_M = 6_371_000

# Pilihan awal percobaan.
DRIFT_START_S = 30.0
FINAL_DRIFT_M = 60.0


def main():
    source = (
        PROJECT_DIR
        / "data_raw"
        / "gps_normal"
        / "synthetic_001.csv"
    )
    normal = pd.read_csv(source)
    drifted = normal.copy()

    time_seconds = normal["time_s"].to_numpy(dtype=float)

    if (
        len(time_seconds) < 2
        or not np.isfinite(time_seconds).all()
        or np.any(np.diff(time_seconds) <= 0)
    ):
        raise ValueError("Waktu harus valid dan berurutan naik.")

    if not time_seconds[0] <= DRIFT_START_S < time_seconds[-1]:
        raise ValueError("Waktu mulai drift harus berada dalam lintasan.")

    # Offset nol sampai detik ke-30, kemudian naik hingga 60 meter.
    progress = np.clip(
        (time_seconds - DRIFT_START_S)
        / (time_seconds[-1] - DRIFT_START_S),
        0.0,
        1.0,
    )
    east_offset_m = FINAL_DRIFT_M * progress

    latitude_origin = np.radians(normal["latitude"].iloc[0])
    longitude_offset_deg = np.degrees(
        east_offset_m
        / (EARTH_RADIUS_M * np.cos(latitude_origin))
    )

    drifted["longitude"] = (
        normal["longitude"].to_numpy() + longitude_offset_deg
    )

    normal_image = encode_heatmap(normal)
    drift_image = encode_heatmap(drifted)

    raw_dir = PROJECT_DIR / "data_raw" / "gps_spoofed"
    output_dir = PROJECT_DIR / "dataset_fixed_10m" / "spoofed"
    figure_dir = PROJECT_DIR / "reports" / "figures"

    for folder in (raw_dir, output_dir, figure_dir):
        folder.mkdir(parents=True, exist_ok=True)

    sample_name = "synthetic_drift_001"

    drifted.to_csv(raw_dir / f"{sample_name}.csv", index=False)
    np.save(output_dir / f"{sample_name}.npy", drift_image)
    Image.fromarray(drift_image).save(
        output_dir / f"{sample_name}.png"
    )

    figure, axes = plt.subplots(1, 2, figsize=(9, 4))

    for axis, label, image in zip(
        axes,
        ("Normal sintetis", "Drift sintetis"),
        (normal_image, drift_image),
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
    preview_path = figure_dir / "drift_comparison_10m.png"
    figure.savefig(preview_path, dpi=150)
    plt.close(figure)

    print("Contoh drift sintetis berhasil dibuat.")
    print(f"Drift dimulai pada detik: {DRIFT_START_S:g}")
    print(f"Offset akhir ke timur: {east_offset_m[-1]:g} meter")
    print(f"Ukuran matriks: {drift_image.shape}")
    print(f"Tipe data: {drift_image.dtype}")
    print(f"Piksel terisi: {np.count_nonzero(drift_image)}")
    print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main()