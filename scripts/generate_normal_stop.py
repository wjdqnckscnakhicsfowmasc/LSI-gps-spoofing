from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from encode_fixed_scale import encode_heatmap, METERS_PER_PIXEL


PROJECT_DIR = Path(__file__).resolve().parent.parent

STOP_START_S = 20.0
STOP_DURATION_S = 20.0


def main():
    source = (
        PROJECT_DIR
        / "data_raw"
        / "gps_normal"
        / "synthetic_001.csv"
    )
    normal = pd.read_csv(source)
    stopped = normal.copy()

    times = normal["time_s"].to_numpy(dtype=float)
    coordinates = normal[["latitude", "longitude"]].to_numpy(
        dtype=float
    )

    if (
        len(times) < 2
        or not np.isfinite(times).all()
        or not np.isfinite(coordinates).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("Koordinat dan waktu harus valid serta berurutan.")

    stop_end_s = STOP_START_S + STOP_DURATION_S

    if not (
        STOP_DURATION_S > 0
        and times[0] <= STOP_START_S
        and stop_end_s < times[-1]
    ):
        raise ValueError("Interval berhenti harus berada dalam lintasan.")

    # Waktu perjalanan tertahan selama berhenti.
    # Setelah berhenti, perjalanan tertunda 20 detik.
    elapsed_stop = np.clip(
        times - STOP_START_S,
        0.0,
        STOP_DURATION_S,
    )
    route_times = times - elapsed_stop

    # Timestamp pengamatan tetap berjalan.
    # Posisi mengikuti waktu perjalanan yang sudah disesuaikan.
    stopped["latitude"] = np.interp(
        route_times, times, coordinates[:, 0]
    )
    stopped["longitude"] = np.interp(
        route_times, times, coordinates[:, 1]
    )

    normal_image = encode_heatmap(normal)
    stopped_image = encode_heatmap(stopped)

    raw_dir = PROJECT_DIR / "data_raw" / "gps_normal"
    output_dir = PROJECT_DIR / "dataset_fixed_10m" / "normal"
    figure_dir = PROJECT_DIR / "reports" / "figures"

    for folder in (raw_dir, output_dir, figure_dir):
        folder.mkdir(parents=True, exist_ok=True)

    sample_name = "synthetic_stop_001"

    stopped.to_csv(raw_dir / f"{sample_name}.csv", index=False)
    np.save(output_dir / f"{sample_name}.npy", stopped_image)
    Image.fromarray(stopped_image).save(
        output_dir / f"{sample_name}.png"
    )

    figure, axes = plt.subplots(1, 2, figsize=(9, 4))

    for axis, label, image in zip(
        axes,
        ("Normal bergerak", "Normal berhenti lalu lanjut"),
        (normal_image, stopped_image),
    ):
        axis.imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=255,
            interpolation="nearest",
        )
        axis.set_title(
            f"{label}\n{METERS_PER_PIXEL:g} meter/piksel"
        )
        axis.set_xlabel("Kolom piksel")
        axis.set_ylabel("Baris piksel")

    figure.tight_layout()
    preview_path = figure_dir / "normal_stop_comparison_10m.png"
    figure.savefig(preview_path, dpi=150)
    plt.close(figure)

    print("Contoh normal berhenti berhasil dibuat.")
    print(f"Mulai berhenti: detik {STOP_START_S:g}")
    print(f"Durasi berhenti: {STOP_DURATION_S:g} detik")
    print(f"Perjalanan dilanjutkan pada detik: {stop_end_s:g}")
    print(f"Ukuran matriks: {stopped_image.shape}")
    print(f"Tipe data: {stopped_image.dtype}")
    print(f"Piksel terisi: {np.count_nonzero(stopped_image)}")
    print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main()