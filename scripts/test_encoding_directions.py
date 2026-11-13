from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gps_encoder import (
    IMAGE_SIZE,
    METERS_PER_PIXEL,
    ORIGIN_COLUMN,
    ORIGIN_ROW,
    encode_local_positions,
)


PROJECT_DIR = Path(__file__).resolve().parent.parent


def main():
    distances = np.linspace(0, 100, 60)
    zeros = np.zeros_like(distances)

    # Koordinat piksel ditulis sebagai (baris, kolom).
    direction_cases = [
        ("Timur", distances, zeros, (16, 26)),
        ("Barat", -distances, zeros, (16, 6)),
        ("Utara", zeros, distances, (6, 16)),
        ("Selatan", zeros, -distances, (26, 16)),
    ]

    figure, axes = plt.subplots(2, 2, figsize=(8, 8))

    for axis, (name, east, north, expected_end) in zip(
        axes.flat, direction_cases
    ):
        image = encode_local_positions(east, north)

        if image.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raise AssertionError(f"Ukuran gambar {name} tidak sesuai.")

        if image.dtype != np.uint8:
            raise AssertionError(f"Tipe data gambar {name} tidak sesuai.")

        # Setiap garis 100 meter melewati 11 piksel,
        # termasuk piksel awal dan akhir.
        expected_mask = np.zeros(
            (IMAGE_SIZE, IMAGE_SIZE),
            dtype=bool,
        )

        end_row, end_column = expected_end

        expected_mask[
            min(ORIGIN_ROW, end_row):max(ORIGIN_ROW, end_row) + 1,
            min(ORIGIN_COLUMN, end_column):max(
                ORIGIN_COLUMN, end_column
            ) + 1,
        ] = True

        if not np.array_equal(image > 0, expected_mask):
            raise AssertionError(
                f"Posisi piksel lintasan {name} tidak sesuai."
            )

        print(
            f"{name}: LULUS | "
            f"awal=({ORIGIN_ROW}, {ORIGIN_COLUMN}), "
            f"akhir={expected_end}"
        )

        axis.imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=255,
            interpolation="nearest",
        )
        axis.set_title(f"{name} — 100 meter")
        axis.set_xlabel("Kolom piksel")
        axis.set_ylabel("Baris piksel")

    outside_cases = [
        ("Timur", [0, 200], [0, 0]),
        ("Barat", [0, -200], [0, 0]),
        ("Utara", [0, 0], [0, 200]),
        ("Selatan", [0, 0], [0, -200]),
    ]

    for name, east, north in outside_cases:
        try:
            encode_local_positions(east, north)
        except ValueError as error:
            if "titik keluar gambar" not in str(error):
                raise
            print(f"Batas {name}: LULUS | titik di luar ditolak")
        else:
            raise AssertionError(
                f"Titik di luar gambar arah {name} tidak terdeteksi."
            )

    figure.tight_layout()

    output_dir = PROJECT_DIR / "reports" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = output_dir / "encoding_directions_10m.png"

    figure.savefig(preview_path, dpi=150)
    plt.close(figure)

    print("Semua 8 pengujian berhasil.")
    print("Encoder yang diuji: gps_encoder.py")
    print(f"Skala: {METERS_PER_PIXEL:g} meter/piksel")
    print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main()