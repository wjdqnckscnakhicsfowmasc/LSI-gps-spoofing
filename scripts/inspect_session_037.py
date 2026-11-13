from pathlib import Path

import numpy as np
import pandas as pd

from gps_encoder import (
    METERS_PER_PIXEL,
    ORIGIN_COLUMN,
    ORIGIN_ROW,
    gps_to_local_meters,
)
from create_time_encoding import encode_time


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset_sessions_v2"
REPORT_DIR = PROJECT_DIR / "reports" / "tables" / "inspect_037"

NORMAL_ID = "v2_sim_session_037_normal"
FROZEN_ID = "v2_sim_session_037_frozen"


def map_pixels(data):
    east, north = gps_to_local_meters(data)

    columns = ORIGIN_COLUMN + np.floor(
        east / METERS_PER_PIXEL + 0.5
    ).astype(int)

    rows = ORIGIN_ROW - np.floor(
        north / METERS_PER_PIXEL + 0.5
    ).astype(int)

    return east, north, rows, columns


def main():
    metadata = pd.read_csv(
        DATASET_DIR / "dataset_metadata.csv"
    ).set_index("sample_id")

    normal_record = metadata.loc[NORMAL_ID]
    frozen_record = metadata.loc[FROZEN_ID]

    normal = pd.read_csv(PROJECT_DIR / normal_record["raw_path"])
    frozen = pd.read_csv(PROJECT_DIR / frozen_record["raw_path"])

    normal_times = normal["time_s"].to_numpy(dtype=float)
    frozen_times = frozen["time_s"].to_numpy(dtype=float)

    if not np.array_equal(normal_times, frozen_times):
        raise ValueError("Timestamp kedua sampel berbeda.")

    normal_image = encode_time(normal)
    frozen_image = encode_time(frozen)

    normal_east, normal_north, normal_rows, normal_columns = (
        map_pixels(normal)
    )
    frozen_east, frozen_north, frozen_rows, frozen_columns = (
        map_pixels(frozen)
    )

    # Kedua sampel harus menggunakan titik referensi GPS yang sama.
    if not np.array_equal(
        normal.loc[0, ["latitude", "longitude"]].to_numpy(),
        frozen.loc[0, ["latitude", "longitude"]].to_numpy(),
    ):
        raise ValueError("Titik referensi GPS berbeda.")

    coordinates_changed = np.any(
        normal[["latitude", "longitude"]].to_numpy()
        != frozen[["latitude", "longitude"]].to_numpy(),
        axis=1,
    )

    pixels_changed = (
        (normal_rows != frozen_rows)
        | (normal_columns != frozen_columns)
    )

    comparison = pd.DataFrame({
        "time_s": normal_times,
        "normal_east_m": normal_east,
        "normal_north_m": normal_north,
        "frozen_east_m": frozen_east,
        "frozen_north_m": frozen_north,
        "normal_row": normal_rows,
        "normal_column": normal_columns,
        "frozen_row": frozen_rows,
        "frozen_column": frozen_columns,
        "coordinates_changed": coordinates_changed,
        "pixels_changed": pixels_changed,
    })

    # Bandingkan seluruh piksel yang pernah dikunjungi.
    visited_pixels = sorted(
        set(zip(normal_rows.tolist(), normal_columns.tolist()))
        | set(zip(frozen_rows.tolist(), frozen_columns.tolist()))
    )

    pixel_records = []

    for row, column in visited_pixels:
        normal_visits = normal_times[
            (normal_rows == row) & (normal_columns == column)
        ]
        frozen_visits = frozen_times[
            (frozen_rows == row) & (frozen_columns == column)
        ]

        pixel_records.append({
            "row": row,
            "column": column,
            "normal_count": len(normal_visits),
            "frozen_count": len(frozen_visits),
            "normal_last_time_s": (
                normal_visits[-1] if len(normal_visits) else np.nan
            ),
            "frozen_last_time_s": (
                frozen_visits[-1] if len(frozen_visits) else np.nan
            ),
            "normal_intensity": int(normal_image[row, column]),
            "frozen_intensity": int(frozen_image[row, column]),
        })

    pixel_report = pd.DataFrame(pixel_records)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(
        REPORT_DIR / "point_comparison.csv",
        index=False,
    )
    pixel_report.to_csv(
        REPORT_DIR / "pixel_comparison.csv",
        index=False,
    )

    print("Pemeriksaan sesi 037 selesai.")
    print(
        "Interval freeze:",
        frozen_record["freeze_start_s"],
        "hingga sebelum",
        frozen_record["freeze_end_s"],
        "detik",
    )
    print(f"Titik dengan koordinat berbeda: {coordinates_changed.sum()}")
    print(f"Titik dengan piksel berbeda: {pixels_changed.sum()}")
    print(
        "Piksel gambar akhir berbeda:",
        np.count_nonzero(normal_image != frozen_image),
    )
    print()
    print("Perbandingan per piksel:")
    print(pixel_report.to_string(index=False))
    print()
    print(f"Folder laporan: {REPORT_DIR}")


if __name__ == "__main__":
    main()