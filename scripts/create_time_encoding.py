import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from gps_encoder import (
    IMAGE_SIZE,
    METERS_PER_PIXEL,
    ORIGIN_COLUMN,
    ORIGIN_ROW,
    gps_to_local_meters,
)


PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_DIR / "dataset_sessions_v2"
OUTPUT_DIR = PROJECT_DIR / "dataset_time_v1"

ENCODING_ID = "last_visit_time_10m_center16_v1"


def encode_time(data):
    times = data["time_s"].to_numpy(dtype=float)

    if (
        len(times) < 2
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("Timestamp tidak valid.")

    east, north = gps_to_local_meters(data)

    column_positions = ORIGIN_COLUMN + np.floor(
        east / METERS_PER_PIXEL + 0.5
    )
    row_positions = ORIGIN_ROW - np.floor(
        north / METERS_PER_PIXEL + 0.5
    )

    inside = (
        (column_positions >= 0)
        & (column_positions < IMAGE_SIZE)
        & (row_positions >= 0)
        & (row_positions < IMAGE_SIZE)
    )

    if not inside.all():
        raise ValueError("Ada titik keluar gambar.")

    columns = column_positions.astype(int)
    rows = row_positions.astype(int)

    # Nilai 0 untuk latar belakang.
    # Waktu pengamatan dipetakan ke intensitas 1–255.
    intensities = np.rint(
        1 + 254 * (times - times[0]) / (times[-1] - times[0])
    ).astype(np.uint8)

    image = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.uint8,
    )

    # Waktu terus meningkat, sehingga nilai terbesar
    # mewakili kunjungan terakhir pada piksel tersebut.
    np.maximum.at(image, (rows, columns), intensities)

    return image


def main():
    metadata = pd.read_csv(SOURCE_DIR / "dataset_metadata.csv")

    if metadata.empty or not metadata["sample_id"].is_unique:
        raise ValueError("Metadata kosong atau sample_id duplikat.")

    generated = []
    records = []
    hash_records = []

    # Validasi seluruh sampel sebelum menyimpan gambar.
    for _, record in metadata.iterrows():
        data = pd.read_csv(PROJECT_DIR / record["raw_path"])
        image = encode_time(data)

        sample_id = record["sample_id"]
        label = record["label"]

        if label not in {"normal", "spoofed"}:
            raise ValueError(f"Label tidak dikenal: {label}")

        npy_path = OUTPUT_DIR / label / f"{sample_id}.npy"
        png_path = OUTPUT_DIR / label / f"{sample_id}.png"

        new_record = record.to_dict()
        new_record.update({
            "encoding": ENCODING_ID,
            "pixel_value_rule": "last_visit_time_1_to_255",
            "time_reference": "relative_to_sample_duration",
            "nonzero_pixels": int(np.count_nonzero(image)),
            "npy_path": npy_path.relative_to(PROJECT_DIR).as_posix(),
            "png_path": png_path.relative_to(PROJECT_DIR).as_posix(),
        })

        # Batas kepadatan tidak digunakan pada encoding waktu.
        new_record.pop("points_for_white", None)

        records.append(new_record)
        generated.append((image, npy_path, png_path))

        hash_records.append({
            "sample_id": sample_id,
            "session_id": record["session_id"],
            "label": label,
            "image_hash": hashlib.sha256(
                image.tobytes()
            ).hexdigest(),
        })

    for image, npy_path, png_path in generated:
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, image)
        Image.fromarray(image).save(png_path)

    output_metadata = pd.DataFrame(records)
    output_metadata.to_csv(
        OUTPUT_DIR / "dataset_metadata.csv",
        index=False,
    )

    hashes = pd.DataFrame(hash_records)
    duplicate_groups = []

    for image_hash, group in hashes.groupby("image_hash"):
        if len(group) < 2:
            continue

        duplicate_groups.append({
            "image_hash": image_hash,
            "sample_ids": " | ".join(group["sample_id"]),
            "label_conflict": group["label"].nunique() > 1,
            "multiple_sessions": group["session_id"].nunique() > 1,
        })

    duplicates = pd.DataFrame(
        duplicate_groups,
        columns=[
            "image_hash",
            "sample_ids",
            "label_conflict",
            "multiple_sessions",
        ],
    )
    duplicates.to_csv(
        OUTPUT_DIR / "duplicate_images.csv",
        index=False,
    )

    conflicts = sum(
        group["label_conflict"] for group in duplicate_groups
    )

    print("Encoding waktu selesai.")
    print(f"Jumlah sampel: {len(output_metadata)}")
    print(f"Jumlah sesi: {output_metadata['session_id'].nunique()}")
    print(f"Kelompok gambar identik: {len(duplicate_groups)}")
    print(f"Kelompok identik dengan label berbeda: {conflicts}")

    if duplicate_groups:
        print()
        print(duplicates[[
            "sample_ids", "label_conflict", "multiple_sessions"
        ]].to_string(index=False))

    print()
    print(f"Folder hasil: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()