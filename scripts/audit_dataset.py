import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from gps_encoder import (
    ENCODING_ID,
    METERS_PER_PIXEL,
    ORIGIN_COLUMN,
    ORIGIN_ROW,
    POINTS_FOR_WHITE,
    encode_gps,
)


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset_sessions_v2"
REPORT_DIR = PROJECT_DIR / "reports" / "tables" / "audit_v2"

EXPECTED_LABELS = {"normal": 0, "spoofed": 1}

ENCODING_SETTINGS = {
    "meters_per_pixel": METERS_PER_PIXEL,
    "origin_column": ORIGIN_COLUMN,
    "origin_row": ORIGIN_ROW,
    "points_for_white": POINTS_FOR_WHITE,
}


def main():
    metadata = pd.read_csv(DATASET_DIR / "dataset_metadata.csv")

    required_columns = {
        "sample_id", "session_id", "parent_sample_id",
        "label", "label_id", "encoding",
        "num_points", "start_time_s", "end_time_s",
        "nonzero_pixels", "raw_path", "npy_path", "png_path",
        *ENCODING_SETTINGS,
    }

    missing = required_columns - set(metadata.columns)
    if missing:
        raise ValueError(f"Kolom metadata belum lengkap: {sorted(missing)}")

    if metadata.empty:
        raise ValueError("Metadata kosong.")

    if metadata[list(required_columns)].isna().any().any():
        raise ValueError("Ada nilai kosong pada kolom metadata wajib.")

    if not metadata["sample_id"].is_unique:
        raise ValueError("sample_id duplikat.")

    for column in ("raw_path", "npy_path", "png_path"):
        if not metadata[column].is_unique:
            raise ValueError(f"Ada path duplikat pada {column}.")

    indexed = metadata.set_index("sample_id")
    file_results = []
    image_records = []

    for _, record in metadata.iterrows():
        sample_id = record["sample_id"]

        try:
            label = record["label"]

            if (
                label not in EXPECTED_LABELS
                or record["label_id"] != EXPECTED_LABELS[label]
            ):
                raise ValueError("Label dan label_id tidak sesuai.")

            if record["encoding"] != ENCODING_ID:
                raise ValueError("Encoding berbeda dari encoder aktif.")

            for key, expected in ENCODING_SETTINGS.items():
                if float(record[key]) != expected:
                    raise ValueError(f"Pengaturan {key} tidak sesuai.")

            parent_id = record["parent_sample_id"]

            if parent_id not in indexed.index:
                raise ValueError("Sampel induk tidak ditemukan.")

            parent = indexed.loc[parent_id]

            if parent["session_id"] != record["session_id"]:
                raise ValueError("Sesi sampel berbeda dari sesi induknya.")

            if (
                parent["label"] != "normal"
                or parent["parent_sample_id"] != parent_id
            ):
                raise ValueError("Induk harus berupa sampel normal dasar.")

            paths = {
                key: PROJECT_DIR / record[key]
                for key in ("raw_path", "npy_path", "png_path")
            }

            for path in paths.values():
                if not path.is_file():
                    raise FileNotFoundError(f"File tidak ditemukan: {path}")

            raw = pd.read_csv(paths["raw_path"])
            times = raw["time_s"].to_numpy(dtype=float)

            if (
                len(times) < 2
                or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0)
            ):
                raise ValueError("Timestamp tidak valid.")

            if len(raw) != record["num_points"]:
                raise ValueError("Jumlah titik berbeda dari metadata.")

            if not (
                np.isclose(times[0], record["start_time_s"])
                and np.isclose(times[-1], record["end_time_s"])
            ):
                raise ValueError("Rentang waktu berbeda dari metadata.")

            image = np.load(paths["npy_path"], allow_pickle=False)

            if image.shape != (32, 32) or image.dtype != np.uint8:
                raise ValueError("NPY harus berukuran 32×32 dan uint8.")

            if np.count_nonzero(image) != record["nonzero_pixels"]:
                raise ValueError("Jumlah piksel terisi berbeda dari metadata.")

            with Image.open(paths["png_path"]) as png:
                if png.mode != "L":
                    raise ValueError("PNG harus grayscale mode L.")

                if not np.array_equal(np.asarray(png), image):
                    raise ValueError("PNG berbeda dari NPY.")

            regenerated = encode_gps(raw)

            if not np.array_equal(regenerated, image):
                raise ValueError("Encoding ulang CSV berbeda dari NPY.")

            image_records.append({
                "sample_id": sample_id,
                "session_id": record["session_id"],
                "label": label,
                "image_hash": hashlib.sha256(image.tobytes()).hexdigest(),
            })

            file_results.append({
                "sample_id": sample_id,
                "status": "PASS",
                "detail": "File dan metadata konsisten",
            })

        except (
            ValueError, KeyError, TypeError, OSError, EOFError
        ) as error:
            file_results.append({
                "sample_id": sample_id,
                "status": "FAIL",
                "detail": str(error),
            })

    duplicates = []

    if image_records:
        hashes = pd.DataFrame(image_records)

        for image_hash, group in hashes.groupby("image_hash"):
            if len(group) < 2:
                continue

            duplicates.append({
                "image_hash": image_hash,
                "num_samples": len(group),
                "sample_ids": " | ".join(group["sample_id"]),
                "session_ids": " | ".join(
                    sorted(group["session_id"].unique())
                ),
                "labels": " | ".join(sorted(group["label"].unique())),
                "label_conflict": group["label"].nunique() > 1,
                "multiple_sessions": group["session_id"].nunique() > 1,
            })

    results = pd.DataFrame(file_results)
    duplicate_report = pd.DataFrame(
        duplicates,
        columns=[
            "image_hash", "num_samples", "sample_ids",
            "session_ids", "labels", "label_conflict",
            "multiple_sessions",
        ],
    )

    session_report = pd.crosstab(
        metadata["session_id"],
        metadata["label"],
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(REPORT_DIR / "dataset_audit.csv", index=False)
    duplicate_report.to_csv(
        REPORT_DIR / "duplicate_images.csv", index=False
    )
    session_report.to_csv(REPORT_DIR / "session_summary.csv")

    failures = int((results["status"] == "FAIL").sum())
    conflicts = sum(item["label_conflict"] for item in duplicates)

    print("Audit dataset selesai.")
    print(f"Jumlah sampel: {len(metadata)}")
    print(f"Jumlah sesi: {metadata['session_id'].nunique()}")
    print(f"File lolos: {len(results) - failures}")
    print(f"File gagal: {failures}")
    print(f"Kelompok gambar identik: {len(duplicates)}")
    print(f"Kelompok identik dengan label berbeda: {conflicts}")
    print()
    print(session_report.to_string())

    if failures:
        print()
        print(results.loc[
            results["status"] == "FAIL"
        ].to_string(index=False))

    if duplicates:
        print()
        print(duplicate_report[[
            "sample_ids", "label_conflict", "multiple_sessions"
        ]].to_string(index=False))

    print()
    print(f"Folder laporan: {REPORT_DIR}")


if __name__ == "__main__":
    main()