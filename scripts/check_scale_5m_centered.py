import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from gps_encoder import gps_to_local_meters


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset_sessions_v2"
REPORT_DIR = PROJECT_DIR / "reports" / "tables" / "scale_5m_centered"

IMAGE_SIZE = 32
METERS_PER_PIXEL = 5.0

SESSION_037_IDS = {
    "v2_sim_session_037_normal",
    "v2_sim_session_037_frozen",
}


def inspect_sample(data):
    times = data["time_s"].to_numpy(dtype=float)

    if (
        len(times) < 2
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("Timestamp harus valid dan berurutan naik.")

    east, north = gps_to_local_meters(data)

    # Pusatkan lintasan berdasarkan batas koordinat seluruh sampel.
    center_east = (east.min() + east.max()) / 2
    center_north = (north.min() + north.max()) / 2
    grid_midpoint = IMAGE_SIZE / 2

    columns = np.floor(
        grid_midpoint + (east - center_east) / METERS_PER_PIXEL
    )
    rows = np.floor(
        grid_midpoint - (north - center_north) / METERS_PER_PIXEL
    )

    inside = (
        (columns >= 0)
        & (columns < IMAGE_SIZE)
        & (rows >= 0)
        & (rows < IMAGE_SIZE)
    )
    outside_count = int(np.count_nonzero(~inside))

    if outside_count:
        return outside_count, None

    # Intensitas 0 untuk latar, 1–255 untuk waktu kunjungan.
    relative_time = (times - times[0]) / (times[-1] - times[0])
    intensities = np.rint(1 + 254 * relative_time).astype(np.uint8)

    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

    # Jika satu piksel dikunjungi berulang, simpan waktu terakhir.
    np.maximum.at(
        image,
        (rows.astype(int), columns.astype(int)),
        intensities,
    )

    return 0, image


def main():
    metadata = pd.read_csv(DATASET_DIR / "dataset_metadata.csv")
    required_columns = {
        "sample_id", "session_id", "label", "scenario", "raw_path"
    }
    missing = required_columns - set(metadata.columns)

    if missing:
        raise ValueError(f"Kolom metadata belum lengkap: {sorted(missing)}")

    if metadata.empty or metadata[list(required_columns)].isna().any().any():
        raise ValueError("Metadata kosong atau ada kolom wajib yang kosong.")

    if not metadata["sample_id"].is_unique:
        raise ValueError("Ada sample_id duplikat dalam metadata.")

    results = []
    hashes = []

    for record in metadata.itertuples(index=False):
        data = pd.read_csv(PROJECT_DIR / record.raw_path)
        outside_count, image = inspect_sample(data)

        results.append({
            "sample_id": record.sample_id,
            "session_id": record.session_id,
            "label": record.label,
            "scenario": record.scenario,
            "outside_points": outside_count,
            "fits_image": outside_count == 0,
        })

        if image is not None:
            hashes.append({
                "sample_id": record.sample_id,
                "session_id": record.session_id,
                "label": record.label,
                "image_hash": hashlib.sha256(image.tobytes()).hexdigest(),
            })

    results_frame = pd.DataFrame(results)
    hash_frame = pd.DataFrame(
        hashes,
        columns=["sample_id", "session_id", "label", "image_hash"],
    )

    duplicate_groups = []

    for image_hash, group in hash_frame.groupby("image_hash"):
        if len(group) < 2:
            continue

        duplicate_groups.append({
            "image_hash": image_hash,
            "sample_ids": " | ".join(group["sample_id"]),
            "label_conflict": group["label"].nunique() > 1,
            "multiple_sessions": group["session_id"].nunique() > 1,
        })

    duplicate_frame = pd.DataFrame(
        duplicate_groups,
        columns=[
            "image_hash", "sample_ids", "label_conflict", "multiple_sessions"
        ],
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results_frame.to_csv(REPORT_DIR / "boundary_checks.csv", index=False)
    duplicate_frame.to_csv(REPORT_DIR / "duplicate_images.csv", index=False)

    fits_count = int(results_frame["fits_image"].sum())
    conflicts = sum(group["label_conflict"] for group in duplicate_groups)

    print("Pemeriksaan skala 5 meter/piksel dengan pemusatan lintasan selesai.")
    print(f"Total sampel: {len(results_frame)}")
    print(f"Masuk gambar: {fits_count}")
    print(f"Keluar batas: {len(results_frame) - fits_count}")
    print(f"Kelompok gambar identik: {len(duplicate_groups)}")
    print(f"Konflik label pada sampel yang masuk gambar: {conflicts}")

    session_037 = hash_frame.loc[
        hash_frame["sample_id"].isin(SESSION_037_IDS)
    ]

    print()
    if len(session_037) == 2:
        identical = session_037["image_hash"].nunique() == 1
        print(f"Normal dan frozen sesi 037 masih identik: {identical}")
    else:
        print("Pasangan sesi 037 tidak lengkap atau keluar batas.")

    outside_by_scenario = (
        results_frame.assign(outside=~results_frame["fits_image"])
        .groupby("scenario")["outside"]
        .sum()
    )

    print()
    print("Sampel keluar batas per kondisi:")
    print(outside_by_scenario.to_string())
    print()
    print(f"Folder laporan: {REPORT_DIR}")


if __name__ == "__main__":
    main()