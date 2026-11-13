from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from gps_encoder import (
    EARTH_RADIUS_M,
    ENCODING_ID,
    METERS_PER_PIXEL,
    encode_gps,
)


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset_sessions_v2"
DRIFT_SEED = 2043


def main():
    source_metadata = pd.read_csv(
        DATASET_DIR / "normal_metadata.csv"
    ).sort_values("session_id")

    previous_metadata = pd.read_csv(
        DATASET_DIR / "normal_jump_metadata.csv"
    )

    if (
        len(source_metadata) != 100
        or not source_metadata["session_id"].is_unique
    ):
        raise ValueError("Diperlukan 100 sesi normal dasar yang berbeda.")

    for metadata in (source_metadata, previous_metadata):
        if not metadata["sample_id"].is_unique:
            raise ValueError("Metadata mengandung sample_id duplikat.")

        if not metadata["encoding"].eq(ENCODING_ID).all():
            raise ValueError("Encoding berbeda dari encoder aktif.")

    if not (
        source_metadata["label"].eq("normal").all()
        and source_metadata["scenario"].eq("moving").all()
    ):
        raise ValueError("Sumber drift harus berupa normal bergerak.")

    if not set(source_metadata["sample_id"]).issubset(
        set(previous_metadata["sample_id"])
    ):
        raise ValueError("Metadata gabungan sebelumnya tidak lengkap.")

    rng = np.random.default_rng(DRIFT_SEED)
    generated = []
    drift_records = []
    checks = []

    # Siapkan dan validasi seluruh sampel sebelum menyimpan.
    for _, record in source_metadata.iterrows():
        source_path = PROJECT_DIR / record["raw_path"]
        normal = pd.read_csv(source_path)
        times = normal["time_s"].to_numpy(dtype=float)

        if (
            len(times) < 2
            or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError(f"Waktu tidak valid: {source_path.name}")

        encode_gps(normal)

        drift_start = int(rng.integers(15, 31))
        drift_distance = float(rng.uniform(30, 60))
        drift_angle = float(rng.uniform(0, 2 * np.pi))

        if not times[0] < drift_start < times[-1]:
            raise ValueError("Waktu mulai drift di luar interval data.")

        # Offset meningkat dari nol sampai nilai akhir.
        progress = np.clip(
            (times - drift_start) / (times[-1] - drift_start),
            0.0,
            1.0,
        )

        final_east = drift_distance * np.cos(drift_angle)
        final_north = drift_distance * np.sin(drift_angle)

        latitude_origin = np.radians(normal["latitude"].iloc[0])

        latitude_offset = np.degrees(
            final_north * progress / EARTH_RADIUS_M
        )
        longitude_offset = np.degrees(
            final_east * progress
            / (EARTH_RADIUS_M * np.cos(latitude_origin))
        )

        drifted = normal.copy()
        drifted["latitude"] = (
            normal["latitude"].to_numpy() + latitude_offset
        )
        drifted["longitude"] = (
            normal["longitude"].to_numpy() + longitude_offset
        )

        session_id = record["session_id"]
        sample_id = f"{session_id}_drift"

        try:
            image = encode_gps(drifted)
        except ValueError as error:
            if "titik keluar gambar" not in str(error):
                raise

            checks.append({
                "sample_id": sample_id,
                "status": "outside",
                "detail": str(error),
            })
            continue

        checks.append({
            "sample_id": sample_id,
            "status": "accepted",
            "detail": "Semua titik masuk gambar",
        })

        raw_path = (
            DATASET_DIR / "raw" / "spoofed" / f"{sample_id}.csv"
        )
        npy_path = DATASET_DIR / "spoofed" / f"{sample_id}.npy"
        png_path = DATASET_DIR / "spoofed" / f"{sample_id}.png"

        drift_record = record.to_dict()
        drift_record.update({
            "sample_id": sample_id,
            "parent_sample_id": record["sample_id"],
            "label": "spoofed",
            "label_id": 1,
            "scenario": "gradual_offset",
            "attack_type": "drift",
            "drift_start_s": drift_start,
            "drift_end_s": float(times[-1]),
            "drift_final_distance_m": drift_distance,
            "drift_angle_deg": np.degrees(drift_angle),
            "drift_final_east_m": final_east,
            "drift_final_north_m": final_north,
            "augmentation_seed": DRIFT_SEED,
            "nonzero_pixels": int(np.count_nonzero(image)),
            "raw_path": raw_path.relative_to(PROJECT_DIR).as_posix(),
            "npy_path": npy_path.relative_to(PROJECT_DIR).as_posix(),
            "png_path": png_path.relative_to(PROJECT_DIR).as_posix(),
        })

        drift_records.append(drift_record)
        generated.append((
            sample_id,
            drifted,
            image,
            raw_path,
            npy_path,
            png_path,
            drift_start,
            drift_distance,
        ))

    report_dir = PROJECT_DIR / "reports" / "tables"
    report_dir.mkdir(parents=True, exist_ok=True)
    checks_path = report_dir / "drift_generation_checks_v2.csv"
    pd.DataFrame(checks).to_csv(checks_path, index=False)

    if len(generated) != len(source_metadata):
        raise ValueError(
            "Ada sampel keluar gambar. "
            f"Periksa {checks_path} sebelum melanjutkan."
        )

    drift_metadata = pd.DataFrame(drift_records)
    combined = pd.concat(
        [previous_metadata, drift_metadata],
        ignore_index=True,
    )

    if not combined["sample_id"].is_unique:
        raise ValueError("sample_id gabungan duplikat.")

    # Simpan seluruh 100 sampel drift.
    for sample in generated:
        (
            sample_id,
            drifted,
            image,
            raw_path,
            npy_path,
            png_path,
            drift_start,
            drift_distance,
        ) = sample

        raw_path.parent.mkdir(parents=True, exist_ok=True)
        npy_path.parent.mkdir(parents=True, exist_ok=True)

        drifted.to_csv(raw_path, index=False)
        np.save(npy_path, image)
        Image.fromarray(image).save(png_path)

    drift_metadata.to_csv(
        DATASET_DIR / "drift_metadata.csv",
        index=False,
    )

    combined_path = DATASET_DIR / "normal_jump_drift_metadata.csv"
    combined.to_csv(combined_path, index=False)

    # Preview hanya menampilkan 10 contoh pertama.
    figure, axes = plt.subplots(2, 5, figsize=(15, 6))

    for axis, sample in zip(axes.flat, generated[:10]):
        (
            sample_id,
            drifted,
            image,
            raw_path,
            npy_path,
            png_path,
            drift_start,
            drift_distance,
        ) = sample

        axis.imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=255,
            interpolation="nearest",
        )
        axis.set_title(
            f"{sample_id}\n"
            f"Drift akhir {drift_distance:.1f} m, mulai {drift_start} s",
            fontsize=9,
        )
        axis.set_xticks([0, 16, 31])
        axis.set_yticks([0, 16, 31])

    figure.suptitle(
        "10 contoh dari 100 sesi drift sintetis"
        f" — {METERS_PER_PIXEL:g} meter/piksel"
    )
    figure.tight_layout()

    figure_dir = PROJECT_DIR / "reports" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    preview_path = figure_dir / "drift_sessions_v2.png"

    figure.savefig(preview_path, dpi=150)
    plt.close(figure)

    print("Pembuatan sesi drift v2 selesai.")
    print(f"Sampel drift: {len(drift_metadata)}")
    print(f"Total sampel: {len(combined)}")
    print(f"Normal: {(combined['label'] == 'normal').sum()}")
    print(f"Spoofed: {(combined['label'] == 'spoofed').sum()}")
    print(f"Jumlah sesi asal: {combined['session_id'].nunique()}")
    print()
    print("Ringkasan 10 sampel pertama:")
    print(drift_metadata[[
        "sample_id",
        "drift_start_s",
        "drift_final_distance_m",
        "nonzero_pixels",
    ]].head(10).round(2).to_string(index=False))
    print()
    print(f"Metadata gabungan: {combined_path}")
    print(f"Pemeriksaan batas: {checks_path}")
    print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main()