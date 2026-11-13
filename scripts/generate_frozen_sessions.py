from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from gps_encoder import ENCODING_ID, METERS_PER_PIXEL, encode_gps


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset_sessions_v2"
FROZEN_SEED = 2044


def main():
    source_metadata = pd.read_csv(
        DATASET_DIR / "normal_metadata.csv"
    ).sort_values("session_id")

    previous_metadata = pd.read_csv(
        DATASET_DIR / "normal_jump_drift_metadata.csv"
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
        raise ValueError("Sumber frozen harus berupa normal bergerak.")

    if not set(source_metadata["sample_id"]).issubset(
        set(previous_metadata["sample_id"])
    ):
        raise ValueError("Metadata gabungan sebelumnya tidak lengkap.")

    rng = np.random.default_rng(FROZEN_SEED)
    generated = []
    frozen_records = []

    # Siapkan dan validasi semua sampel sebelum menyimpan.
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

        freeze_start = int(rng.integers(15, 26))
        freeze_duration = int(rng.integers(10, 21))
        freeze_end = freeze_start + freeze_duration

        if not times[0] <= freeze_start < freeze_end < times[-1]:
            raise ValueError(
                f"Interval freeze tidak sesuai: {source_path.name}"
            )

        freeze_mask = (
            (times >= freeze_start)
            & (times < freeze_end)
        )
        freeze_indices = np.flatnonzero(freeze_mask)

        if len(freeze_indices) < 2:
            raise ValueError("Freeze harus mencakup minimal dua titik.")

        anchor_index = int(freeze_indices[0])
        coordinate_columns = ["latitude", "longitude"]

        frozen = normal.copy()

        # Timestamp tetap berjalan, koordinat ditahan.
        frozen.loc[freeze_mask, coordinate_columns] = (
            normal.iloc[anchor_index][coordinate_columns].to_numpy(
                dtype=float
            )
        )

        image = encode_gps(frozen)

        session_id = record["session_id"]
        sample_id = f"{session_id}_frozen"

        raw_path = (
            DATASET_DIR / "raw" / "spoofed" / f"{sample_id}.csv"
        )
        npy_path = DATASET_DIR / "spoofed" / f"{sample_id}.npy"
        png_path = DATASET_DIR / "spoofed" / f"{sample_id}.png"

        frozen_record = record.to_dict()
        frozen_record.update({
            "sample_id": sample_id,
            "parent_sample_id": record["sample_id"],
            "label": "spoofed",
            "label_id": 1,
            "scenario": "position_freeze",
            "attack_type": "frozen",
            "freeze_start_s": freeze_start,
            "freeze_end_s": freeze_end,
            "freeze_duration_s": freeze_duration,
            "freeze_num_points": len(freeze_indices),
            "augmentation_seed": FROZEN_SEED,
            "nonzero_pixels": int(np.count_nonzero(image)),
            "raw_path": raw_path.relative_to(PROJECT_DIR).as_posix(),
            "npy_path": npy_path.relative_to(PROJECT_DIR).as_posix(),
            "png_path": png_path.relative_to(PROJECT_DIR).as_posix(),
        })

        frozen_records.append(frozen_record)
        generated.append((
            sample_id,
            frozen,
            image,
            raw_path,
            npy_path,
            png_path,
            freeze_start,
            freeze_duration,
        ))

    frozen_metadata = pd.DataFrame(frozen_records)
    combined = pd.concat(
        [previous_metadata, frozen_metadata],
        ignore_index=True,
    )

    if not combined["sample_id"].is_unique:
        raise ValueError("sample_id gabungan duplikat.")

    # Simpan seluruh 100 sampel frozen.
    for sample in generated:
        (
            sample_id,
            frozen,
            image,
            raw_path,
            npy_path,
            png_path,
            freeze_start,
            freeze_duration,
        ) = sample

        raw_path.parent.mkdir(parents=True, exist_ok=True)
        npy_path.parent.mkdir(parents=True, exist_ok=True)

        frozen.to_csv(raw_path, index=False)
        np.save(npy_path, image)
        Image.fromarray(image).save(png_path)

    frozen_metadata.to_csv(
        DATASET_DIR / "frozen_metadata.csv",
        index=False,
    )

    combined_path = DATASET_DIR / "dataset_metadata.csv"
    combined.to_csv(combined_path, index=False)

    # Preview hanya menampilkan 10 contoh pertama.
    figure, axes = plt.subplots(2, 5, figsize=(15, 6))

    for axis, sample in zip(axes.flat, generated[:10]):
        (
            sample_id,
            frozen,
            image,
            raw_path,
            npy_path,
            png_path,
            freeze_start,
            freeze_duration,
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
            f"Freeze {freeze_duration} s, mulai {freeze_start} s",
            fontsize=9,
        )
        axis.set_xticks([0, 16, 31])
        axis.set_yticks([0, 16, 31])

    figure.suptitle(
        "10 contoh dari 100 sesi frozen sintetis"
        f" — {METERS_PER_PIXEL:g} meter/piksel"
    )
    figure.tight_layout()

    figure_dir = PROJECT_DIR / "reports" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    preview_path = figure_dir / "frozen_sessions_v2.png"

    figure.savefig(preview_path, dpi=150)
    plt.close(figure)

    print("Pembuatan sesi frozen v2 selesai.")
    print(f"Sampel frozen: {len(frozen_metadata)}")
    print(f"Total sampel: {len(combined)}")
    print(f"Normal: {(combined['label'] == 'normal').sum()}")
    print(f"Spoofed: {(combined['label'] == 'spoofed').sum()}")
    print(f"Jumlah sesi asal: {combined['session_id'].nunique()}")
    print()
    print("Ringkasan 10 sampel pertama:")
    print(frozen_metadata[[
        "sample_id",
        "freeze_start_s",
        "freeze_duration_s",
        "nonzero_pixels",
    ]].head(10).to_string(index=False))
    print()
    print(f"Metadata gabungan: {combined_path}")
    print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main()