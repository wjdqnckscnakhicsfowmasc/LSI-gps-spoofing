from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from gps_encoder import (
    EARTH_RADIUS_M,
    ENCODING_ID,
    METERS_PER_PIXEL,
    ORIGIN_COLUMN,
    ORIGIN_ROW,
    POINTS_FOR_WHITE,
    encode_gps,
)


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "dataset_sessions_v2"
RAW_DIR = OUTPUT_DIR / "raw" / "normal"
IMAGE_DIR = OUTPUT_DIR / "normal"
FIGURE_DIR = PROJECT_DIR / "reports" / "figures"

NUM_SESSIONS = 100
NUM_POINTS = 60
RANDOM_SEED = 2040

LATITUDE_ORIGIN = -6.973
LONGITUDE_ORIGIN = 107.630


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    times = np.arange(NUM_POINTS, dtype=float)

    samples = []
    records = []

    # Buat dan validasi seluruh sesi sebelum menyimpan.
    for index in range(NUM_SESSIONS):
        session_id = f"v2_sim_session_{index + 1:03d}"
        sample_id = f"{session_id}_normal"

        heading_start = rng.uniform(0, 2 * np.pi)
        total_turn = rng.uniform(-1.2, 1.2)
        base_speed = rng.uniform(0.8, 1.6)

        progress = np.linspace(0, 1, NUM_POINTS - 1)
        heading = heading_start + total_turn * progress

        speed = base_speed * (
            1 + 0.15 * np.sin(2 * np.pi * progress)
        )

        # Interval antartitik adalah satu detik.
        east = np.concatenate((
            [0.0],
            np.cumsum(speed * np.cos(heading)),
        ))
        north = np.concatenate((
            [0.0],
            np.cumsum(speed * np.sin(heading)),
        ))

        east += rng.normal(0, 0.5, NUM_POINTS)
        north += rng.normal(0, 0.5, NUM_POINTS)

        east -= east[0]
        north -= north[0]

        latitude = LATITUDE_ORIGIN + np.degrees(
            north / EARTH_RADIUS_M
        )
        longitude = LONGITUDE_ORIGIN + np.degrees(
            east
            / (
                EARTH_RADIUS_M
                * np.cos(np.radians(LATITUDE_ORIGIN))
            )
        )

        data = pd.DataFrame({
            "time_s": times,
            "latitude": latitude,
            "longitude": longitude,
        })

        image = encode_gps(data)
        samples.append((sample_id, data, image))

        raw_path = RAW_DIR / f"{sample_id}.csv"
        npy_path = IMAGE_DIR / f"{sample_id}.npy"
        png_path = IMAGE_DIR / f"{sample_id}.png"

        records.append({
            "sample_id": sample_id,
            "session_id": session_id,
            "parent_sample_id": sample_id,
            "label": "normal",
            "label_id": 0,
            "source": "synthetic",
            "scenario": "moving",
            "attack_type": "none",
            "start_time_s": float(times[0]),
            "end_time_s": float(times[-1]),
            "num_points": NUM_POINTS,
            "base_speed_m_s": base_speed,
            "heading_start_deg": np.degrees(heading_start),
            "total_turn_deg": np.degrees(total_turn),
            "noise_std_m": 0.5,
            "random_seed": RANDOM_SEED,
            "encoding": ENCODING_ID,
            "meters_per_pixel": METERS_PER_PIXEL,
            "origin_column": ORIGIN_COLUMN,
            "origin_row": ORIGIN_ROW,
            "points_for_white": POINTS_FOR_WHITE,
            "nonzero_pixels": int(np.count_nonzero(image)),
            "raw_path": raw_path.relative_to(PROJECT_DIR).as_posix(),
            "npy_path": npy_path.relative_to(PROJECT_DIR).as_posix(),
            "png_path": png_path.relative_to(PROJECT_DIR).as_posix(),
        })

    metadata = pd.DataFrame(records)

    if not metadata["sample_id"].is_unique:
        raise ValueError("sample_id duplikat.")

    if metadata["session_id"].nunique() != NUM_SESSIONS:
        raise ValueError("Jumlah sesi tidak sesuai.")

    for folder in (RAW_DIR, IMAGE_DIR, FIGURE_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    # Simpan seluruh 100 sampel.
    for sample_id, data, image in samples:
        data.to_csv(RAW_DIR / f"{sample_id}.csv", index=False)
        np.save(IMAGE_DIR / f"{sample_id}.npy", image)
        Image.fromarray(image).save(
            IMAGE_DIR / f"{sample_id}.png"
        )

    metadata_path = OUTPUT_DIR / "normal_metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    # Preview hanya menampilkan 10 contoh pertama.
    figure, axes = plt.subplots(2, 5, figsize=(15, 6))

    for axis, (sample_id, data, image) in zip(
        axes.flat, samples[:10]
    ):
        axis.imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=255,
            interpolation="nearest",
        )
        axis.set_title(
            sample_id.replace("_normal", ""),
            fontsize=9,
        )
        axis.set_xticks([0, 16, 31])
        axis.set_yticks([0, 16, 31])

    figure.suptitle(
        f"10 contoh dari {NUM_SESSIONS} sesi normal sintetis"
        f" — {METERS_PER_PIXEL:g} meter/piksel"
    )
    figure.tight_layout()

    preview_path = FIGURE_DIR / "normal_sessions_v2.png"
    figure.savefig(preview_path, dpi=150)
    plt.close(figure)

    print("Pembuatan sesi normal v2 selesai.")
    print(f"Jumlah sampel: {len(metadata)}")
    print(f"Jumlah sesi: {metadata['session_id'].nunique()}")
    print(f"Encoder: {ENCODING_ID}")
    print(f"Seed: {RANDOM_SEED}")
    print()
    print("Ringkasan 10 sampel pertama:")
    print(metadata[[
        "session_id",
        "label",
        "nonzero_pixels",
    ]].head(10).to_string(index=False))
    print()
    print(f"Folder dataset: {OUTPUT_DIR}")
    print(f"Metadata: {metadata_path}")
    print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main()