from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from check_scale_5m_centered import METERS_PER_PIXEL, inspect_sample
from gps_encoder import EARTH_RADIUS_M


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "dataset_sim_time_5m_centered_30s_v1"

ENCODING_ID = "last_visit_time_5m_bbox_center_v1"

NUM_SESSIONS = 100
NUM_POINTS = 30          # 30 titik, interval satu detik

# Seed sama dengan generator v2 agar hasil dapat diulang.
NORMAL_SEED = 2040
STOP_SEED = 2041
JUMP_SEED = 2042
DRIFT_SEED = 2043
FROZEN_SEED = 2044

LATITUDE_ORIGIN = -6.973
LONGITUDE_ORIGIN = 107.630


def offset_to_degrees(east_m, north_m):
    latitude = np.degrees(north_m / EARTH_RADIUS_M)
    longitude = np.degrees(
        east_m / (EARTH_RADIUS_M * np.cos(np.radians(LATITUDE_ORIGIN)))
    )
    return latitude, longitude


def make_normal(rng, times):
    heading_start = rng.uniform(0, 2 * np.pi)
    total_turn = rng.uniform(-1.2, 1.2)
    base_speed = rng.uniform(0.8, 1.6)

    progress = np.linspace(0, 1, NUM_POINTS - 1)
    heading = heading_start + total_turn * progress
    speed = base_speed * (1 + 0.15 * np.sin(2 * np.pi * progress))

    east = np.concatenate(([0.0], np.cumsum(speed * np.cos(heading))))
    north = np.concatenate(([0.0], np.cumsum(speed * np.sin(heading))))

    east += rng.normal(0, 0.5, NUM_POINTS)
    north += rng.normal(0, 0.5, NUM_POINTS)
    east -= east[0]
    north -= north[0]

    latitude_offset, longitude_offset = offset_to_degrees(east, north)

    data = pd.DataFrame({
        "time_s": times,
        "latitude": LATITUDE_ORIGIN + latitude_offset,
        "longitude": LONGITUDE_ORIGIN + longitude_offset,
    })
    details = {
        "base_speed_m_s": base_speed,
        "heading_start_deg": np.degrees(heading_start),
        "total_turn_deg": np.degrees(total_turn),
    }
    return data, details


def make_stop(rng, normal, times):
    # Berhenti 5-10 detik, mulai pada detik 7-12.
    start = int(rng.integers(7, 13))
    duration = int(rng.integers(5, 11))

    route_times = times - np.clip(times - start, 0, duration)

    data = normal.copy()
    data["latitude"] = np.interp(route_times, times, normal["latitude"])
    data["longitude"] = np.interp(route_times, times, normal["longitude"])

    return data, {"stop_start_s": start, "stop_duration_s": duration}


def make_jump(rng, normal, times):
    # Loncatan 40-60 meter, mulai pada detik 10-17.
    start = int(rng.integers(10, 18))
    distance = float(rng.uniform(40, 60))
    angle = float(rng.uniform(0, 2 * np.pi))

    latitude_offset, longitude_offset = offset_to_degrees(
        distance * np.cos(angle), distance * np.sin(angle)
    )

    data = normal.copy()
    mask = times >= start
    data.loc[mask, "latitude"] += latitude_offset
    data.loc[mask, "longitude"] += longitude_offset

    return data, {
        "jump_start_s": start,
        "jump_distance_m": distance,
        "jump_angle_deg": np.degrees(angle),
    }


def make_drift(rng, normal, times):
    # Pergeseran bertahap sampai 15-30 meter, mulai pada detik 7-15.
    # Jarak dibagi dua dari versi 60 detik agar laju drift tetap sama.
    start = int(rng.integers(7, 16))
    distance = float(rng.uniform(15, 30))
    angle = float(rng.uniform(0, 2 * np.pi))

    progress = np.clip((times - start) / (times[-1] - start), 0.0, 1.0)
    latitude_offset, longitude_offset = offset_to_degrees(
        distance * np.cos(angle) * progress,
        distance * np.sin(angle) * progress,
    )

    data = normal.copy()
    data["latitude"] = normal["latitude"].to_numpy() + latitude_offset
    data["longitude"] = normal["longitude"].to_numpy() + longitude_offset

    return data, {
        "drift_start_s": start,
        "drift_final_distance_m": distance,
        "drift_angle_deg": np.degrees(angle),
    }


def make_frozen(rng, normal, times):
    # Posisi ditahan 5-10 detik, mulai pada detik 7-12.
    start = int(rng.integers(7, 13))
    duration = int(rng.integers(5, 11))

    mask = (times >= start) & (times < start + duration)
    anchor = int(np.flatnonzero(mask)[0])
    columns = ["latitude", "longitude"]

    data = normal.copy()
    data.loc[mask, columns] = normal.iloc[anchor][columns].to_numpy(dtype=float)

    return data, {"freeze_start_s": start, "freeze_duration_s": duration}


SCENARIOS = [
    # akhiran, label, skenario, jenis serangan, pembuat, seed
    ("stop", "normal", "stop_and_resume", "none", make_stop, STOP_SEED),
    ("jump", "spoofed", "position_jump", "jump", make_jump, JUMP_SEED),
    ("drift", "spoofed", "gradual_offset", "drift", make_drift, DRIFT_SEED),
    ("frozen", "spoofed", "position_freeze", "frozen", make_frozen, FROZEN_SEED),
]


def main():
    times = np.arange(NUM_POINTS, dtype=float)
    normal_rng = np.random.default_rng(NORMAL_SEED)
    scenario_rngs = {
        suffix: np.random.default_rng(seed)
        for suffix, _, _, _, _, seed in SCENARIOS
    }

    samples = []

    # Buat dan validasi seluruh sampel sebelum menyimpan.
    for index in range(NUM_SESSIONS):
        session_id = f"sim30_session_{index + 1:03d}"
        normal, normal_details = make_normal(normal_rng, times)
        parent_id = f"{session_id}_normal"

        variants = [
            (parent_id, "normal", "moving", "none", normal, normal_details)
        ]

        for suffix, label, scenario, attack, maker, _ in SCENARIOS:
            data, details = maker(scenario_rngs[suffix], normal, times)
            variants.append((
                f"{session_id}_{suffix}", label, scenario, attack,
                data, {**normal_details, **details},
            ))

        for sample_id, label, scenario, attack, data, details in variants:
            outside_count, image = inspect_sample(data)

            if image is None:
                raise ValueError(
                    f"{sample_id}: {outside_count} titik keluar batas gambar."
                )

            samples.append({
                "sample_id": sample_id,
                "session_id": session_id,
                "parent_sample_id": parent_id,
                "label": label,
                "label_id": 0 if label == "normal" else 1,
                "source": "synthetic",
                "scenario": scenario,
                "attack_type": attack,
                "start_time_s": float(times[0]),
                "end_time_s": float(times[-1]),
                "num_points": NUM_POINTS,
                "noise_std_m": 0.5,
                **details,
                "encoding": ENCODING_ID,
                "meters_per_pixel": METERS_PER_PIXEL,
                "centering": "trajectory_bounding_box",
                "pixel_value_rule": "last_visit_time_1_to_255",
                "time_reference": "relative_to_sample_duration",
                "nonzero_pixels": int(np.count_nonzero(image)),
                "_data": data,
                "_image": image,
            })

    records = []
    identical_to_parent = []
    parent_images = {
        sample["sample_id"]: sample["_image"]
        for sample in samples if sample["scenario"] == "moving"
    }

    for sample in samples:
        data = sample.pop("_data")
        image = sample.pop("_image")
        sample_id = sample["sample_id"]
        label = sample["label"]

        raw_path = OUTPUT_DIR / "raw" / label / f"{sample_id}.csv"
        npy_path = OUTPUT_DIR / label / f"{sample_id}.npy"
        png_path = OUTPUT_DIR / label / f"{sample_id}.png"

        raw_path.parent.mkdir(parents=True, exist_ok=True)
        npy_path.parent.mkdir(parents=True, exist_ok=True)

        data.to_csv(raw_path, index=False)
        np.save(npy_path, image, allow_pickle=False)
        Image.fromarray(image).save(png_path)

        if label == "spoofed" and np.array_equal(
            image, parent_images[sample["parent_sample_id"]]
        ):
            identical_to_parent.append(sample_id)

        sample.update({
            "raw_path": raw_path.relative_to(PROJECT_DIR).as_posix(),
            "npy_path": npy_path.relative_to(PROJECT_DIR).as_posix(),
            "png_path": png_path.relative_to(PROJECT_DIR).as_posix(),
        })
        records.append(sample)

    metadata = pd.DataFrame(records)

    if (
        len(metadata) != NUM_SESSIONS * 5
        or not metadata["sample_id"].is_unique
        or metadata["session_id"].nunique() != NUM_SESSIONS
    ):
        raise ValueError("Jumlah sampel, ID sampel, atau jumlah sesi tidak sesuai.")

    metadata_path = OUTPUT_DIR / "dataset_metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    print("Pembuatan dataset simulasi 30 detik selesai.")
    print(f"Jumlah sampel: {len(metadata)}")
    print(f"Jumlah sesi: {metadata['session_id'].nunique()}")
    print(metadata["label"].value_counts().to_string())
    print()
    print("Piksel menyala per skenario (min / median / maks):")
    print(
        metadata.groupby("scenario")["nonzero_pixels"]
        .agg(["min", "median", "max"]).to_string()
    )
    print()
    print(
        "Gambar spoofed yang identik dengan versi normalnya: "
        f"{len(identical_to_parent)}"
    )
    for sample_id in identical_to_parent:
        print(f"  {sample_id}")
    print()
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()