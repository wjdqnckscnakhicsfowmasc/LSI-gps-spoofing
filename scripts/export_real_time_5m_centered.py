from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from check_scale_5m_centered import METERS_PER_PIXEL, inspect_sample


PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_NMEA_DIR = PROJECT_DIR / "data_raw" / "gps_normal"
OUTPUT_DIR = PROJECT_DIR / "dataset_real_time_5m_centered_30s_v1"
WINDOW_DIR = OUTPUT_DIR / "raw" / "normal"
IMAGE_DIR = OUTPUT_DIR / "normal"

ENCODING_ID = "last_visit_time_5m_bbox_center_v1"

# Jendela 30 detik sesuai dokumen LSI: 30 titik dengan interval satu detik.
NUM_POINTS = 30
MAX_WINDOW_SPAN_S = 35.0


def nmea_to_decimal(value, hemisphere):
    degree_digits = 2 if hemisphere in ("N", "S") else 3
    degrees = float(value[:degree_digits]) + float(value[degree_digits:]) / 60
    return -degrees if hemisphere in ("S", "W") else degrees


def read_valid_points(nmea_path):
    """Ambil titik RMC berstatus A beserta waktu UTC-nya."""
    rows = []

    with open(nmea_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            parts = line.strip().split("*")[0].split(",")

            if not parts[0].endswith("RMC") or len(parts) < 10:
                continue
            if parts[2] != "A" or not parts[3] or not parts[5]:
                continue

            try:
                moment = datetime.strptime(
                    parts[9] + parts[1].split(".")[0], "%d%m%y%H%M%S"
                )
                latitude = nmea_to_decimal(parts[3], parts[4])
                longitude = nmea_to_decimal(parts[5], parts[6])
            except ValueError:
                continue

            rows.append((moment, latitude, longitude))

    points = pd.DataFrame(rows, columns=["utc", "latitude", "longitude"])
    points = points.drop_duplicates("utc").sort_values("utc")
    points["time_s"] = (points["utc"] - points["utc"].iloc[0]).dt.total_seconds()
    return points.reset_index(drop=True)


def main():
    nmea_files = sorted(RAW_NMEA_DIR.glob("session_*.nmea"))

    if not nmea_files:
        raise ValueError(f"Tidak ada file NMEA di {RAW_NMEA_DIR}")

    WINDOW_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    summary = []

    for nmea_path in nmea_files:
        session_id = f"real_{nmea_path.stem}"
        scenario = "stationary" if "diam" in nmea_path.stem else "moving"
        points = read_valid_points(nmea_path)

        kept = gap_skipped = outside_skipped = 0

        # Jendela tidak saling tumpang tindih.
        for start in range(0, len(points) - NUM_POINTS + 1, NUM_POINTS):
            window = points.iloc[start:start + NUM_POINTS]
            span = window["time_s"].iloc[-1] - window["time_s"].iloc[0]

            if span > MAX_WINDOW_SPAN_S:
                gap_skipped += 1
                continue

            data = pd.DataFrame({
                "time_s": window["time_s"].to_numpy() - window["time_s"].iloc[0],
                "latitude": window["latitude"].to_numpy(),
                "longitude": window["longitude"].to_numpy(),
            })

            outside_count, image = inspect_sample(data)

            if image is None:
                outside_skipped += 1
                continue

            sample_id = f"{session_id}_w{start // NUM_POINTS:03d}"
            raw_path = WINDOW_DIR / f"{sample_id}.csv"
            npy_path = IMAGE_DIR / f"{sample_id}.npy"
            png_path = IMAGE_DIR / f"{sample_id}.png"

            data.to_csv(raw_path, index=False)
            np.save(npy_path, image, allow_pickle=False)
            Image.fromarray(image).save(png_path)

            records.append({
                "sample_id": sample_id,
                "session_id": session_id,
                "label": "normal",
                "label_id": 0,
                "source": "real",
                "scenario": scenario,
                "attack_type": "none",
                "start_utc": window["utc"].iloc[0].isoformat(),
                "start_time_s": float(window["time_s"].iloc[0]),
                "end_time_s": float(window["time_s"].iloc[-1]),
                "num_points": NUM_POINTS,
                "encoding": ENCODING_ID,
                "meters_per_pixel": METERS_PER_PIXEL,
                "centering": "trajectory_bounding_box",
                "pixel_value_rule": "last_visit_time_1_to_255",
                "time_reference": "relative_to_sample_duration",
                "nonzero_pixels": int(np.count_nonzero(image)),
                "raw_path": raw_path.relative_to(PROJECT_DIR).as_posix(),
                "npy_path": npy_path.relative_to(PROJECT_DIR).as_posix(),
                "png_path": png_path.relative_to(PROJECT_DIR).as_posix(),
            })
            kept += 1

        summary.append({
            "session_id": session_id,
            "scenario": scenario,
            "valid_points": len(points),
            "windows_kept": kept,
            "skipped_time_gap": gap_skipped,
            "skipped_outside_160m": outside_skipped,
        })

    metadata = pd.DataFrame(records)

    if metadata.empty or not metadata["sample_id"].is_unique:
        raise ValueError("Tidak ada sampel, atau sample_id duplikat.")

    metadata_path = OUTPUT_DIR / "dataset_metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    print("Ekspor data GPS asli ke encoding waktu 5 meter/piksel selesai.")
    print(pd.DataFrame(summary).to_string(index=False))
    print()
    print(f"Jumlah sampel: {len(metadata)}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()