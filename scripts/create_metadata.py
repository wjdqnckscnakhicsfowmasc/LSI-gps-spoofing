from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset_fixed_10m"

# Nama sampel, label, jenis anomali, dan kondisi.
SAMPLES = [
    ("synthetic_001", "normal", "none", "moving"),
    ("synthetic_stop_001", "normal", "none", "stop_and_resume"),
    ("synthetic_jump_001", "spoofed", "jump", "position_jump"),
    ("synthetic_drift_001", "spoofed", "drift", "gradual_offset"),
    ("synthetic_frozen_001", "spoofed", "frozen", "position_freeze"),
]


def main():
    records = []

    for sample_id, label, attack_type, scenario in SAMPLES:
        raw_path = (
            PROJECT_DIR
            / "data_raw"
            / f"gps_{label}"
            / f"{sample_id}.csv"
        )
        npy_path = DATASET_DIR / label / f"{sample_id}.npy"
        png_path = DATASET_DIR / label / f"{sample_id}.png"

        for path in (raw_path, npy_path, png_path):
            if not path.is_file():
                raise FileNotFoundError(f"File belum tersedia: {path}")

        raw = pd.read_csv(raw_path)
        times = raw["time_s"].to_numpy(dtype=float)

        if (
            len(times) == 0
            or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError(f"Waktu tidak valid pada {sample_id}")

        image = np.load(npy_path, allow_pickle=False)

        if image.shape != (32, 32) or image.dtype != np.uint8:
            raise ValueError(f"Format NPY tidak sesuai pada {sample_id}")

        with Image.open(png_path) as png:
            if png.mode != "L" or not np.array_equal(
                np.asarray(png), image
            ):
                raise ValueError(f"PNG dan NPY berbeda pada {sample_id}")

        records.append({
            "sample_id": sample_id,
            "label": label,
            "label_id": 0 if label == "normal" else 1,
            "source": "synthetic",
            "attack_type": attack_type,
            "scenario": scenario,
            "session_id": "synthetic_session_001",
            "parent_sample_id": "synthetic_001",
            "start_time_s": float(times[0]),
            "end_time_s": float(times[-1]),
            "num_points": len(raw),
            "encoding": "fixed_scale_density_10m_v1",
            "meters_per_pixel": 10.0,
            "origin_column": 8,
            "origin_row": 16,
            "points_for_white": 8,
            "nonzero_pixels": int(np.count_nonzero(image)),
            "raw_path": raw_path.relative_to(PROJECT_DIR).as_posix(),
            "npy_path": npy_path.relative_to(PROJECT_DIR).as_posix(),
            "png_path": png_path.relative_to(PROJECT_DIR).as_posix(),
        })

    metadata = pd.DataFrame(records)
    output_path = DATASET_DIR / "dataset_metadata.csv"

    metadata.to_csv(output_path, index=False)

    print("Metadata berhasil dibuat.")
    print(f"Jumlah sampel: {len(metadata)}")
    print(f"Normal: {(metadata['label'] == 'normal').sum()}")
    print(f"Spoofed: {(metadata['label'] == 'spoofed').sum()}")
    print(f"Jumlah sesi asal: {metadata['session_id'].nunique()}")
    print()
    print(metadata[
        ["sample_id", "label", "attack_type", "nonzero_pixels"]
    ].to_string(index=False))
    print()
    print(f"File metadata: {output_path}")


if __name__ == "__main__":
    main()