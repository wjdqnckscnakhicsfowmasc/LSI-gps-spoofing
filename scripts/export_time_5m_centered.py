from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from check_scale_5m_centered import METERS_PER_PIXEL, inspect_sample


PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCE_METADATA = PROJECT_DIR / "dataset_sessions_v2" / "dataset_metadata.csv"
OUTPUT_DIR = PROJECT_DIR / "dataset_time_5m_centered_v1"
ENCODING_ID = "last_visit_time_5m_bbox_center_v1"


def main():
    metadata = pd.read_csv(SOURCE_METADATA)

    if len(metadata) != 500 or not metadata["sample_id"].is_unique:
        raise ValueError("Metadata harus berisi 500 sampel dengan ID unik.")

    samples = []

    # Periksa seluruh gambar sebelum menulis hasil.
    for record in metadata.to_dict("records"):
        sample_id = record["sample_id"]
        label = record["label"]

        if label not in {"normal", "spoofed"}:
            raise ValueError(f"Label tidak valid pada {sample_id}: {label}")

        data = pd.read_csv(PROJECT_DIR / record["raw_path"])
        outside_count, image = inspect_sample(data)

        if image is None:
            raise ValueError(
                f"{sample_id}: {outside_count} titik keluar batas gambar."
            )

        samples.append((record, image))

    records = []

    for record, image in samples:
        sample_dir = OUTPUT_DIR / record["label"]
        sample_dir.mkdir(parents=True, exist_ok=True)

        sample_id = record["sample_id"]
        npy_path = sample_dir / f"{sample_id}.npy"
        png_path = sample_dir / f"{sample_id}.png"

        np.save(npy_path, image, allow_pickle=False)
        Image.fromarray(image).save(png_path)

        # Buang pengaturan yang hanya berlaku pada encoding kepadatan.
        for field in ("origin_column", "origin_row", "points_for_white"):
            record.pop(field, None)

        record.update({
            "encoding": ENCODING_ID,
            "meters_per_pixel": METERS_PER_PIXEL,
            "centering": "trajectory_bounding_box",
            "pixel_value_rule": "last_visit_time_1_to_255",
            "time_reference": "relative_to_sample_duration",
            "nonzero_pixels": int(np.count_nonzero(image)),
            "npy_path": npy_path.relative_to(PROJECT_DIR).as_posix(),
            "png_path": png_path.relative_to(PROJECT_DIR).as_posix(),
        })
        records.append(record)

    exported = pd.DataFrame(records)
    metadata_path = OUTPUT_DIR / "dataset_metadata.csv"
    exported.to_csv(metadata_path, index=False)

    print("Ekspor encoding waktu 5 meter/piksel selesai.")
    print(f"Jumlah sampel: {len(exported)}")
    print(f"Jumlah sesi: {exported['session_id'].nunique()}")
    print(exported["label"].value_counts().to_string())
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()