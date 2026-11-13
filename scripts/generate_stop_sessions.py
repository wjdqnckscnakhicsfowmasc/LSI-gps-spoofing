from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from gps_encoder import ENCODING_ID, METERS_PER_PIXEL, encode_gps


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset_sessions_v2"
STOP_SEED = 2041


def main():
    source_metadata = pd.read_csv(
        DATASET_DIR / "normal_metadata.csv"
    ).sort_values("session_id")

    if len(source_metadata) != 100:
        raise ValueError("Diperlukan 100 sampel normal bergerak.")

    if (
        not source_metadata["sample_id"].is_unique
        or not source_metadata["session_id"].is_unique
    ):
        raise ValueError("ID sampel atau sesi dasar duplikat.")

    if not source_metadata["encoding"].eq(ENCODING_ID).all():
        raise ValueError("Encoding berbeda dari encoder aktif.")

    if not (
        source_metadata["label"].eq("normal").all()
        and source_metadata["scenario"].eq("moving").all()
    ):
        raise ValueError("Input harus berupa sampel normal bergerak.")

    rng = np.random.default_rng(STOP_SEED)
    generated = []
    stop_records = []

    # Siapkan dan validasi seluruh sampel sebelum menyimpan.
    for _, record in source_metadata.iterrows():
        source_path = PROJECT_DIR / record["raw_path"]
        normal = pd.read_csv(source_path)

        times = normal["time_s"].to_numpy(dtype=float)
        coordinates = normal[["latitude", "longitude"]].to_numpy(
            dtype=float
        )

        if (
            len(times) < 2
            or not np.isfinite(times).all()
            or not np.isfinite(coordinates).all()
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError(f"Data tidak valid: {source_path.name}")

        encode_gps(normal)

        stop_start = int(rng.integers(15, 26))
        stop_duration = int(rng.integers(10, 21))
        stop_end = stop_start + stop_duration

        if not times[0] <= stop_start < stop_end < times[-1]:
            raise ValueError(
                f"Interval berhenti tidak sesuai: {source_path.name}"
            )

        # Waktu pengamatan tetap berjalan.
        # Pergerakan mengikuti waktu perjalanan yang tertunda.
        elapsed_stop = np.clip(
            times - stop_start,
            0,
            stop_duration,
        )
        route_times = times - elapsed_stop

        stopped = normal.copy()
        stopped["latitude"] = np.interp(
            route_times, times, coordinates[:, 0]
        )
        stopped["longitude"] = np.interp(
            route_times, times, coordinates[:, 1]
        )

        image = encode_gps(stopped)

        session_id = record["session_id"]
        sample_id = f"{session_id}_stop"

        raw_path = (
            DATASET_DIR / "raw" / "normal" / f"{sample_id}.csv"
        )
        npy_path = DATASET_DIR / "normal" / f"{sample_id}.npy"
        png_path = DATASET_DIR / "normal" / f"{sample_id}.png"

        stop_record = record.to_dict()
        stop_record.update({
            "sample_id": sample_id,
            "parent_sample_id": record["sample_id"],
            "scenario": "stop_and_resume",
            "stop_start_s": stop_start,
            "stop_duration_s": stop_duration,
            "stop_end_s": stop_end,
            "augmentation_seed": STOP_SEED,
            "nonzero_pixels": int(np.count_nonzero(image)),
            "raw_path": raw_path.relative_to(PROJECT_DIR).as_posix(),
            "npy_path": npy_path.relative_to(PROJECT_DIR).as_posix(),
            "png_path": png_path.relative_to(PROJECT_DIR).as_posix(),
        })

        stop_records.append(stop_record)
        generated.append((
            session_id,
            sample_id,
            stopped,
            image,
            raw_path,
            npy_path,
            png_path,
            stop_start,
            stop_duration,
        ))

    stop_metadata = pd.DataFrame(stop_records)
    combined = pd.concat(
        [source_metadata, stop_metadata],
        ignore_index=True,
    )

    if not combined["sample_id"].is_unique:
        raise ValueError("sample_id gabungan duplikat.")

    # Simpan seluruh 100 versi berhenti.
    for sample in generated:
        (
            session_id,
            sample_id,
            stopped,
            image,
            raw_path,
            npy_path,
            png_path,
            stop_start,
            stop_duration,
        ) = sample

        raw_path.parent.mkdir(parents=True, exist_ok=True)
        npy_path.parent.mkdir(parents=True, exist_ok=True)

        stopped.to_csv(raw_path, index=False)
        np.save(npy_path, image)
        Image.fromarray(image).save(png_path)

    metadata_path = DATASET_DIR / "normal_with_stops_metadata.csv"
    combined.to_csv(metadata_path, index=False)

    # Tampilkan 10 contoh pertama.
    figure, axes = plt.subplots(2, 5, figsize=(15, 6))

    for axis, sample in zip(axes.flat, generated[:10]):
        (
            session_id,
            sample_id,
            stopped,
            image,
            raw_path,
            npy_path,
            png_path,
            stop_start,
            stop_duration,
        ) = sample

        axis.imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=255,
            interpolation="nearest",
        )
        axis.set_title(
            f"{session_id}\n"
            f"Berhenti {stop_duration} s, mulai {stop_start} s",
            fontsize=9,
        )
        axis.set_xticks([0, 16, 31])
        axis.set_yticks([0, 16, 31])

    figure.suptitle(
        "10 contoh dari 100 sesi berhenti lalu lanjut"
        f" — {METERS_PER_PIXEL:g} meter/piksel"
    )
    figure.tight_layout()

    figure_dir = PROJECT_DIR / "reports" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    preview_path = figure_dir / "stop_sessions_v2.png"

    figure.savefig(preview_path, dpi=150)
    plt.close(figure)

    print("Pembuatan sesi berhenti v2 selesai.")
    print(f"Sampel bergerak: {len(source_metadata)}")
    print(f"Sampel berhenti: {len(stop_metadata)}")
    print(f"Total sampel normal: {len(combined)}")
    print(f"Jumlah sesi asal: {combined['session_id'].nunique()}")
    print()
    print("Ringkasan 10 sampel pertama:")
    print(stop_metadata[[
        "sample_id",
        "stop_start_s",
        "stop_duration_s",
        "nonzero_pixels",
    ]].head(10).to_string(index=False))
    print()
    print(f"Metadata gabungan: {metadata_path}")
    print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main()