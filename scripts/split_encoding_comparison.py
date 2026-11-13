from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent

DENSITY_DIR = PROJECT_DIR / "dataset_sessions_v2"
TIME_DIR = PROJECT_DIR / "dataset_time_5m_centered_v1"
SPLIT_SEED = 2031

MATCH_COLUMNS = [
    "session_id",
    "parent_sample_id",
    "label",
    "label_id",
    "raw_path",
]


def load_metadata(dataset_dir):
    metadata = pd.read_csv(dataset_dir / "dataset_metadata.csv")
    required = ["sample_id", *MATCH_COLUMNS, "npy_path"]

    if metadata[required].isna().any().any():
        raise ValueError(f"Metadata tidak lengkap di {dataset_dir.name}.")

    if len(metadata) != 500 or not metadata["sample_id"].is_unique:
        raise ValueError(
            f"{dataset_dir.name} harus memiliki 500 ID sampel unik."
        )

    return metadata.set_index("sample_id").sort_index()


def main():
    density = load_metadata(DENSITY_DIR)
    time = load_metadata(TIME_DIR)

    if not density[MATCH_COLUMNS].equals(time[MATCH_COLUMNS]):
        raise ValueError(
            "ID, label, sesi, atau sumber GPS berbeda antara kedua encoding."
        )

    session_counts = pd.crosstab(density["session_id"], density["label"])

    if (
        len(session_counts) != 100
        or set(session_counts.columns) != {"normal", "spoofed"}
        or not session_counts["normal"].eq(2).all()
        or not session_counts["spoofed"].eq(3).all()
    ):
        raise ValueError("Harus ada 100 sesi, masing-masing 2 normal dan 3 spoofed.")

    rng = np.random.default_rng(SPLIT_SEED)
    sessions = rng.permutation(session_counts.index.to_numpy())

    split_groups = {
        "train": sessions[:60],
        "validation": sessions[60:80],
        "test": sessions[80:],
    }
    session_to_split = {
        session_id: split_name
        for split_name, group in split_groups.items()
        for session_id in group
    }

    for dataset_dir, metadata in ((DENSITY_DIR, density), (TIME_DIR, time)):
        split_dir = dataset_dir / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)

        assigned = metadata.reset_index()
        assigned["split"] = assigned["session_id"].map(session_to_split)

        for split_name in split_groups:
            samples = assigned.loc[assigned["split"].eq(split_name)]
            samples.to_csv(split_dir / f"{split_name}.csv", index=False)

        print(f"CSV tersimpan di {split_dir}")

    print()
    for split_name, group in split_groups.items():
        count = len(group)
        print(
            f"{split_name}: {count} sesi, "
            f"{2 * count} normal, {3 * count} spoofed"
        )


if __name__ == "__main__":
    main()