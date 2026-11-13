from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset_sessions_v1"
OUTPUT_DIR = DATASET_DIR / "splits"

SPLIT_SEED = 2031


def main():
    metadata = pd.read_csv(DATASET_DIR / "dataset_metadata.csv")

    required = {
        "sample_id",
        "session_id",
        "parent_sample_id",
        "label",
        "label_id",
    }

    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Kolom belum lengkap: {sorted(missing)}")

    if metadata[list(required)].isna().any().any():
        raise ValueError("Metadata wajib mengandung nilai kosong.")

    if not metadata["sample_id"].is_unique:
        raise ValueError("sample_id duplikat.")

    sessions = np.array(sorted(metadata["session_id"].unique()))

    if len(sessions) != 10:
        raise ValueError("Pembagian ini membutuhkan tepat 10 sesi.")

    indexed = metadata.set_index("sample_id")

    for _, record in metadata.iterrows():
        parent_id = record["parent_sample_id"]

        if parent_id not in indexed.index:
            raise ValueError(f"Induk tidak ditemukan: {parent_id}")

        if indexed.loc[parent_id, "session_id"] != record["session_id"]:
            raise ValueError("Sesi sampel berbeda dari sesi induk.")

    rng = np.random.default_rng(SPLIT_SEED)
    shuffled_sessions = rng.permutation(sessions)

    session_groups = {
        "train": shuffled_sessions[:6],
        "validation": shuffled_sessions[6:8],
        "test": shuffled_sessions[8:],
    }

    session_sets = {
        name: set(values)
        for name, values in session_groups.items()
    }

    if (
        session_sets["train"] & session_sets["validation"]
        or session_sets["train"] & session_sets["test"]
        or session_sets["validation"] & session_sets["test"]
    ):
        raise ValueError("Ada sesi yang masuk lebih dari satu kelompok.")

    session_to_split = {
        session_id: split_name
        for split_name, group in session_groups.items()
        for session_id in group
    }

    combined = metadata.copy()
    combined["split"] = combined["session_id"].map(session_to_split)

    if combined["split"].isna().any():
        raise ValueError("Ada sampel yang belum mendapat kelompok.")

    parts = {}

    for name in session_groups:
        part = combined.loc[
            combined["split"] == name
        ].sort_values(["session_id", "sample_id"])

        if set(part["label"]) != {"normal", "spoofed"}:
            raise ValueError(f"Kelompok {name} belum memiliki kedua kelas.")

        parts[name] = part

    sample_sets = {
        name: set(part["sample_id"])
        for name, part in parts.items()
    }

    if (
        sample_sets["train"] & sample_sets["validation"]
        or sample_sets["train"] & sample_sets["test"]
        or sample_sets["validation"] & sample_sets["test"]
    ):
        raise ValueError("Ada sampel yang masuk lebih dari satu kelompok.")

    if sum(len(part) for part in parts.values()) != len(metadata):
        raise ValueError("Jumlah sampel setelah pembagian tidak sesuai.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, part in parts.items():
        part.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)

    combined.to_csv(
        OUTPUT_DIR / "all_samples_with_split.csv",
        index=False,
    )

    pd.DataFrame([
        {"session_id": session_id, "split": split_name}
        for session_id, split_name in sorted(session_to_split.items())
    ]).to_csv(
        OUTPUT_DIR / "session_assignments.csv",
        index=False,
    )

    print("Pembagian dataset selesai.")
    print(f"Seed: {SPLIT_SEED}")

    for name, part in parts.items():
        print()
        print(
            f"{name}: {len(part)} sampel, "
            f"{part['session_id'].nunique()} sesi"
        )
        print(f"Normal: {(part['label'] == 'normal').sum()}")
        print(f"Spoofed: {(part['label'] == 'spoofed').sum()}")
        print("Sesi:", ", ".join(sorted(part["session_id"].unique())))

    print()
    print("Pemeriksaan pemisahan sesi dan sampel: LULUS")
    print(f"Folder hasil: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()