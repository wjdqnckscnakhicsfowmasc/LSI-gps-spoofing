from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset_real_time_5m_centered_30s_v1"
FIGURE_DIR = PROJECT_DIR / "reports" / "figures"

COLUMNS = 6


def main():
    metadata = pd.read_csv(DATASET_DIR / "dataset_metadata.csv")
    metadata = metadata.sort_values("sample_id").reset_index(drop=True)

    rows = int(np.ceil(len(metadata) / COLUMNS))
    figure, axes = plt.subplots(
        rows, COLUMNS, figsize=(2.2 * COLUMNS, 2.4 * rows), squeeze=False
    )

    for axis in axes.flat:
        axis.axis("off")

    for axis, record in zip(axes.flat, metadata.itertuples(index=False)):
        image = np.load(PROJECT_DIR / record.npy_path, allow_pickle=False)

        if image.shape != (32, 32) or image.dtype != np.uint8:
            raise ValueError(f"Format gambar tidak sesuai: {record.sample_id}")

        axis.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axis.set_title(
            f"{record.sample_id[-4:]}  ({record.nonzero_pixels} piksel)",
            fontsize=9,
        )
        axis.set_xticks([0, 16, 31])
        axis.set_yticks([0, 16, 31])
        axis.tick_params(labelsize=6)
        axis.axis("on")

    figure.suptitle(
        f"{len(metadata)} heatmap GPS asli — jendela 30 detik, 5 meter/piksel"
    )
    figure.tight_layout()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    preview_path = FIGURE_DIR / "real_time_5m_centered_30s_preview.png"
    figure.savefig(preview_path, dpi=150)
    plt.close(figure)

    print(f"Jumlah heatmap: {len(metadata)}")
    print("Piksel menyala per heatmap:")
    print(metadata["nonzero_pixels"].value_counts().sort_index().to_string())
    print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main()