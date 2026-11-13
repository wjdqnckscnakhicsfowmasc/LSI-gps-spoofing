from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "dataset" / "spoofed"
RAW_DIR = PROJECT_DIR / "data_raw" / "gps_spoofed"
FIGURE_DIR = PROJECT_DIR / "reports" / "figures"

for folder in (OUTPUT_DIR, RAW_DIR, FIGURE_DIR):
    folder.mkdir(parents=True, exist_ok=True)

normal = pd.read_csv(
    PROJECT_DIR / "data_raw" / "gps_normal" / "synthetic_001.csv"
)
spoofed = normal.copy()

# Mulai titik ke-31, geser posisi sekitar 100 meter ke arah timur.
earth_radius_m = 6_371_000
latitude_origin = np.radians(normal["latitude"].iloc[0])
jump_m = 100.0
longitude_offset = np.degrees(
    jump_m / (earth_radius_m * np.cos(latitude_origin))
)
spoofed.loc[30:, "longitude"] += longitude_offset

spoofed.to_csv(RAW_DIR / "synthetic_jump_001.csv", index=False)

latitude_origin_deg = normal["latitude"].iloc[0]
longitude_origin_deg = normal["longitude"].iloc[0]

trajectories = []

for data in (normal, spoofed):
    east_m = (
        earth_radius_m
        * np.radians(data["longitude"].to_numpy() - longitude_origin_deg)
        * np.cos(latitude_origin)
    )
    north_m = (
        earth_radius_m
        * np.radians(data["latitude"].to_numpy() - latitude_origin_deg)
    )
    trajectories.append((east_m, north_m))

# Skala bersama khusus untuk visualisasi perbandingan dua lintasan ini.
all_east = np.concatenate([east for east, north in trajectories])
all_north = np.concatenate([north for east, north in trajectories])

east_min = all_east.min()
north_min = all_north.min()
span_m = max(float(np.ptp(all_east)), float(np.ptp(all_north)), 1.0)

heatmaps = []

for east_m, north_m in trajectories:
    pixel_x = np.rint((east_m - east_min) / span_m * 31).astype(int)
    pixel_y = np.rint((north_m - north_min) / span_m * 31).astype(int)

    counts = np.zeros((32, 32), dtype=np.uint32)
    np.add.at(counts, (31 - pixel_y, pixel_x), 1)
    heatmaps.append(counts)

# Kecerahan bersama agar kedua gambar bisa dibandingkan.
max_count = max(int(counts.max()) for counts in heatmaps)
images = [
    np.rint(counts / max_count * 255).astype(np.uint8)
    for counts in heatmaps
]

np.save(OUTPUT_DIR / "synthetic_jump_001.npy", images[1])
Image.fromarray(images[1]).save(
    OUTPUT_DIR / "synthetic_jump_001.png"
)

fig, axes = plt.subplots(2, 2, figsize=(10, 7))

for column, (label, trajectory, image) in enumerate(
    zip(("Normal sintetis", "Lonjakan sintetis"), trajectories, images)
):
    east_m, north_m = trajectory

    axes[0, column].plot(east_m, north_m, ".-")
    axes[0, column].set_title(label)
    axes[0, column].set_xlabel("Timur (meter)")
    axes[0, column].set_ylabel("Utara (meter)")
    axes[0, column].set_xlim(east_min - 5, east_min + span_m + 5)
    axes[0, column].set_ylim(north_min - 5, north_min + span_m + 5)
    axes[0, column].set_aspect("equal", adjustable="box")
    axes[0, column].grid(True)

    axes[1, column].imshow(
        image, cmap="gray", vmin=0, vmax=255, interpolation="nearest"
    )
    axes[1, column].set_title(f"Heatmap {label.lower()} 32 × 32")
    axes[1, column].set_xlabel("Kolom piksel")
    axes[1, column].set_ylabel("Baris piksel")

fig.tight_layout()
fig.savefig(FIGURE_DIR / "jump_comparison.png", dpi=150)
plt.close(fig)

print("Contoh lonjakan posisi berhasil dibuat")
print("Pergeseran sintetis: 100 meter ke timur mulai titik ke-31")
print(f"Preview: {FIGURE_DIR / 'jump_comparison.png'}")