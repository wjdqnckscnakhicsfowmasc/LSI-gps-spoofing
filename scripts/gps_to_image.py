from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data_raw" / "gps_normal"
OUTPUT_DIR = PROJECT_DIR / "dataset" / "normal"
FIGURE_DIR = PROJECT_DIR / "reports" / "figures"

for folder in (RAW_DIR, OUTPUT_DIR, FIGURE_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# Contoh lintasan sintetis sebanyak 60 titik, satu titik per detik.
time_seconds = np.arange(60)
latitude = -6.973 + time_seconds * 0.00001
longitude = (
    107.630
    + time_seconds * 0.00001
    + 0.00008 * np.sin(time_seconds / 8)
)

gps_data = pd.DataFrame({
    "time_s": time_seconds,
    "latitude": latitude,
    "longitude": longitude,
})
gps_data.to_csv(RAW_DIR / "synthetic_001.csv", index=False)

# Proyeksi lokal pendekatan ke meter, relatif terhadap titik pertama.
earth_radius_m = 6_371_000
latitude_rad = np.radians(latitude)
longitude_rad = np.radians(longitude)

east_m = (
    earth_radius_m
    * (longitude_rad - longitude_rad[0])
    * np.cos(latitude_rad[0])
)
north_m = earth_radius_m * (latitude_rad - latitude_rad[0])

# Gunakan skala yang sama pada kedua sumbu agar bentuk tetap proporsional.
image_size = 32
span_m = max(float(np.ptp(east_m)), float(np.ptp(north_m)), 1.0)

pixel_x = np.rint(
    (east_m - east_m.min()) / span_m * (image_size - 1)
).astype(int)

pixel_y = np.rint(
    (north_m - north_m.min()) / span_m * (image_size - 1)
).astype(int)

# Baris gambar bertambah ke bawah, sehingga sumbu utara dibalik.
pixel_row = image_size - 1 - pixel_y

counts = np.zeros((image_size, image_size), dtype=np.uint32)
np.add.at(counts, (pixel_row, pixel_x), 1)

heatmap = np.rint(
    counts / counts.max() * 255
).astype(np.uint8)

np.save(OUTPUT_DIR / "synthetic_001.npy", heatmap)
Image.fromarray(heatmap).save(OUTPUT_DIR / "synthetic_001.png")

# Gambar pembanding untuk memeriksa lintasan dan hasil encoding.
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

axes[0].plot(east_m, north_m, ".-")
axes[0].set_title("Lintasan GPS sintetis")
axes[0].set_xlabel("Timur (meter)")
axes[0].set_ylabel("Utara (meter)")
axes[0].set_aspect("equal", adjustable="box")
axes[0].grid(True)

axes[1].imshow(
    heatmap,
    cmap="gray",
    vmin=0,
    vmax=255,
    interpolation="nearest",
)
axes[1].set_title("Heatmap 32 × 32")
axes[1].set_xlabel("Kolom piksel")
axes[1].set_ylabel("Baris piksel")

fig.tight_layout()
fig.savefig(FIGURE_DIR / "synthetic_001_preview.png", dpi=150)
plt.close(fig)

print("Heatmap berhasil dibuat")
print(f"Ukuran matriks: {heatmap.shape}")
print(f"Tipe data: {heatmap.dtype}")
print(f"Jumlah titik GPS: {len(gps_data)}")
print(f"Folder hasil: {OUTPUT_DIR}")