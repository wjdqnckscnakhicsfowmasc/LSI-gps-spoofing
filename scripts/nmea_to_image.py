from pathlib import Path
import sys
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

def nmea_to_decimal(val, hemi):
    if not val or not hemi:
        return None
    deg_len = 2 if hemi in ("N", "S") else 3
    deg = float(val[:deg_len]) + float(val[deg_len:]) / 60.0
    return -deg if hemi in ("S", "W") else deg

def process_nmea(nmea_filename="session_001_diam.nmea"):
    nmea_path = RAW_DIR / nmea_filename
    stem = nmea_path.stem

    lats, lons = [], []
    with open(nmea_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "RMC" in line:
                parts = line.strip().split(",")
                if len(parts) > 6 and parts[2] == "A":
                    lat = nmea_to_decimal(parts[3], parts[4])
                    lon = nmea_to_decimal(parts[5], parts[6])
                    if lat is not None and lon is not None:
                        lats.append(lat)
                        lons.append(lon)

    latitude = np.array(lats)
    longitude = np.array(lons)
    time_seconds = np.arange(len(latitude))

    # Simpan juga ke format CSV standar B2 (time_s,latitude,longitude)
    gps_data = pd.DataFrame({
        "time_s": time_seconds,
        "latitude": latitude,
        "longitude": longitude,
    })
    csv_path = RAW_DIR / f"{stem}.csv"
    gps_data.to_csv(csv_path, index=False)

    # Proyeksi lokal ke meter (identik dengan gps_to_image.py)
    earth_radius_m = 6_371_000
    latitude_rad = np.radians(latitude)
    longitude_rad = np.radians(longitude)

    east_m = (
        earth_radius_m
        * (longitude_rad - longitude_rad[0])
        * np.cos(latitude_rad[0])
    )
    north_m = earth_radius_m * (latitude_rad - latitude_rad[0])

    # Buat Heatmap 32x32
    image_size = 32
    span_m = max(float(np.ptp(east_m)), float(np.ptp(north_m)), 1.0)

    pixel_x = np.rint(
        (east_m - east_m.min()) / span_m * (image_size - 1)
    ).astype(int)
    pixel_y = np.rint(
        (north_m - north_m.min()) / span_m * (image_size - 1)
    ).astype(int)
    pixel_row = image_size - 1 - pixel_y

    counts = np.zeros((image_size, image_size), dtype=np.uint32)
    np.add.at(counts, (pixel_row, pixel_x), 1)

    heatmap = np.rint(counts / counts.max() * 255).astype(np.uint8)

    np.save(OUTPUT_DIR / f"{stem}.npy", heatmap)
    Image.fromarray(heatmap).save(OUTPUT_DIR / f"{stem}.png")

    # Gambar pembanding berdampingan (Lintasan GPS Nyata vs Heatmap 32x32)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    axes[0].plot(east_m, north_m, ".-")
    axes[0].set_title(f"Lintasan GPS Nyata ({stem})")
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
    preview_path = FIGURE_DIR / f"{stem}_preview.png"
    fig.savefig(preview_path, dpi=150)
    plt.close(fig)

    print(f"[SUKSES] File CSV disimpan di    : {csv_path}")
    print(f"[SUKSES] Heatmap 32x32 disimpan : {OUTPUT_DIR / f'{stem}.png'}")
    print(f"[SUKSES] Grafik Preview disimpan: {preview_path}")
    print(f"Jumlah titik GPS valid: {len(gps_data)} titik | Rentang gerak: {span_m:.2f} meter")

if __name__ == "__main__":
    fname = sys.argv[1] if len(sys.argv) > 1 else "session_001_diam.nmea"
    process_nmea(fname)