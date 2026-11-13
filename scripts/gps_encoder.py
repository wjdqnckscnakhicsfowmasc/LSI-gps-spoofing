import numpy as np


IMAGE_SIZE = 32
METERS_PER_PIXEL = 10.0
ORIGIN_COLUMN = 16
ORIGIN_ROW = 16
POINTS_FOR_WHITE = 8
EARTH_RADIUS_M = 6_371_000

ENCODING_ID = "fixed_density_10m_center16_v1"


def encode_local_positions(east_m, north_m):
    """Ubah posisi lokal relatif terhadap titik awal menjadi heatmap."""
    east = np.asarray(east_m, dtype=np.float64)
    north = np.asarray(north_m, dtype=np.float64)

    if (
        east.ndim != 1
        or north.ndim != 1
        or east.shape != north.shape
        or east.size == 0
    ):
        raise ValueError("Posisi harus berupa dua array 1D dengan panjang sama.")

    if not np.isfinite(east).all() or not np.isfinite(north).all():
        raise ValueError("Posisi mengandung nilai tidak valid.")

    column_positions = ORIGIN_COLUMN + np.floor(
        east / METERS_PER_PIXEL + 0.5
    )
    row_positions = ORIGIN_ROW - np.floor(
        north / METERS_PER_PIXEL + 0.5
    )

    inside = (
        (column_positions >= 0)
        & (column_positions < IMAGE_SIZE)
        & (row_positions >= 0)
        & (row_positions < IMAGE_SIZE)
    )

    if not inside.all():
        outside_count = int(np.count_nonzero(~inside))
        raise ValueError(f"{outside_count} titik keluar gambar.")

    columns = column_positions.astype(np.int64)
    rows = row_positions.astype(np.int64)

    counts = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint32)
    np.add.at(counts, (rows, columns), 1)

    heatmap = np.rint(
        np.minimum(counts, POINTS_FOR_WHITE)
        / POINTS_FOR_WHITE
        * 255
    ).astype(np.uint8)

    return heatmap


def gps_to_local_meters(data):
    """Proyeksi lokal untuk lintasan pendek, relatif ke GPS pertama."""
    coordinates = data[["latitude", "longitude"]].to_numpy(
        dtype=np.float64
    )

    if len(coordinates) == 0 or not np.isfinite(coordinates).all():
        raise ValueError("Koordinat GPS kosong atau tidak valid.")

    latitude_deg = coordinates[:, 0]
    longitude_deg = coordinates[:, 1]

    if (
        np.any(np.abs(latitude_deg) > 90)
        or np.any(np.abs(longitude_deg) > 180)
    ):
        raise ValueError("Latitude atau longitude di luar rentang.")

    latitude = np.radians(latitude_deg)
    longitude = np.radians(longitude_deg)

    longitude_difference = (
        longitude - longitude[0] + np.pi
    ) % (2 * np.pi) - np.pi

    east_m = (
        EARTH_RADIUS_M
        * longitude_difference
        * np.cos(latitude[0])
    )
    north_m = EARTH_RADIUS_M * (latitude - latitude[0])

    return east_m, north_m


def encode_gps(data):
    """Ubah koordinat GPS menjadi gambar grayscale 32 × 32."""
    east_m, north_m = gps_to_local_meters(data)
    return encode_local_positions(east_m, north_m)