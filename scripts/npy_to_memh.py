from pathlib import Path
import sys

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent


def write_memh(path, values, bits):
    """Tulis satu nilai heksadesimal per baris dalam urutan C."""
    mask = (1 << bits) - 1
    width = bits // 4
    lines = [
        f"{int(value) & mask:0{width}X}"
        for value in np.asarray(values).ravel(order="C")
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Cara pakai: python scripts\\npy_to_memh.py <file.npy>"
        )

    npy_path = Path(sys.argv[1])
    if not npy_path.is_absolute():
        npy_path = PROJECT_DIR / npy_path

    image = np.load(npy_path, allow_pickle=False)

    if image.shape != (32, 32) or image.dtype != np.uint8:
        raise ValueError(
            f"Gambar harus 32x32 uint8, ditemukan {image.shape} {image.dtype}."
        )

    # Format sama dengan input.memh pada golden vectors:
    # 1.024 baris, dua digit heksadesimal (00-FF), urutan baris lalu kolom.
    memh_path = npy_path.with_suffix(".memh")
    write_memh(memh_path, image, 8)

    lines = memh_path.read_text(encoding="ascii").split()

    if len(lines) != 1024:
        raise ValueError(f"Jumlah baris {len(lines)}, seharusnya 1024.")

    restored = np.array([int(line, 16) for line in lines], dtype=np.uint8)

    if not np.array_equal(restored.reshape(32, 32), image):
        raise ValueError("Isi file memh tidak sama dengan gambar asal.")

    print(f"Sumber: {npy_path}")
    print(f"Hasil : {memh_path}")
    print(f"Jumlah baris: {len(lines)}")
    print(f"Piksel menyala: {int(np.count_nonzero(image))}")
    print(f"Nilai piksel: min {int(image.min())}, maks {int(image.max())}")
    print("Pemeriksaan baca ulang: cocok")


if __name__ == "__main__":
    main()