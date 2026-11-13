import csv
import json
import re

from export_b2_golden_vectors import EXPORT_DIR, SCENARIOS


# Jumlah nilai, lebar bit, dan apakah nilainya bertanda.
WEIGHT_FILES = {
    "conv1_weights": (36, 8, True),
    "conv1_bias": (4, 32, True),
    "conv2_weights": (288, 8, True),
    "conv2_bias": (8, 32, True),
    "linear_weights": (1024, 8, True),
    "linear_bias": (2, 32, True),
    "pool1_multiplier": (1, 32, False),
    "pool2_multiplier": (1, 32, False),
}

SAMPLE_FILES = {
    "input": (1024, 8, False),
    "conv1_acc": (4096, 32, True),
    "pool1": (1024, 8, False),
    "conv2_acc": (2048, 32, True),
    "pool2": (512, 8, False),
    "logits": (2, 32, True),
    "predicted_class": (1, 8, False),
}


def read_memh(path, count, bits, signed):
    lines = path.read_text(encoding="ascii").splitlines()
    if len(lines) != count:
        raise ValueError(f"{path}: diharapkan {count} nilai, ada {len(lines)}.")

    pattern = re.compile(rf"[0-9A-Fa-f]{{{bits // 4}}}")
    values = []

    for line_number, line in enumerate(lines, start=1):
        if not pattern.fullmatch(line):
            raise ValueError(f"{path}, baris {line_number}: format hex salah.")

        value = int(line, 16)
        if signed and value >= (1 << (bits - 1)):
            value -= 1 << bits
        values.append(value)

    return values


def main():
    manifest = json.loads(
        (EXPORT_DIR / "manifest.json").read_text(encoding="utf-8")
    )

    with (EXPORT_DIR / "golden_summary.csv").open(
        encoding="utf-8", newline=""
    ) as summary_file:
        samples = list(csv.DictReader(summary_file))

    if len(samples) != len(SCENARIOS) or {
        row["scenario"] for row in samples
    } != set(SCENARIOS):
        raise ValueError("Ringkasan harus berisi satu sampel per skenario.")

    if len({row["sample_id"] for row in samples}) != len(samples):
        raise ValueError("Ada sample_id duplikat.")

    for name, spec in WEIGHT_FILES.items():
        values = read_memh(EXPORT_DIR / "weights" / f"{name}.memh", *spec)
        if name.endswith("_multiplier") and values != [manifest[name]]:
            raise ValueError(f"{name} berbeda dari manifest.json.")

    for row in samples:
        sample_id = row["sample_id"]
        sample_dir = EXPORT_DIR / sample_id
        tensors = {
            name: read_memh(sample_dir / f"{name}.memh", *spec)
            for name, spec in SAMPLE_FILES.items()
        }

        logits = tensors["logits"]
        expected_logits = [int(row["logit_normal"]), int(row["logit_spoofed"])]

        if logits != expected_logits:
            raise ValueError(f"{sample_id}: logit berbeda dari ringkasan.")

        expected_class = int(row["expected_class"])
        class_from_logits = int(logits[1] > logits[0])
        if (
            tensors["predicted_class"] != [expected_class]
            or class_from_logits != expected_class
        ):
            raise ValueError(f"{sample_id}: kelas tidak sesuai dengan logit.")

        print(f"LULUS {sample_id} | logit={logits} | kelas={expected_class}")

    total = len(WEIGHT_FILES) + len(samples) * len(SAMPLE_FILES)
    print(f"\nAudit lulus. {total} file .memh diperiksa.")


if __name__ == "__main__":
    main()