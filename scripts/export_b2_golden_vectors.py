import json

import numpy as np
import pandas as pd
import torch

from compare_encodings import DATASETS, load_split
from evaluate_b2_fixed_point import (
    CHECKPOINT_PATH,
    EXPERIMENT,
    REPORT_DIR,
    RESCALE_BITS,
    activation_scale,
    check_int32,
    conv_int,
    quantize_layer,
    relu_pool_rescale,
)
from train_cnn import SmallCNN


EXPORT_DIR = REPORT_DIR / "golden_vectors"

SCENARIOS = (
    "moving",
    "stop_and_resume",
    "position_jump",
    "gradual_offset",
    "position_freeze",
)


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
    torch.set_num_threads(2)

    dataset_dir, encoding_id = DATASETS[EXPERIMENT]
    train_meta, train_images, _ = load_split(
        dataset_dir, encoding_id, "train"
    )
    val_meta, val_images, _ = load_split(
        dataset_dir, encoding_id, "validation"
    )

    if set(train_meta["session_id"]) & set(val_meta["session_id"]):
        raise ValueError("Sesi train dan validation bertumpang tindih.")

    checkpoint = torch.load(
        CHECKPOINT_PATH, map_location="cpu", weights_only=True
    )
    expected_checkpoint = {
        "architecture": "SmallCNN_1_4_8_v1",
        "encoding": encoding_id,
        "input_normalization": "uint8 divided by 255",
        "epoch": 26,
    }

    for key, expected in expected_checkpoint.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"Checkpoint tidak sesuai pada bagian {key}.")

    model = SmallCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Hitung ulang skala aktivasi dari data train yang sama.
    with torch.inference_mode():
        train_pool1 = model.features[:3](train_images)
        train_pool2 = model.features[3:](train_pool1)
        scale1 = activation_scale(train_pool1)
        scale2 = activation_scale(train_pool2)

    input_scale = 1.0 / 255.0
    w1, b1, acc_scale1 = quantize_layer(
        model.features[0], input_scale, "Conv1"
    )
    w2, b2, acc_scale2 = quantize_layer(
        model.features[3], scale1, "Conv2"
    )
    w3, b3, logit_scale = quantize_layer(
        model.classifier, scale2, "Linear"
    )

    reference_table = pd.read_csv(
        REPORT_DIR / "validation_predictions.csv"
    ).set_index("sample_id")

    if not reference_table.index.is_unique:
        raise ValueError("Ada sample_id duplikat pada hasil validation.")

    reference = reference_table["integer_prediction"]

    if set(reference.index) != set(val_meta["sample_id"]):
        raise ValueError("Daftar sampel berbeda dari evaluasi fixed-point.")

    selected = []
    for scenario in SCENARIOS:
        candidates = val_meta.loc[val_meta["scenario"].eq(scenario)]

        if candidates.empty:
            raise ValueError(f"Skenario tidak ditemukan: {scenario}")

        # Sertakan contoh alarm palsu saat bergerak normal jika tersedia.
        if scenario == "moving":
            false_alarms = candidates.loc[
                candidates["label_id"].eq(0)
                & candidates["sample_id"].map(reference).eq(1)
            ]
            if not false_alarms.empty:
                candidates = false_alarms

        selected.append(candidates.index[0])

    sample_metadata = val_meta.loc[selected]
    pixels = np.rint(
        val_images[selected].numpy() * 255.0
    ).astype(np.uint8)

    conv1 = conv_int(pixels, w1, b1, "Conv1")
    pool1, multiplier1, _ = relu_pool_rescale(conv1, acc_scale1, scale1)

    conv2 = conv_int(pool1, w2, b2, "Conv2")
    pool2, multiplier2, _ = relu_pool_rescale(conv2, acc_scale2, scale2)

    flattened = pool2.reshape(len(pool2), -1).astype(np.int64)
    logits = flattened @ w3.astype(np.int64).T + b3
    check_int32(logits, "Linear accumulator")

    predictions = logits.argmax(axis=1)
    expected = sample_metadata["sample_id"].map(reference).to_numpy()

    if not np.array_equal(predictions, expected):
        raise ValueError("Prediksi ekspor berbeda dari evaluasi fixed-point.")

    parameters = {
        "conv1_weights": (w1, 8),
        "conv1_bias": (b1, 32),
        "conv2_weights": (w2, 8),
        "conv2_bias": (b2, 32),
        "linear_weights": (w3, 8),
        "linear_bias": (b3, 32),
        "pool1_multiplier": ([multiplier1], 32),
        "pool2_multiplier": ([multiplier2], 32),
    }

    weights_dir = EXPORT_DIR / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    for name, (values, bits) in parameters.items():
        write_memh(weights_dir / f"{name}.memh", values, bits)

    records = []
    for index, row in enumerate(sample_metadata.itertuples(index=False)):
        sample_dir = EXPORT_DIR / row.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        tensors = {
            "input": (pixels[index, 0], 8),
            "conv1_acc": (conv1[index], 32),
            "pool1": (pool1[index], 8),
            "conv2_acc": (conv2[index], 32),
            "pool2": (pool2[index], 8),
            "logits": (logits[index], 32),
            "predicted_class": ([predictions[index]], 8),
        }

        for name, (values, bits) in tensors.items():
            write_memh(sample_dir / f"{name}.memh", values, bits)

        records.append({
            "sample_id": row.sample_id,
            "scenario": row.scenario,
            "true_label": int(row.label_id),
            "expected_class": int(predictions[index]),
            "logit_normal": int(logits[index, 0]),
            "logit_spoofed": int(logits[index, 1]),
        })

    rounding_offset = 1 << (RESCALE_BITS - 1)
    manifest = {
        "architecture": checkpoint["architecture"],
        "encoding": encoding_id,
        "checkpoint_epoch": checkpoint["epoch"],
        "class_order": ["normal", "spoofed"],
        "tie_break_class": 0,
        "serialization": "C order, one hexadecimal value per line",
        "signed_values": "two's complement",
        "layout": {
            "input": "row, column",
            "activations": "channel, row, column",
            "conv_weights": "out_channel, in_channel, kernel_row, kernel_column",
            "linear_weights": "out_feature, in_feature",
            "flatten": "channel, row, column",
        },
        "rescale_bits": RESCALE_BITS,
        "rescale_rule": (
            "After ReLU and max pooling: "
            f"clip((x * M + {rounding_offset}) >> {RESCALE_BITS}, 0, 255)"
        ),
        "input_scale": input_scale,
        "pool1_multiplier": multiplier1,
        "pool2_multiplier": multiplier2,
        "pool1_scale": scale1,
        "pool2_scale": scale2,
        "logit_scale": logit_scale,
    }

    (EXPORT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = pd.DataFrame(records)
    summary.to_csv(EXPORT_DIR / "golden_summary.csv", index=False)

    print("Ekspor golden vectors selesai.")
    print(summary.to_string(index=False))
    print(f"\nFolder hasil: {EXPORT_DIR}")


if __name__ == "__main__":
    main()