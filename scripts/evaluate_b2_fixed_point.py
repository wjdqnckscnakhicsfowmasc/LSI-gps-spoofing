from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

from compare_encodings import DATASETS, OUTPUT_DIR, load_split
from train_cnn import SmallCNN


EXPERIMENT = "time_5m_centered"
CHECKPOINT_PATH = OUTPUT_DIR / EXPERIMENT / "best_model.pth"
REPORT_DIR = OUTPUT_DIR / "fixed_point_time_5m_v1"

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
RESCALE_BITS = 31


def check_int32(values, name):
    if np.any(values < INT32_MIN) or np.any(values > INT32_MAX):
        raise OverflowError(f"{name} melampaui rentang signed INT32.")


def quantize_layer(layer, input_scale, name):
    weights = layer.weight.detach().cpu().numpy().astype(np.float64)
    bias = layer.bias.detach().cpu().numpy().astype(np.float64)

    max_weight = float(np.max(np.abs(weights)))
    weight_scale = max_weight / 127.0 if max_weight > 0 else 1.0
    accumulator_scale = input_scale * weight_scale

    weight_int = np.clip(
        np.rint(weights / weight_scale), -127, 127
    ).astype(np.int8)

    bias_rounded = np.rint(bias / accumulator_scale)
    check_int32(bias_rounded, f"{name} bias")
    bias_int = bias_rounded.astype(np.int32)

    # Batas konservatif untuk seluruh hasil kali dan bias.
    # Input setiap layer memiliki rentang 0–255.
    worst_case = (
        255 * np.abs(weight_int.astype(np.int64))
        .reshape(weight_int.shape[0], -1)
        .sum(axis=1)
        + np.abs(bias_int.astype(np.int64))
    )
    if np.any(worst_case > INT32_MAX):
        raise OverflowError(f"{name} berisiko overflow pada accumulator INT32.")

    return weight_int, bias_int, accumulator_scale


def conv_int(images, weights, bias, name):
    padded = np.pad(
        images.astype(np.int64),
        ((0, 0), (0, 0), (1, 1), (1, 1)),
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, (3, 3), axis=(2, 3)
    )

    accumulator = np.einsum(
        "nchwij,ocij->nohw",
        windows,
        weights.astype(np.int64),
        optimize=False,
    )
    accumulator += bias.astype(np.int64)[None, :, None, None]

    check_int32(accumulator, f"{name} accumulator")
    return accumulator


def relu_pool_rescale(accumulator, accumulator_scale, output_scale):
    # ReLU lalu max pooling 2×2 dengan stride 2.
    positive = np.maximum(accumulator, 0)
    batch, channels, height, width = positive.shape
    pooled = positive.reshape(
        batch, channels, height // 2, 2, width // 2, 2
    ).max(axis=(3, 5))

    multiplier = int(round(
        accumulator_scale / output_scale * (1 << RESCALE_BITS)
    ))
    if not 1 <= multiplier <= INT32_MAX:
        raise ValueError("Multiplier rescale berada di luar rentang usulan.")

    # Hasil ReLU nonnegatif. Nilai tepat di tengah dibulatkan ke atas.
    rounded = (
        pooled * multiplier + (1 << (RESCALE_BITS - 1))
    ) >> RESCALE_BITS

    clipped_count = int(np.count_nonzero(rounded > 255))
    activation = np.clip(rounded, 0, 255).astype(np.uint8)

    return activation, multiplier, clipped_count


def activation_scale(activation):
    peak = float(activation.max().item())
    return peak / 255.0 if peak > 0 else 1.0


def main():
    torch.set_num_threads(2)

    dataset_dir, encoding_id = DATASETS[EXPERIMENT]
    train_meta, train_images, _ = load_split(
        dataset_dir, encoding_id, "train"
    )
    val_meta, val_images, val_labels = load_split(
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

    with torch.inference_mode():
        train_pool1 = model.features[:3](train_images)
        train_pool2 = model.features[3:](train_pool1)
        scale_pool1 = activation_scale(train_pool1)
        scale_pool2 = activation_scale(train_pool2)
        float_predictions = model(val_images).argmax(dim=1).numpy()

    input_scale = 1.0 / 255.0
    pixels = np.rint(val_images.numpy() * 255.0).astype(np.uint8)

    w1, b1, acc_scale1 = quantize_layer(
        model.features[0], input_scale, "Conv1"
    )
    w2, b2, acc_scale2 = quantize_layer(
        model.features[3], scale_pool1, "Conv2"
    )
    w3, b3, _ = quantize_layer(
        model.classifier, scale_pool2, "Linear"
    )

    pool1, multiplier1, clipped1 = relu_pool_rescale(
        conv_int(pixels, w1, b1, "Conv1"), acc_scale1, scale_pool1
    )
    pool2, multiplier2, clipped2 = relu_pool_rescale(
        conv_int(pool1, w2, b2, "Conv2"), acc_scale2, scale_pool2
    )

    flattened = pool2.reshape(len(pool2), -1).astype(np.int64)
    integer_logits = flattened @ w3.astype(np.int64).T + b3
    check_int32(integer_logits, "Linear accumulator")

    # Kedua logit memakai skala yang sama, jadi bisa langsung dibandingkan.
    integer_predictions = integer_logits.argmax(axis=1)
    actual = val_labels.numpy()

    results = val_meta[["sample_id", "session_id", "scenario", "label_id"]].copy()
    results["float_prediction"] = float_predictions
    results["integer_prediction"] = integer_predictions
    results["predictions_match"] = float_predictions == integer_predictions

    metrics = pd.DataFrame([
        {
            "model": name,
            "accuracy": float(np.mean(actual == prediction)),
            "balanced_accuracy": balanced_accuracy_score(actual, prediction),
            "macro_f1": f1_score(
                actual, prediction, average="macro", zero_division=0
            ),
        }
        for name, prediction in (
            ("float", float_predictions),
            ("integer", integer_predictions),
        )
    ])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORT_DIR / "validation_predictions.csv", index=False)
    metrics.to_csv(REPORT_DIR / "validation_metrics.csv", index=False)

    print("Perbandingan float dan integer pada validation")
    print(metrics.round(4).to_string(index=False))
    print(f"\nPrediksi sama: {int(results['predictions_match'].sum())}/{len(results)}")
    print(f"Pool1: scale={scale_pool1:.8g}, multiplier={multiplier1}, clipped={clipped1}")
    print(f"Pool2: scale={scale_pool2:.8g}, multiplier={multiplier2}, clipped={clipped2}")
    print(f"Folder laporan: {REPORT_DIR}")


if __name__ == "__main__":
    main()