"""Training, measurements, checkpoints, and plots for Assignment 02."""

import csv
import json
import platform
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.data import make_dataset


RESULT_COLUMNS = [
    "run_id", "architecture_variant", "input_size", "seed", "trainable_params",
    "total_params", "epochs", "best_val_accuracy", "macro_f1",
    "final_train_accuracy", "generalization_gap", "training_minutes",
    "peak_gpu_memory_mb", "key_change_caveat", "checkpoint",
]


def vgg_parameter_comparison():
    """Analytic VGG16 parameter totals for a 96x96 input and 10 classes."""
    channels = [3, 64, 64, 128, 128, 256, 256, 256,
                512, 512, 512, 512, 512, 512]
    conv_base = sum((3 * 3 * cin + 1) * cout
                    for cin, cout in zip(channels[:-1], channels[1:]))
    canonical_head = ((3 * 3 * 512 + 1) * 4096
                      + (4096 + 1) * 4096 + (4096 + 1) * 10)
    gap_head = (512 + 1) * 10
    return {
        "convolutional_base_params": conv_base,
        "canonical_classifier_params": canonical_head,
        "gap_classifier_params": gap_head,
        "canonical_total_params": conv_base + canonical_head,
        "gap_total_params": conv_base + gap_head,
        "saved_parameters": canonical_head - gap_head,
    }


def record_environment(output_dir="artifacts"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        gpu_info = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                                  timeout=15).stdout
    except (OSError, subprocess.TimeoutExpired):
        gpu_info = "nvidia-smi unavailable"
    try:
        commit_hash = subprocess.run(["git", "rev-parse", "HEAD"],
                                     capture_output=True, text=True,
                                     timeout=5).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        commit_hash = None
    info = {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
        "nvidia_smi": gpu_info,
        "code_commit_hash": commit_hash,
    }
    (output_dir / "environment.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8")
    return info


def _manual_parameter_check(model):
    layer = next(layer for layer in model.layers
                 if isinstance(layer, keras.layers.Conv2D))
    kh, kw, cin, cout = [int(v) for v in layer.kernel.shape]
    manual = kh * kw * cin * cout + (cout if layer.use_bias else 0)
    assert manual == layer.count_params()
    return {"layer": layer.name, "formula": f"{kh}*{kw}*{cin}*{cout}"
            + (f"+{cout}" if layer.use_bias else ""),
            "manual": manual, "keras": layer.count_params()}


def _fallback_diagram(name, path):
    stages = {
        "AlexNet": ["96x96 RGB", "11x11 conv 96 + pool", "5x5 conv 256 + pool",
                    "3x3 conv 384", "3x3 conv 384", "3x3 conv 256 + pool",
                    "FC 4096", "FC 4096", "10 logits"],
        "VGG16": ["96x96 RGB", "2 x 3x3 conv 64 + pool",
                  "2 x 3x3 conv 128 + pool", "3 x 3x3 conv 256 + pool",
                  "3 x 3x3 conv 512 + pool", "3 x 3x3 conv 512 + pool",
                  "FC 4096", "FC 4096", "10 logits"],
        "GoogLeNet": ["96x96 RGB", "7x7 / 1x1 / 3x3 stem",
                      "Inception 3a, 3b", "Inception 4a-4e",
                      "Inception 5a, 5b", "GAP", "10 logits"],
        "ResNet18": ["96x96 RGB", "7x7 stem + pool",
                     "2 basic blocks, 64", "2 basic blocks, 128",
                     "2 basic blocks, 256", "2 basic blocks, 512",
                     "Each block: 3x3 -> 3x3 + shortcut", "GAP", "10 logits"],
    }[name]
    fig, ax = plt.subplots(figsize=(7, max(5, len(stages) * 0.75)))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(stages) - 0.5)
    ax.axis("off")
    for i, label in enumerate(stages):
        y = len(stages) - i - 1
        ax.text(0.5, y, label, ha="center", va="center",
                bbox={"boxstyle": "round,pad=0.5", "fc": "#e8f0ff",
                      "ec": "#315a99"})
        if i:
            ax.annotate("", xy=(0.5, y + 0.27), xytext=(0.5, y + 0.73),
                        arrowprops={"arrowstyle": "->", "color": "#315a99"})
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def inspect_architecture(name, model, output_dir="artifacts"):
    output_dir = Path(output_dir) / "architectures"
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    model.summary(print_fn=lambda line, **kwargs: lines.append(line))
    (output_dir / f"{name}_summary.txt").write_text(
        "\n".join(lines), encoding="utf-8")
    input_shape = tuple(int(v) for v in model.input_shape[1:])
    logits = model(tf.zeros((2,) + input_shape), training=False)
    assert tuple(logits.shape) == (2, 10), (name, logits.shape)
    assert bool(tf.reduce_all(tf.math.is_finite(logits)))
    checked = _manual_parameter_check(model)
    (output_dir / f"{name}_check.json").write_text(
        json.dumps({"forward_shape": [2, 10], "parameter_check": checked},
                   indent=2), encoding="utf-8")
    try:
        keras.utils.plot_model(model, to_file=str(output_dir / f"{name}.png"),
                               show_shapes=True, show_layer_names=True,
                               expand_nested=False, dpi=80)
        diagram = str(output_dir / f"{name}.png")
    except Exception as exc:
        diagram = str(output_dir / f"{name}.png")
        _fallback_diagram(name, diagram)
        print(f"Graphviz diagram unavailable ({exc}); saved stage diagram instead")
    return {"forward_shape": (2, 10), "parameter_check": checked,
            "diagram": diagram}


def _memory_before():
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception:
        pass


def _peak_memory_mb():
    try:
        return round(tf.config.experimental.get_memory_info("GPU:0")["peak"]
                     / 1024**2, 1)
    except Exception:
        return None


def _plot_history(history, run_id, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in zip(axes, ("accuracy", "loss"),
                              ("Accuracy", "Loss")):
        ax.plot(history[key], label="train")
        ax.plot(history[f"val_{key}"], label="validation")
        ax.set(xlabel="Epoch", title=f"{run_id}: {title}")
        ax.legend()
    fig.tight_layout()
    path = output_dir / "figures" / f"{run_id}_curves.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _append_result(row, output_dir):
    path = output_dir / "results.csv"
    prior = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=RESULT_COLUMNS)
    prior = prior[prior.run_id != row["run_id"]]
    pd.concat([prior, pd.DataFrame([row])], ignore_index=True).to_csv(path,
                                                                      index=False)


def train_run(run_id, name, model, x_train, y_train, train_idx, val_idx,
              *, size, seed=42, epochs=5, batch_size=16, augment=False,
              transfer=False, learning_rate=3e-4, caveat="",
              output_dir="artifacts"):
    """Fit one run; select by validation accuracy and retain every measured result."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(seed)
    train_ds = make_dataset(x_train, y_train, train_idx, size, batch_size,
                            training=True, augment=augment, transfer=transfer,
                            seed=seed)
    val_ds = make_dataset(x_train, y_train, val_idx, size, batch_size,
                          transfer=transfer)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    checkpoint = output_dir / "checkpoints" / f"{run_id}.weights.h5"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    callback = keras.callbacks.ModelCheckpoint(
        str(checkpoint), monitor="val_accuracy", mode="max",
        save_best_only=True, save_weights_only=True)
    _memory_before()
    start = time.perf_counter()
    fit = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                    callbacks=[callback], verbose=2)
    minutes = (time.perf_counter() - start) / 60
    history = {k: [float(v) for v in values]
               for k, values in fit.history.items()}
    history_path = output_dir / "histories" / f"{run_id}.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    _plot_history(history, run_id, output_dir)
    model.load_weights(str(checkpoint))
    y_pred = np.argmax(model.predict(val_ds, verbose=0), axis=1)
    y_true = y_train[val_idx]
    best_val = max(history["val_accuracy"])
    assert abs(accuracy_score(y_true, y_pred) - best_val) < 0.005
    total = model.count_params()
    trainable = sum(int(np.prod(v.shape)) for v in model.trainable_weights)
    row = {
        "run_id": run_id, "architecture_variant": name,
        "input_size": f"{size}x{size}", "seed": seed,
        "trainable_params": trainable, "total_params": total,
        "epochs": len(history["loss"]),
        "best_val_accuracy": round(best_val, 6),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 6),
        "final_train_accuracy": round(history["accuracy"][-1], 6),
        "generalization_gap": round(history["accuracy"][-1]
                                    - history["val_accuracy"][-1], 6),
        "training_minutes": round(minutes, 3),
        "peak_gpu_memory_mb": _peak_memory_mb(),
        "key_change_caveat": caveat,
        "checkpoint": str(checkpoint),
    }
    _append_result(row, output_dir)
    return row


def log_failure(run_id, exc, batch_size, output_dir="artifacts"):
    path = Path(output_dir) / "failed_runs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"run_id": run_id, "batch_size": batch_size,
                                 "error": repr(exc), "time": time.time()}) + "\n")


def plot_comparisons(results_path="artifacts/results.csv", output_dir="artifacts"):
    frame = pd.read_csv(results_path)
    output_dir = Path(output_dir) / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    for title, subset, filename in [
        ("Four architectures", frame[frame.run_id.str.startswith("arch_")],
         "architecture_comparison.png"),
        ("Ablation and transfer", frame[frame.run_id.str.startswith("A")],
         "ablation_comparison.png"),
    ]:
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(subset.run_id, subset.best_val_accuracy)
        ax.set(ylabel="Best validation accuracy", title=title, ylim=(0, 1))
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)

    architecture_runs = frame[frame.run_id.str.startswith("arch_")]
    if not architecture_runs.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for run_id in architecture_runs.run_id:
            history_path = output_dir.parent / "histories" / f"{run_id}.json"
            if not history_path.exists():
                continue
            history = json.loads(history_path.read_text(encoding="utf-8"))
            for ax, metric in zip(axes, ("accuracy", "loss")):
                ax.plot(history[metric], label=f"{run_id} train")
                ax.plot(history[f"val_{metric}"], linestyle="--",
                        label=f"{run_id} val")
        for ax, metric in zip(axes, ("accuracy", "loss")):
            ax.set(xlabel="Epoch", title=f"Four architectures: {metric}")
            ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(output_dir / "four_architecture_curves.png", dpi=160)
        plt.close(fig)


def final_test_once(model, checkpoint, x_test, y_test, *, size,
                    batch_size=16, transfer=False, output_dir="artifacts"):
    output_dir = Path(output_dir)
    marker = output_dir / "final_test_result.json"
    if marker.exists():
        raise RuntimeError("Final test has already been recorded; do not repeat it.")
    model.load_weights(str(checkpoint))
    test_ds = make_dataset(x_test, y_test, None, size, batch_size,
                           transfer=transfer)
    prediction = np.argmax(model.predict(test_ds, verbose=1), axis=1)
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog",
               "frog", "horse", "ship", "truck"]
    matrix = confusion_matrix(y_test, prediction)
    top_confusions = sorted(
        ({"true": classes[i], "predicted": classes[j], "count": int(matrix[i, j])}
         for i in range(10) for j in range(10) if i != j),
        key=lambda item: item["count"], reverse=True)[:5]
    result = {"checkpoint": str(checkpoint),
              "test_accuracy": float(accuracy_score(y_test, prediction)),
              "test_macro_f1": float(f1_score(y_test, prediction,
                                               average="macro")),
              "class_names": classes,
              "confusion_matrix": matrix.tolist(),
              "top_confusions": top_confusions}
    marker.write_text(json.dumps(result, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(10), classes, rotation=45, ha="right")
    ax.set_yticks(range(10), classes)
    ax.set(xlabel="Predicted class", ylabel="True class", title="Final test confusion matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "final_confusion_matrix.png", dpi=160)
    plt.close(fig)
    return result
