"""Run Assignment 02 stages in WSL without Jupyter.

Examples:
    python run_local.py --stage checks
    python run_local.py --stage sanity
    python run_local.py --stage architectures
    python run_local.py --stage ablations --second-seed
    python run_local.py --stage transfer
    python run_local.py --stage plots
    python run_local.py --stage final --winner arch_ResNet18_s42

Local results default to artifacts_local, separate from the Colab T4 results.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from src.data import load_cifar_and_split, make_dataset
from src.experiments import (final_test_once, inspect_architecture, log_failure,
                             plot_comparisons, record_environment, train_run,
                             vgg_parameter_comparison)
from src.models import BUILDERS, build_ablation, build_transfer


ROOT = Path(__file__).resolve().parent


def completed(run_id, output_dir):
    results = output_dir / "results.csv"
    checkpoint = output_dir / "checkpoints" / f"{run_id}.weights.h5"
    if not (results.exists() and checkpoint.exists()):
        return False
    return run_id in set(pd.read_csv(results).run_id.astype(str))


def data(output_dir):
    split = load_cifar_and_split(output_dir)
    (x_train, y_train), (x_test, y_test), train_idx, val_idx = split
    print("Train/validation/test:", len(train_idx), len(val_idx), len(x_test))
    print("Train class counts:", np.bincount(y_train[train_idx]).tolist())
    print("Validation class counts:", np.bincount(y_train[val_idx]).tolist())
    return (x_train, y_train), (x_test, y_test), train_idx, val_idx


def sanity(x_train, y_train, train_idx, val_idx, output_dir):
    values = {}
    for lr in (1e-4, 3e-4):
        keras.backend.clear_session()
        keras.utils.set_random_seed(42)
        model = build_ablation("A0")
        model.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
        train_ds = make_dataset(x_train, y_train, train_idx[:512], 64, 16,
                                training=True, seed=42)
        val_ds = make_dataset(x_train, y_train, val_idx[:512], 64, 16)
        history = model.fit(train_ds, validation_data=val_ds,
                            epochs=1, verbose=2).history
        values[str(lr)] = {key: float(v[-1]) for key, v in history.items()}
        assert np.isfinite(values[str(lr)]["loss"])
        print("Learning-rate check:", lr, values[str(lr)])
        del model
    (output_dir / "lr_sanity.json").write_text(json.dumps(values, indent=2),
                                                encoding="utf-8")


def checks(output_dir):
    vgg_actual = None
    for name, builder in BUILDERS.items():
        keras.backend.clear_session()
        keras.utils.set_random_seed(42)
        model = builder()
        result = inspect_architecture(name, model, output_dir)
        print(name, "parameters:", model.count_params(), "check:", result)
        if name == "VGG16":
            vgg_actual = model.count_params()
        del model
    comparison = vgg_parameter_comparison()
    assert vgg_actual == comparison["canonical_total_params"]
    (output_dir / "vgg_head_analysis.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8")
    print("VGG16 canonical versus GAP totals:", comparison)


def run_architectures(x_train, y_train, train_idx, val_idx, args):
    for name, builder in BUILDERS.items():
        run_id = f"arch_{name}_s42"
        if completed(run_id, args.output_dir):
            print("Already complete:", run_id)
            continue
        keras.backend.clear_session()
        keras.utils.set_random_seed(42)
        try:
            model = builder()
            row = train_run(run_id, name, model, x_train, y_train,
                            train_idx, val_idx, size=96, seed=42,
                            epochs=args.epochs, batch_size=args.batch_size,
                            caveat="Local RTX 4060 scratch run; 96x96; one seed",
                            output_dir=args.output_dir)
            print(row)
        except Exception as exc:
            log_failure(run_id, exc, args.batch_size, args.output_dir)
            print("FAILED:", run_id, repr(exc))
        finally:
            if "model" in locals():
                del model


def run_ablations(x_train, y_train, train_idx, val_idx, args):
    seeds = (42, 123) if args.second_seed else (42,)
    for seed in seeds:
        for variant in ("A0", "A1", "A2", "A3-P", "A3-R"):
            run_id = f"{variant}_s{seed}"
            if completed(run_id, args.output_dir):
                print("Already complete:", run_id)
                continue
            keras.backend.clear_session()
            keras.utils.set_random_seed(seed)
            try:
                model = build_ablation(variant)
                caveat = ("Projection shortcuts add a small parameter difference"
                          if variant == "A3-R" else
                          "Local controlled 64x64 scratch ablation")
                row = train_run(run_id, variant, model, x_train, y_train,
                                train_idx, val_idx, size=64, seed=seed,
                                epochs=args.epochs, batch_size=args.batch_size,
                                augment=variant in ("A2", "A3-P", "A3-R"),
                                caveat=caveat, output_dir=args.output_dir)
                print(row)
            except Exception as exc:
                log_failure(run_id, exc, args.batch_size, args.output_dir)
                print("FAILED:", run_id, repr(exc))
            finally:
                if "model" in locals():
                    del model


def run_transfer(x_train, y_train, train_idx, val_idx, args):
    for variant, fine_tune, lr in (("A4-F", False, 3e-4),
                                   ("A4-T", True, 1e-5)):
        run_id = f"{variant}_s42"
        frozen = args.output_dir / "checkpoints" / "A4-F_s42.weights.h5"
        if fine_tune and not frozen.exists():
            print("Skipping A4-T: A4-F checkpoint missing")
            break
        if completed(run_id, args.output_dir):
            print("Already complete:", run_id)
            continue
        keras.backend.clear_session()
        keras.utils.set_random_seed(42)
        try:
            model = build_transfer(fine_tune=fine_tune)
            if fine_tune:
                model.load_weights(str(frozen))
            row = train_run(run_id, variant, model, x_train, y_train,
                            train_idx, val_idx, size=96, seed=42,
                            epochs=args.epochs, batch_size=args.batch_size,
                            transfer=True, learning_rate=lr,
                            caveat="Local ImageNet-pretrained transfer branch",
                            output_dir=args.output_dir)
            print(row)
        except Exception as exc:
            log_failure(run_id, exc, args.batch_size, args.output_dir)
            print("FAILED:", run_id, repr(exc))
        finally:
            if "model" in locals():
                del model


def final(args):
    if not args.winner:
        raise SystemExit("Set --winner to a run ID from results.csv")
    results = pd.read_csv(args.output_dir / "results.csv")
    selected = results.loc[results.run_id == args.winner]
    if len(selected) != 1:
        raise SystemExit("Winner must match exactly one completed result row")
    row = selected.iloc[0]
    marker = args.output_dir / "selected_winner.json"
    if marker.exists():
        prior = json.loads(marker.read_text(encoding="utf-8"))
        if prior["run_id"] != args.winner:
            raise SystemExit("A different winner was already recorded")
    else:
        marker.write_text(json.dumps({
            "run_id": args.winner,
            "validation_accuracy": float(row.best_val_accuracy),
            "checkpoint": row.checkpoint}, indent=2), encoding="utf-8")
    name = row.architecture_variant
    if name in BUILDERS:
        model = BUILDERS[name]()
    elif name in ("A4-F", "A4-T"):
        model = build_transfer(fine_tune=name == "A4-T")
    else:
        model = build_ablation(name)
    (_, _), (x_test, y_test), _, _ = data(args.output_dir)
    size = int(str(row.input_size).split("x")[0])
    print(final_test_once(model, row.checkpoint, x_test, y_test,
                          size=size, batch_size=args.batch_size,
                          transfer=name in ("A4-F", "A4-T"),
                          output_dir=args.output_dir))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=("checks", "data", "sanity", "architectures",
                                 "ablations", "transfer", "plots", "final", "all"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts_local")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--second-seed", action="store_true")
    parser.add_argument("--float32", action="store_true",
                        help="Disable mixed precision if GPU training is unstable")
    parser.add_argument("--winner")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not 1 <= args.epochs <= 5:
        parser.error("epochs must be between 1 and 5")
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    if tf.config.list_physical_devices("GPU") and not args.float32:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("TensorFlow", tf.__version__, "GPU", tf.config.list_physical_devices("GPU"))
    record_environment(args.output_dir)
    if args.stage == "checks":
        checks(args.output_dir)
    elif args.stage == "plots":
        plot_comparisons(args.output_dir / "results.csv", args.output_dir)
        print(pd.read_csv(args.output_dir / "results.csv").to_string(index=False))
    elif args.stage == "final":
        final(args)
    else:
        (x_train, y_train), _, train_idx, val_idx = data(args.output_dir)
        if args.stage in ("sanity", "all"):
            sanity(x_train, y_train, train_idx, val_idx, args.output_dir)
        if args.stage == "all":
            checks(args.output_dir)
        if args.stage in ("architectures", "all"):
            run_architectures(x_train, y_train, train_idx, val_idx, args)
        if args.stage in ("ablations", "all"):
            run_ablations(x_train, y_train, train_idx, val_idx, args)
        if args.stage in ("transfer", "all"):
            run_transfer(x_train, y_train, train_idx, val_idx, args)
        if args.stage == "all":
            plot_comparisons(args.output_dir / "results.csv", args.output_dir)
    print("Stage complete:", args.stage)


if __name__ == "__main__":
    main()
