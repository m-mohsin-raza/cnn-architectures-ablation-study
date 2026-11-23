"""Create a self-contained Colab notebook from the checked source files."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(True)}


def writefile(path):
    source = (ROOT / path).read_text(encoding="utf-8")
    return code(f"%%writefile {path}\n{source}")


cells = [
    md("""# Assignment 02 — CNN architectures and controlled ablations

Run cells from top to bottom in **Google Colab with a T4 GPU**. This notebook writes its
own `src/` modules, saves every completed run under `artifacts/`, and leaves the
reserved CIFAR-10 test split untouched until the final selection cell. If a session
disconnects, download `artifacts/` before the session expires and restore it before
rerunning; existing checkpoints are the evidence of completed runs.

The four scratch architecture runs use 96×96 images; A0–A3 use 64×64; A4 is a
separate ImageNet transfer branch at 96×96. Training uses Adam, sparse categorical
cross-entropy from logits, batch 16, and a five-epoch maximum.
"""),
    code("""from pathlib import Path
Path('src').mkdir(exist_ok=True)
Path('src/__init__.py').touch()
Path('artifacts').mkdir(exist_ok=True)
import tensorflow as tf
USE_MIXED_PRECISION = True  # Set False if GPU training becomes unstable.
if USE_MIXED_PRECISION and tf.config.list_physical_devices('GPU'):
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
print('TensorFlow:', tf.__version__)
print('GPU:', tf.config.list_physical_devices('GPU'))
print('Precision:', tf.keras.mixed_precision.global_policy())
"""),
    writefile("src/models.py"),
    writefile("src/data.py"),
    writefile("src/experiments.py"),
    md("""## Step 1 — Data split and environment

One fixed stratified 12,000/3,000 split is saved. The 10,000-image original test
partition is loaded but not evaluated here. Resizing happens in the `tf.data`
pipeline rather than making permanent enlarged copies.
"""),
    code("""import json
import numpy as np
from tensorflow import keras
from src.data import load_cifar_and_split, make_dataset
from src.models import BUILDERS, build_ablation, build_transfer, build_vgg16
from src.experiments import (record_environment, inspect_architecture, train_run,
                             log_failure, plot_comparisons, final_test_once,
                             vgg_parameter_comparison)
(x_train, y_train), (x_test, y_test), train_idx, val_idx = load_cifar_and_split()
environment = record_environment()
print('Split sizes:', len(train_idx), len(val_idx), len(x_test))
print('Train class counts:', np.bincount(y_train[train_idx]))
print('Validation class counts:', np.bincount(y_train[val_idx]))
print('Environment saved to artifacts/environment.json')
"""),
    md("""## Step 2 — Brief learning-rate sanity check

Use a small training/validation slice to check that the common 3e-4 Adam setting
produces finite loss before the main runs. These are diagnostics, not benchmark rows.
"""),
    code("""sanity = {}
for lr in (1e-4, 3e-4):
    keras.backend.clear_session()
    keras.utils.set_random_seed(42)
    model = build_ablation('A0')
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy')])
    small_train = make_dataset(x_train, y_train, train_idx[:512], 64, 16,
                               training=True, seed=42)
    small_val = make_dataset(x_train, y_train, val_idx[:512], 64, 16)
    history = model.fit(small_train, validation_data=small_val,
                        epochs=1, verbose=2).history
    sanity[str(lr)] = {k: float(v[-1]) for k, v in history.items()}
    assert np.isfinite(sanity[str(lr)]['loss'])
Path('artifacts/lr_sanity.json').write_text(json.dumps(sanity, indent=2))
print(sanity)
"""),
    md("""## Step 3 — Build and verify four architectures

For each model, save `model.summary()`, a diagram where Graphviz is available,
a manual parameter calculation, and proof of a 2×96×96×3 to 2×10 logits pass.

**Adaptations:** AlexNet uses same-padded convolutions and pooling for 96×96,
with LRN and grouped convolutions omitted; VGG16 retains all 13 convolutions and
two 4096-unit FC layers; GoogLeNet retains its stem and nine Inception modules
but omits auxiliary heads; ResNet-18 retains its 2-2-2-2 basic-block stages.
All output heads have 10 logits rather than ImageNet classes.
"""),
    code("""architecture_checks = {}
vgg_actual = None
for name, builder in BUILDERS.items():
    keras.backend.clear_session()
    keras.utils.set_random_seed(42)
    model = builder()
    architecture_checks[name] = inspect_architecture(name, model)
    print(name, model.count_params(), architecture_checks[name])
    if name == 'VGG16':
        vgg_actual = model.count_params()
    del model
vgg_comparison = vgg_parameter_comparison()
assert vgg_actual == vgg_comparison['canonical_total_params']
print('VGG16 canonical versus GAP totals:', vgg_comparison)
Path('artifacts/vgg_head_analysis.json').write_text(
    json.dumps(vgg_comparison, indent=2))
"""),
    md("""## Step 4 — Four scratch architecture runs (96×96)

Each run uses the identical fixed split, initial learning rate 3e-4, batch 16,
five epochs, and best-validation-accuracy checkpoint rule. Run this cell once;
it may take considerable time. A failed run is logged instead of silently dropped.
"""),
    code("""ARCH_EPOCHS = 5
for name, builder in BUILDERS.items():
    run_id = f'arch_{name}_s42'
    if Path(f'artifacts/checkpoints/{run_id}.weights.h5').exists() and \
       Path('artifacts/results.csv').exists() and \
       run_id in Path('artifacts/results.csv').read_text():
        print('Already complete:', run_id)
        continue
    keras.backend.clear_session()
    keras.utils.set_random_seed(42)
    try:
        model = builder()
        row = train_run(run_id, name, model, x_train, y_train, train_idx, val_idx,
                        size=96, epochs=ARCH_EPOCHS, batch_size=16,
                        caveat='Scratch model; 96x96 upscaled CIFAR-10; one seed')
        print(row)
    except Exception as exc:
        log_failure(run_id, exc, 16)
        print('FAILED:', run_id, repr(exc))
    finally:
        if 'model' in locals():
            del model
"""),
    md("""## Step 5 — Controlled A0–A3 ablations (64×64)

The first pass uses seed 42. Set `RUN_SECOND_SEED=True` to repeat A0, A1, A2,
A3-P, and A3-R with seed 123 if quota permits. A3-R has projection shortcuts;
their extra parameters must be reported as a caveat against A3-P.
"""),
    code("""RUN_SECOND_SEED = False
ABLATION_EPOCHS = 5
for seed in ((42, 123) if RUN_SECOND_SEED else (42,)):
    for variant in ('A0', 'A1', 'A2', 'A3-P', 'A3-R'):
        run_id = f'{variant}_s{seed}'
        if Path(f'artifacts/checkpoints/{run_id}.weights.h5').exists() and \
           Path('artifacts/results.csv').exists() and \
           run_id in Path('artifacts/results.csv').read_text():
            print('Already complete:', run_id)
            continue
        keras.backend.clear_session()
        keras.utils.set_random_seed(seed)
        try:
            model = build_ablation(variant)
            caveat = 'Projection shortcuts add a small parameter difference' \
                if variant == 'A3-R' else 'Controlled 64x64 scratch ablation'
            row = train_run(run_id, variant, model, x_train, y_train,
                            train_idx, val_idx, size=64, seed=seed,
                            epochs=ABLATION_EPOCHS, batch_size=16,
                            augment=variant in ('A2', 'A3-P', 'A3-R'),
                            caveat=caveat)
            print(row)
        except Exception as exc:
            log_failure(run_id, exc, 16)
            print('FAILED:', run_id, repr(exc))
        finally:
            if 'model' in locals():
                del model
"""),
    md("""## Step 6 — Separate A4 transfer branch

A4-F uses an ImageNet-pretrained frozen ResNet50 with original 0–255 scale
passed through `resnet50.preprocess_input`. A4-T loads the A4-F checkpoint,
unfreezes selected high layers while keeping BatchNorm in inference mode,
recompiles at 1e-5, and trains again. These are not one-factor extensions of A3.
"""),
    code("""for variant, fine_tune, lr in [('A4-F', False, 3e-4),
                                     ('A4-T', True, 1e-5)]:
    run_id = f'{variant}_s42'
    frozen_checkpoint = Path('artifacts/checkpoints/A4-F_s42.weights.h5')
    if fine_tune and not frozen_checkpoint.exists():
        print('Skipping A4-T: A4-F checkpoint is missing')
        break
    if Path(f'artifacts/checkpoints/{run_id}.weights.h5').exists() and \
       Path('artifacts/results.csv').exists() and \
       run_id in Path('artifacts/results.csv').read_text():
        print('Already complete:', run_id)
        continue
    keras.backend.clear_session()
    keras.utils.set_random_seed(42)
    try:
        model = build_transfer(fine_tune=fine_tune)
        if fine_tune:
            model.load_weights(str(frozen_checkpoint))
        row = train_run(run_id, variant, model, x_train, y_train,
                        train_idx, val_idx, size=96, epochs=5, batch_size=16,
                        transfer=True, learning_rate=lr,
                        caveat='Separate ImageNet-pretrained transfer branch')
        print(row)
    except Exception as exc:
        log_failure(run_id, exc, 16)
        print('FAILED:', run_id, repr(exc))
    finally:
        if 'model' in locals():
            del model
"""),
    md("""## Step 7 — Review validation evidence and choose final model

Generate the comparison figures, inspect `artifacts/results.csv`, and record
the winning run ID **before** the reserved test split is evaluated. Do not use
test performance to pick a model.
"""),
    code("""import pandas as pd
results = pd.read_csv('artifacts/results.csv')
plot_comparisons()
display(results.sort_values('best_val_accuracy', ascending=False))
print('Figures are in artifacts/figures/')
"""),
    md("""## Step 8 — Final test, exactly once

After reviewing validation results, set `WINNER_RUN_ID` to one run ID from the
table and execute this cell once. It records the choice before test prediction
and refuses to repeat if a final result already exists.
"""),
    code("""WINNER_RUN_ID = ''  # e.g. 'arch_ResNet18_s42'; fill after reviewing validation results
if not WINNER_RUN_ID:
    print('Set WINNER_RUN_ID after reviewing validation results; test not run.')
else:
    selected = results.loc[results.run_id == WINNER_RUN_ID]
    assert len(selected) == 1, 'WINNER_RUN_ID must match exactly one result row'
    row = selected.iloc[0]
    selection_path = Path('artifacts/selected_winner.json')
    if not selection_path.exists():
        selection_path.write_text(json.dumps({
            'run_id': WINNER_RUN_ID,
            'validation_accuracy': float(row.best_val_accuracy),
            'checkpoint': row.checkpoint}, indent=2))
    else:
        assert json.loads(selection_path.read_text())['run_id'] == WINNER_RUN_ID
    keras.backend.clear_session()
    name = row.architecture_variant
    if name in BUILDERS:
        model = BUILDERS[name]()
    elif name in ('A4-F', 'A4-T'):
        model = build_transfer(fine_tune=(name == 'A4-T'))
    else:
        model = build_ablation(name)
    size = int(str(row.input_size).split('x')[0])
    test_result = final_test_once(model, row.checkpoint, x_test, y_test,
                                  size=size, transfer=name in ('A4-F', 'A4-T'))
    print(test_result)
"""),
    md("""## Step 9 — Save submission evidence

Download the `artifacts/` directory before ending the Colab session. Include
`results.csv`, figures, summaries, logs, environment, split indices, notebook,
and the report in the GitHub repository. Never replace missing results with
numbers from papers or other experiments.
"""),
    code("""from pathlib import Path
import shutil
import zipfile

artifact_dir = Path('/content/artifacts') if Path('/content/artifacts').exists() else Path('artifacts')
artifact_files = sorted(p for p in artifact_dir.rglob('*') if p.is_file())
print('Artifact folder:', artifact_dir.resolve())
print('Files found:', len(artifact_files))
for path in artifact_files[:30]:
    print(' ', path.relative_to(artifact_dir), f'({path.stat().st_size:,} bytes)')
if not artifact_files:
    raise RuntimeError('The current Colab runtime has no saved artifacts. Check Files on the left. Notebook outputs can remain visible after a runtime reset, but the temporary files do not. Restore a previous artifact ZIP if you have one; do not download this empty folder.')

archive = shutil.make_archive('assignment02_artifacts', 'zip', artifact_dir)
with zipfile.ZipFile(archive) as saved_zip:
    assert saved_zip.namelist(), 'Export ZIP unexpectedly contains no files'
print('Download:', archive, f'({Path(archive).stat().st_size:,} bytes)')
try:
    from google.colab import files
    files.download(archive)
except ImportError:
    pass  # In local Jupyter, the zip remains in the current folder.
"""),
]

for index, cell in enumerate(cells):
    cell["id"] = f"cell-{index:03d}"

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

target = ROOT / "cnn_assignment.ipynb"
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1),
                  encoding="utf-8")
print(target)
