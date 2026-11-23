# CNN Assignment 02 — fast execution guide

The self-contained `cnn_assignment.ipynb` is the main deliverable. It writes the
source modules into Colab when run, so it does not require a GitHub clone first.

## Measured results and repository files

The recorded T4 run produced 11 five-epoch results in the root `results.csv`.
The best validation model was A4-T at 86.03% accuracy. Its single reserved
test evaluation recorded 84.97% accuracy and 84.99% macro-F1 in
`artifacts/final_test_result.json`; the selection was saved first in
`artifacts/selected_winner.json`. The final test used a separate Colab CPU
runtime, recorded in `artifacts/final_test_environment.json`. All scratch and
ablation results are single-seed findings; seed 123 was not run.

The tracked `artifacts/` directory contains metrics, split indices, model
summaries, checks, diagrams, and plots. Model weight files and the full ZIP are
kept locally and excluded from Git because of their size. Recreate weights by
running the notebook in Colab. The 6–9-page `report.pdf` and an executed copy
of the notebook still need to be added before final submission.

## Fast local first step

If WSL's `python3` cannot import TensorFlow, create a separate environment for
this assignment (using Python 3.10–3.13):

On Ubuntu 24.04 with Python 3.12, install the virtual-environment support once
if `python3 -m venv` reports that `ensurepip` is unavailable:

```bash
sudo apt update
sudo apt install -y python3.12-venv
```

```bash
python3 -m venv ~/.venvs/cnn-assignment02
source ~/.venvs/cnn-assignment02/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-wsl-gpu.txt
```

TensorFlow 2.21.0 may report `Cannot dlopen some GPU libraries` in WSL even
after the GPU packages install. If that happens, activate the venv and source
the per-session cuSOLVER path helper before starting Python:

```bash
source ~/.venvs/cnn-assignment02/bin/activate
source ./enable_wsl_gpu.sh
python -c 'import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))'
```

The GPU check should show an NVIDIA device. The helper changes only the
current shell environment; source it again in a new terminal before running
local GPU jobs.

Then run `python local_smoke_check.py`
from this project directory. It builds all four models, checks their `2×10`
logits, key layer counts and a parameter formula, then takes one optimizer step
on synthetic data. It uses the GPU when TensorFlow detects it. It does not
download CIFAR-10 or run a long training job. If the environment lacks
GPU access after installation, inspect the GPU verification output before
running long training jobs.

The local command-line runner shares the same model, data, and measurement
code as the Colab notebook. It stores results in `artifacts_local/`, so local
results cannot be confused with final Colab T4 results:

The data loader primes Keras's normal CIFAR-10 cache from Keras's
Google-hosted mirror, verifying the official SHA-256 hash before use. This
avoids the slower Toronto download used by some installed Keras versions.

For the quickest assignment path, use the local smoke check, then `checks`,
`data`, and `sanity` locally. Run the full architecture, ablation, and transfer
benchmarks in Colab to match the specified T4 protocol. The remaining local
stages are available if you choose to do additional laptop runs.

```bash
python run_local.py --stage checks
python run_local.py --stage data
python run_local.py --stage sanity
python run_local.py --stage architectures
python run_local.py --stage ablations
python run_local.py --stage transfer
python run_local.py --stage plots
```

Each training stage resumes completed runs. The `--second-seed` option adds
seed 123 to the ablation stage. `--epochs` defaults to five and `--batch-size`
defaults to 16. To run a faster local trial, use `--epochs 1`; do not report
that trial as the final five-epoch comparison. Use `--float32` if mixed precision
causes instability. The final test is an explicit
separate command after selecting a winner from validation results. Use this
local final command only if the local runs are the official submission runs;
otherwise evaluate the final model once in Colab:

```bash
python run_local.py --stage final --winner arch_ResNet18_s42
```

Replace the example winner with the selected run ID. The final test command
records the choice before evaluating the original CIFAR-10 test split and
refuses to repeat it.

## Run in the required environment

1. Upload `cnn_assignment.ipynb` to Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run notebook Steps 1–3. Confirm the split, learning-rate check, four
   architecture summaries, `2×10` forward passes, and diagrams.
4. Run Step 4 for the four 96×96 architecture benchmarks. It saves each completed
   run separately and skips it if the cell is rerun.
5. Run Step 5 for A0, A1, A2, A3-P, and A3-R at 64×64. If quota permits, set
   `RUN_SECOND_SEED=True` and rerun for seed 123.
6. Run Step 6 for the separate ImageNet transfer branch A4-F and A4-T.
7. Run Step 7 and review validation results. Set `WINNER_RUN_ID` in Step 8 and
   run it once to evaluate the reserved test set.
8. Run Step 9 and download the `assignment02_artifacts.zip` archive. Keep it
   with the executed notebook for the report and GitHub submission.

The scratch runs use Adam at `3e-4`, sparse categorical cross-entropy with
`from_logits=True`, batch 16, a five-epoch maximum, and best-validation-accuracy
checkpoint selection. The A4-T fine-tuning run uses `1e-5`. The full results
table is created only from actual completed runs; failed attempts go into
`failed_runs.jsonl`.

Validation macro-F1 is measured from the best-validation-accuracy checkpoint.
The reported generalization gap is final-epoch training accuracy minus
final-epoch validation accuracy; both terms therefore come from the same epoch.

## Development on the laptop

The laptop's RTX 4060 can be used to develop and debug the notebook. On Windows,
recent TensorFlow GPU use requires WSL2. The assignment specifies a Colab T4
protocol for the reported benchmark, so the notebook defaults to that platform.

## Files produced by the notebook

- `artifacts/split_indices.npz`: exact training and validation indices.
- `artifacts/environment.json`: TensorFlow/Keras versions and `nvidia-smi` output.
- `artifacts/architectures/`: summaries, diagrams, and forward/parameter checks.
- `artifacts/checkpoints/`: best-validation model weights.
- `artifacts/histories/`: per-epoch training and validation metrics.
- `artifacts/figures/`: learning curves, ablation plot, and final confusion matrix.
- `artifacts/results.csv`: measured experiment table.
- `artifacts/failed_runs.jsonl`: any failed attempts and attempted batch sizes.
- `artifacts/selected_winner.json` and `final_test_result.json`: final choice
  recorded before the one permitted test evaluation.

## Report still to prepare

The report should contain Part I calculations, fidelity/deviation table, data
protocol, architecture and ablation comparisons, error analysis, compute budget,
limitations, conclusions, and citations to the architecture papers and lectures.
Do not fill the results table until training has produced measured results.
