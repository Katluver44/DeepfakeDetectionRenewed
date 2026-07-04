# DeepfakeDetectionRenewed Reproduction README

This repository contains the code, notebooks, cached result artifacts, and reviewer-facing rerun paths for the MOSS@COLM submission on deepfake speech detection. The fastest way to verify the artifact is to run the demo notebook against the committed outputs and downloaded checkpoints. The more expensive path is the full regeneration notebook, which rebuilds datasets, caches, metrics, and audits from the original scripts.

## Reviewer Quick Start

Run all commands from the repository root:

```bash
cd DeepfakeDetectionRenewed
```

Create an environment using one of the provided setup scripts:

```bash
bash repro_setup_py310.sh
```

or, for the older tested Python 3.9/CUDA 12.1 stack:

```bash
bash repro_setup_py39.sh
```

Then open the demo notebook:

```bash
jupyter notebook new_demo.ipynb
```

In the paper/package this notebook may be referred to as `demo.ipynb` or `final_demo.ipynb`; in this checkout the reviewer-facing lightweight notebook is `new_demo.ipynb`. The full cold-rerun notebook is `reproducibility_full.ipynb`.

## Checkpoints

Download the submitted checkpoint artifacts from Figshare:

https://figshare.com/s/fda6530ff838a2813db9

Place the downloaded checkpoint files under:

```text
models/good_models/
```

The demo notebook expects these main checkpoints:

```text
models/good_models/robust_goat.ckpt
models/good_models/mini_goat-best-epoch=02-val-eer=0.0933.ckpt
models/good_models/mlaad_goat-best-epoch=05-val-eer=0.2795.ckpt
models/good_models/mlaad_robust_goat.ckpt
models/good_models/mlaad_robust_goat_seed42-best-epoch=05-val-eer=0.3030.ckpt
models/good_models/mlaad_robust_goat_seed1024-best-epoch=03-val-eer=0.2976.ckpt
```

`new_demo.ipynb` includes a "Checkpoint QA" section that CPU-loads these files, reports parameter counts and short hashes, and flags missing or corrupt checkpoints. `reproducibility_full.ipynb` also creates compatibility aliases in `models/` and `experiments/checkpoints/` for historical script paths.

## Data Artifacts

The code uses public datasets and repository-generated caches:

- ASVspoof 2019 LA: loaded through Hugging Face in the notebooks/scripts and cached under `data/asvspoof_2019_la`.
- MLAAD-tiny: prepared by `experiments/scripts/prepare_mlaad_tiny.py` into `experiments/data/mlaad_tiny_processed/`.
- In-the-Wild speech: optional, used by ITW transfer experiments; see the optional ITW section in `reproducibility_full.ipynb`.
- ASVspoof 2021 LA keys: optional, used by the prospective ASVspoof21 checks; the notebook expects `/tmp/keys/LA/CM/trial_metadata.txt`.

To regenerate the core processed dataset artifacts:

```bash
python experiments/scripts/prepare_mlaad_tiny.py
```

The expected MLAAD-tiny outputs are:

```text
experiments/data/mlaad_tiny_processed/splits/train.json
experiments/data/mlaad_tiny_processed/splits/val.json
experiments/data/mlaad_tiny_processed/splits/test.json
```

## Demo Notebook

Use `new_demo.ipynb` for the main review pass. It:

- validates package imports and repository paths;
- loads paper target numbers from `final_outputs2/summary_assets/key_numbers.json`;
- reruns scripts when their raw caches are available;
- otherwise validates the committed artifacts;
- checks metrics for EER, AUC, balanced accuracy, fusion deltas, and audit outputs;
- loads submitted checkpoints from `models/good_models/`;
- prints validation flags for unavailable external assets without hiding metric mismatches.

The final cell should print that all collected paper metric checks matched their targets within tolerance. If it prints validation flags, inspect the flagged missing external assets or caches; cached metric validation can still be successful.

## Full Reproduction Notebook

Use `reproducibility_full.ipynb` for a fuller rerun. Important toggles are defined in the first code cell:

```python
FORCE_REBUILD = False
ALLOW_DOWNLOADS = True
INSTALL_MISSING_PACKAGES = False
STOP_ON_FAILURE = False
RUN_HEAVY_EXPERIMENTS = True
```

Set `FORCE_REBUILD=True` for a cold regeneration. Set `RUN_HEAVY_EXPERIMENTS=False` for preflight, checkpoint, dataset, and audit checks without the longest experiment runs. The notebook writes a machine-readable report to:

```text
experiments/results/reproducibility_notebook_report.json
```

## Metrics and Result Files

Evaluation metrics and dataset references are surfaced in `new_demo.ipynb` and `reproducibility_full.ipynb`. The main committed result artifacts are in:

```text
experiments/results/
outputs/
final_outputs/
final_outputs2/
```

High-level paper numbers are summarized in:

```text
final_outputs2/summary_assets/key_numbers.json
final_outputs2/tables/
final_outputs2/report/paper_style_summary.md
```

## Training Logs

Reviewers should look for training logs in the `training_logs` folders. The main submitted training logs include:

```text
experiments/results/e_mini_goat_fusion/training_logs/
experiments/results/mlaad/training_logs/
```

Some exploratory or ablation runs store Lightning-style logs as `metrics.csv` and `hparams.yaml` under experiment-specific folders such as:

```text
experiments/results/mlaad/*/logs_seed*/
experiments/results/mlaad/ct_feature_injection/logs_seed*/
```

## Optional Training Recipes

`new_demo.ipynb` includes disabled-by-default training recipes. Set:

```python
RUN_TRAINING = True
```

to intentionally regenerate checkpoints. Leave `TRAINING_SMOKE_TEST=True` for one-batch sanity jobs where supported. Full recipes include:

```bash
python experiments/results/e_mini_goat_fusion/prepare_mini_goat_data.py
python experiments/results/e_mini_goat_fusion/train_mini_goat.py --checkpoint-path models/mini_goat.ckpt --ckpt-dir experiments/checkpoints --log-dir experiments/results/e_mini_goat_fusion/training_logs
python experiments/train_new_seed.py --seed 3 --epochs 7 --wandb 0
python experiments/train_new_seed.py --seed 7 --epochs 7 --wandb 0
python experiments/scripts/prepare_mlaad_tiny.py
python experiments/scripts/train_mlaad_regular.py --checkpoint-path experiments/checkpoints/mlaad_goat.ckpt --log-dir experiments/results/mlaad/training_logs/mlaad_goat
python experiments/scripts/train_mlaad_adversarial.py --checkpoint-path models/mlaad_robust_goat.ckpt --log-dir experiments/results/mlaad/training_logs/mlaad_robust_goat
```

## Estimated Compute and FLOP Budgets

These are order-of-magnitude estimates for reviewer planning, not profiler-certified FLOP counts. The logs show WavLM-GAT-style models with about 107M total parameters, about 12.9M trainable parameters in several MLAAD runs, batch size 20 for MLAAD training scripts, 3 second / 16 kHz audio crops, and 7 epoch defaults for the main MLAAD regular/adversarial recipes.

| Task | Expected hardware | Approximate budget |
| --- | --- | --- |
| `new_demo.ipynb` cached validation | CPU or single GPU helpful | minutes to under 1 GPU-hour; mostly loads artifacts, reads CSV/JSON/NPZ files, and reruns only scripts whose caches are present |
| `reproducibility_full.ipynb` with existing caches | single NVIDIA GPU recommended | several GPU-hours depending on which heavy cells are enabled |
| Cold dataset/cache rebuild | CPU plus network; GPU for scoring | dominated by dataset download and WavLM embedding/scoring; plan for many GB of disk and multi-hour wall time |
| One MLAAD WavLM-GAT training run | A100-class GPU recommended | roughly 7 epochs, batch size 20; logs for related runs show hundreds of batches per epoch and minute-scale epochs on A100; budget on the order of 1e17 FLOPs per checkpoint |
| Full multi-seed/audit reproduction | one or more GPUs | order of 1e18 FLOPs when regenerating all checkpoints, embeddings, and audit runs from scratch |

For exact wall-clock evidence, inspect the relevant `training_logs` folders and `experiments/results/reproducibility_notebook_report.json`.

## Expected Review Workflow

1. Download Figshare checkpoints into `models/good_models/`.
2. Build the Python environment with `repro_setup_py310.sh` or `repro_setup_py39.sh`.
3. Run `new_demo.ipynb` end to end and inspect the final validation summary.
4. Inspect `training_logs` folders for training curves and hyperparameters.
5. Use `reproducibility_full.ipynb` when a cold or near-cold regeneration is required.

