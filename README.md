# Reproducing the Deepfake-Speech Detection Results

This zip package lets you **re-check the numbers in our paper** by running a single
Jupyter notebook: [`final_demo.ipynb`](final_demo.ipynb).

1. set up the software environment (copy-paste a few commands),
2. put the provided model files in the right folder,
3. open the notebook and click **“Run All”**,
4. read the summary table it prints at the end.

If the last table says **“All … metric checks matched targets within tolerance,”** the
results reproduced. That's the whole goal.

## 0. What you need before starting

| Requirement | Details |
|---|---|
| **A computer with an NVIDIA GPU** | Strongly recommended. The models use a GPU; on a CPU-only machine the notebook still runs but some steps are very slow. A cloud GPU (e.g. an “L4” or “A100” instance) works great. |
| **~20 GB of free disk space** | For the environment, model files, and cached data. |
| **Python 3.12** | The programming language runtime. Check with `python --version`. If you don't have it, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and run `conda create -n repro python=3.12 && conda activate repro`. |
| **The model files (“checkpoints”)** | Provided separately — see [Step 3](#3-download-the-model-files-checkpoints). |
| **(Optional) A free Hugging Face account** | Only if a dataset download asks you to log in. See [Step 4](#4-optional-hugging-face-login). |

**What is a “terminal”?** It's a text window where you type commands.
- **Mac:** open the app called *Terminal*.
- **Windows:** open *Anaconda Prompt* (comes with Miniconda) or *PowerShell*.
- **Linux / cloud server:** you're probably already in one.

---

## 1. Unzip the artifact package

Unzip the submitted artifact package, then move into its folder in the terminal:

```bash
cd DeepfakeDetectionRenewed
```

Everything below is run from **inside this folder**.

---

## 2. Set up the software environment

You only do this **once**. Copy-paste these two commands:

```bash
# 1) Install PyTorch built for CUDA 12.9 (the deep-learning engine)
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129

# 2) Install everything else the notebook needs
pip install -r requirements.txt
```

- **No NVIDIA GPU, or a different CUDA version?** Replace the first command's
  `--index-url ...` with the matching one from
  <https://pytorch.org/get-started/locally/> (keep version `2.8.0`).
- The notebook also has an **“Environment setup” cell at the very top** that runs these
  installs for you automatically — so if you skip this step, just run that first cell and
  it will fix the environment. It prints `All core imports OK.` when everything is ready.

---

## 3. Download the model files (“checkpoints”)

The trained models are large files provided by the authors (they **cannot be re-trained**
by reviewers — that needs the full dataset and days of GPU time). Download them from Figshare:

**https://figshare.com/s/fda6530ff838a2813db9**

Then put **all of these files** into the folder `models/good_models/`:

```text
models/good_models/robust_goat.ckpt
models/good_models/robust_goat_seed3.ckpt
models/good_models/mini_goat-best-epoch=02-val-eer=0.0933.ckpt
models/good_models/mlaad_goat-best-epoch=05-val-eer=0.2795.ckpt
models/good_models/mlaad_robust_goat.ckpt
models/good_models/mlaad_robust_goat_seed42-best-epoch=05-val-eer=0.3030.ckpt
models/good_models/mlaad_robust_goat_seed1024-best-epoch=03-val-eer=0.2976.ckpt
```

Don't worry about memorizing these — the notebook has a **“Checkpoint”** cell that checks
the folder and **prints the exact name of any file that is missing**, so you'll know if one
is in the wrong place.

---

## 4. (Optional) Hugging Face login

Some datasets are downloaded automatically from Hugging Face. If a download step fails with
a “not authorized” message, make a free account at <https://huggingface.co>, create an
access token (Settings → Access Tokens), and run:

```bash
export HF_TOKEN=hf_your_token_here      # Mac/Linux
# On Windows PowerShell:  $env:HF_TOKEN = "hf_your_token_here"
```

Then re-run the notebook cell that failed. (Most of the core results don't need this.)

---

## 5. Run the notebook

Start Jupyter:

```bash
jupyter notebook final_demo.ipynb
```

This opens a page in your web browser. Then, in the menu at the top:

**Run → Run All Cells**

Now wait. The notebook runs each step from top to bottom. Cells that are already computed
are skipped quickly; the heavy steps (loading models, computing features) take longer. A
full run typically takes **tens of minutes to a couple of hours** depending on your GPU.

You'll see progress messages under each cell. A ▶️ number like `[5]` next to a cell means it
finished; `[*]` means it's still working.

---

## 6. How to know it worked

Scroll to the bottom section, **“Final Reproducibility Report.”** It prints:

1. **A table of every step** and whether it ran, was skipped (already cached), or failed.
2. **A metric table** comparing our re-computed numbers to the paper's targets. Each row
   says `ok = True` when it matches.

✅ **Success looks like:** the final line prints
**`All extracted paper metric checks matched targets within tolerance.`**

The key numbers you should see match (within a tiny tolerance):

| What it measures | Paper target |
|---|---|
| mini_goat detector-alone error rate (EER) | 0.14375 |
| mini_goat fused error rate | 0.11375 |
| AASIST zero-shot MLAAD error rate | 0.375959 |
| AASIST zero-shot MLAAD fused error rate | 0.116490 |
| AASIST fine-tuned test error rate | 0.200451 |
| AASIST fine-tuned “sd_along” correlation | 0.349498 |
| ASVspoof-2021 WavLM P3 correlation / p-value | 0.598901 / 0.030554 |

If a row is flagged as a failure, look at the step table above it — usually it means an
**optional dataset** (below) wasn't downloaded. That does not affect the core results.

---

## 7. Optional extras (you can skip these)

Two experiments need extra public datasets that are **not required** for the main results.
They live in clearly-marked cells at the **bottom** of the notebook:

- **In-the-Wild** (adds the `I5`/`I6`/`audit4` rows). The cell downloads the
  `mueller91/In-The-Wild` dataset from Hugging Face automatically (you may need the token
  from Step 4).
- **ASVspoof 2021** (adds the `J4`/`J5`/`audit3` rows). This one needs an official “keys”
  file placed at `/tmp/keys/LA/CM/trial_metadata.txt`; the cell explains exactly where to
  download it. Without it, the cell simply skips itself.

Run these only if you want to reproduce those specific extra rows.

---

## 8. What's in this zip package

| Path | What it is |
|---|---|
| `final_demo.ipynb` | **The notebook you run.** Everything starts here. |
| `requirements.txt` | The exact list of software versions. |
| `models/good_models/` | Where you put the downloaded model files (Step 3). |
| `experiments/scripts/` | The analysis programs the notebook runs. |
| `experiments/axis_audits/` | Independent “audit” checks of each scientific claim. |
| `experiments/results/` | Saved result files the notebook validates against. |
| `experiments/data/`, `data/`, `outputs/` | Prepared datasets and cached computations. |
| `baselines/aasist/` | The official AASIST baseline model (auto-downloaded if missing). |
| `phoneme_GAT/`, `callbacks.py`, `loader.py` | The detector model code. |

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` / an import fails | Re-run the **Environment setup** cell at the top of the notebook, or re-run the two `pip install` commands in [Step 2](#2-set-up-the-software-environment). |
| A cell about **checkpoints** says files are “MISSING” | Make sure all 7 files from [Step 3](#3-download-the-model-files-checkpoints) are in `models/good_models/` with their exact names. |
| A **dataset download** fails with an authorization error | Do [Step 4](#4-optional-hugging-face-login) (Hugging Face token), then re-run that cell. |
| `torchaudio` complains about **CUDA versions** | Your GPU's CUDA doesn't match. Reinstall torch/torchaudio using the correct `--index-url` from <https://pytorch.org/get-started/locally/>. |
| It's **very slow** | You're likely running on CPU. A machine with an NVIDIA GPU is much faster. |
| An **optional** (In-the-Wild / ASVspoof-2021) cell fails | That's expected if you didn't set up those datasets — it doesn't affect the main results. |

If a step fails, the notebook keeps going and records the full error in the **Final
Reproducibility Report** at the bottom, so you can always see exactly what happened.
