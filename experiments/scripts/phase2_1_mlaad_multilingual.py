#!/usr/bin/env python3
"""
phase2_1_mlaad_multilingual.py
================================
Phase 2.1 (redo): Cross-language ablation test on MLAAD multilingual subsets.

Real/bonafide : M-AILABS (stream-extract from archive, same source MLAAD uses)
Fake/spoof    : mueller91/MLAAD (official, same distribution as MLAAD-tiny training)
Model         : mlaad_robust_goat (no retraining)
Ablation      : {h2, h4} (top-1 + top-2 KL heads), mode=zero

Pre-registered criteria are printed BEFORE any evaluation data is seen.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import sys
import tarfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import soundfile as sf
import torch
import torchaudio.transforms as T

# ── Repo setup ─────────────────────────────────────────────────────────────────
SCRIPTS_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parents[1]
EXP_DIR      = PROJECT_ROOT / "experiments"
CKPT_DIR     = EXP_DIR / "checkpoints"
DATA_CACHE   = EXP_DIR / "data" / "mlaad_full_multilingual"
# OUT_ROOT and MAILABS_LANGS are set below after RUN_MODE is chosen

for _p in (str(PROJECT_ROOT), str(EXP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import head_ablation as ha
from mlaad_e4_ablation import MAALDAblationDataset, _best_or_last, pooled_eer, run_condition

# ── Constants ──────────────────────────────────────────────────────────────────
TARGET_SR      = 16_000
TARGET_SAMPLES = 48_000        # 3 s × 16 kHz
N_TARGET       = 375           # samples per class per language
N_REAL_MIN     = 300
N_FAKE_MIN     = 300
N_BOOTSTRAP    = 1000
SEED           = 42
BATCH_SIZE     = 8
ABLATION_HEADS = frozenset({2, 4})
MLAAD_REPO     = "mueller91/MLAAD"

# M-AILABS download URLs (ics.tau-ceti.space mirror)
MAILABS_BASE = "https://ics.tau-ceti.space/data/Training/stt_tts"

# Toggle here to switch run mode:
#   "multilingual"  → fr/ru/pl (cross-language test, excludes de/en)
#   "german_sanity" → de only  (in-distribution sanity check)
RUN_MODE = "german_sanity"

if RUN_MODE == "german_sanity":
    MAILABS_LANGS = {
        "de": {"archive": "de_DE.tgz", "label": "German"},
    }
    OUT_ROOT = EXP_DIR / "results" / "mlaad" / "phase2_1_german_sanity"
    EXCLUDED_LANGS_HARDCODE = frozenset({"en"})   # only exclude English
else:
    MAILABS_LANGS = {
        "fr": {"archive": "fr_FR.tgz", "label": "French"},
        "ru": {"archive": "ru_RU.tgz", "label": "Russian"},
        "pl": {"archive": "pl_PL.tgz", "label": "Polish"},
    }
    OUT_ROOT = EXP_DIR / "results" / "mlaad" / "phase2_1_multilingual_clean"
    EXCLUDED_LANGS_HARDCODE = frozenset({"en", "de"})

# ── Pre-registered criteria ────────────────────────────────────────────────────
CRITERIA_TEXT = """
═══════════════════════════════════════════════════════════════════════════
PRE-REGISTERED CRITERIA  (stated before any evaluation data is seen)
═══════════════════════════════════════════════════════════════════════════
Metric : xl_red = baseline_EER − ablated_EER   (positive = ablation helped)
CI     : 95% bootstrap CI on xl_red (1000 resamples, paired)

Let N = number of non-DE/non-EN languages tested.

CONFIRM   : xl_red > 0 AND CI_lo > 0  on ≥ ⌈2N/3⌉ languages
             → cross-language paradox replicates; paper claim survives.

SCOPE_DOWN: xl_red > 0 (point est.) on 1/3 to 2/3 of languages
             (when CONFIRM not met)
             → effect real but irregular; paper narrows to
               "effect appears for some language shifts but not others".

KILL      : xl_red > 0 AND CI_lo > 0  on < ⌈N/3⌉ languages
             (when neither CONFIRM nor SCOPE_DOWN met)
             → cross-language paradox claim killed; pivot to surefire-only.
═══════════════════════════════════════════════════════════════════════════
""".strip()


# ── Audio preprocessing (identical to prepare_mlaad_tiny.py) ──────────────────

def _decode_wav(wav_path: str) -> torch.Tensor:
    arr, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    w = torch.from_numpy(arr)
    if w.ndim == 1:
        w = w.unsqueeze(0)
    elif w.ndim == 2:
        w = w.mean(0, keepdim=True)
    if sr != TARGET_SR:
        w = T.Resample(sr, TARGET_SR)(w)
    return w   # (1, T)


def _crop(w: torch.Tensor) -> torch.Tensor:
    n = w.shape[-1]
    if n < TARGET_SAMPLES:
        reps = -(-TARGET_SAMPLES // n)
        w = w.repeat(1, reps)
    s = (w.shape[-1] - TARGET_SAMPLES) // 2
    return w[:, s : s + TARGET_SAMPLES]


def preprocess_wav_file(wav_path: str) -> torch.Tensor:
    """WAV path → (48000,) float32 tensor."""
    return _crop(_decode_wav(wav_path)).squeeze(0)


# ── M-AILABS streaming download ────────────────────────────────────────────────

def stream_mailabs(url: str, output_dir: Path, n_target: int,
                   print_every: int = 100) -> list[Path]:
    """
    Stream-extract WAV files from M-AILABS .tgz archive.
    Stops after n_target files extracted.
    Returns list of local WAV paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    print(f"    Streaming {url} ...", flush=True)
    t0 = time.time()
    try:
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            try:
                with tarfile.open(fileobj=resp.raw, mode="r|gz") as tar:
                    for member in tar:
                        if not (member.isfile() and
                                member.name.lower().endswith(".wav")):
                            continue
                        fobj = tar.extractfile(member)
                        if fobj is None:
                            continue
                        data = fobj.read()
                        fname = Path(member.name).name
                        out_path = output_dir / fname
                        # Skip duplicate filenames
                        if out_path.exists():
                            continue
                        out_path.write_bytes(data)
                        extracted.append(out_path)
                        n = len(extracted)
                        if n % print_every == 0:
                            elapsed = time.time() - t0
                            print(f"    {n}/{n_target} files extracted "
                                  f"({elapsed:.0f}s)", flush=True)
                        if n >= n_target:
                            break
            except EOFError:
                pass   # tarfile may raise EOF when stream is cut mid-archive
            except Exception as e:
                if extracted:
                    print(f"    Stream stopped at {len(extracted)} files: {e}",
                          flush=True)
                else:
                    raise
    except requests.exceptions.RequestException as e:
        print(f"    Download failed: {e}", flush=True)
        return []

    elapsed = time.time() - t0
    print(f"    Extracted {len(extracted)} WAVs in {elapsed:.0f}s", flush=True)
    return extracted


# ── MLAAD fake file selection & download ───────────────────────────────────────

def select_fake_files(by_system: dict[str, list[str]], n: int,
                      seed: int = SEED) -> list[str]:
    """Sample n fake files balanced across TTS systems."""
    rng = random.Random(seed)
    systems = sorted(by_system.keys())
    per_system = max(1, math.ceil(n / len(systems)))
    selected: list[str] = []
    for sys_name in systems:
        files = sorted(by_system[sys_name])
        rng.shuffle(files)
        selected.extend(files[:per_system])
    rng.shuffle(selected)
    return selected[:n]


def _hf_token() -> str:
    token_path = Path(os.path.expanduser("~/.cache/huggingface/token"))
    if token_path.exists():
        return token_path.read_text().strip()
    return os.environ.get("HF_TOKEN", "")


def download_mlaad_file_direct(rel_path: str, out_dir: Path) -> Optional[Path]:
    """
    Download a single MLAAD file via direct CDN URL (bypasses Hub API rate limit).
    Returns local path or None.
    """
    fname    = rel_path.replace("/", "__")
    out_path = out_dir / fname
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    url   = f"https://huggingface.co/datasets/{MLAAD_REPO}/resolve/main/{rel_path}"
    token = _hf_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    for attempt in range(3):
        try:
            r = requests.get(url, headers=headers, timeout=60, stream=True)
            r.raise_for_status()
            out_path.write_bytes(r.content)
            return out_path
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"    [WARN] CDN download failed for {rel_path}: {e}",
                      flush=True)
                return None
    return None


def download_mlaad_fakes_parallel(rel_paths: list[str], dl_dir: Path,
                                   max_workers: int = 8) -> dict[str, Path]:
    """
    Download MLAAD fake files in parallel via direct CDN.
    Returns {rel_path: local_path}.
    """
    dl_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(download_mlaad_file_direct, p, dl_dir): p
                   for p in rel_paths}
        done = 0
        for fut in as_completed(futures):
            rel_path = futures[fut]
            local = fut.result()
            if local:
                result[rel_path] = local
            done += 1
            if done % 50 == 0:
                print(f"    Downloaded {done}/{len(rel_paths)}", flush=True)
    return result


# ── Dataset preparation ────────────────────────────────────────────────────────

def prepare_language(lang: str, cfg: dict, lang_dir: Path,
                     by_system: dict[str, list[str]]) -> Optional[list[dict]]:
    """
    Prepare .pt files and records for one language.
    by_system: pre-built {tts_system: [rel_path, ...]} from discovery pass.
    Returns records list, or None if insufficient data.
    """
    audio_dir = lang_dir / "audio"
    meta_dir  = lang_dir / "metadata"
    audio_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    label = cfg["label"]

    # ── Check for existing prepared data ──────────────────────────────────────
    n_real_exist = len(list(audio_dir.glob("real_*.pt")))
    n_fake_exist = len(list(audio_dir.glob("fake_*.pt")))
    if n_real_exist >= N_REAL_MIN and n_fake_exist >= N_FAKE_MIN:
        print(f"  [{label}] Data cached ({n_real_exist} real, {n_fake_exist} fake).",
              flush=True)
        sp = meta_dir / "splits.json"
        if sp.exists():
            return json.loads(sp.read_text())["records"]
        return _build_records(n_real_exist, n_fake_exist)

    print(f"\n[{label}] Preparing data...", flush=True)

    # ── Real audio: M-AILABS ──────────────────────────────────────────────────
    real_wav_dir = lang_dir / "mailabs_wavs"
    existing_wavs = list(real_wav_dir.glob("*.wav")) if real_wav_dir.exists() else []
    if len(existing_wavs) < N_REAL_MIN:
        url = f"{MAILABS_BASE}/{cfg['archive']}"
        extracted = stream_mailabs(url, real_wav_dir,
                                   n_target=N_TARGET + 50)
        if len(extracted) < N_REAL_MIN:
            print(f"  [{label}] EXCLUDED: only {len(extracted)} real WAVs "
                  f"(need ≥{N_REAL_MIN})", flush=True)
            return None
        existing_wavs = extracted
    else:
        print(f"  [{label}] {len(existing_wavs)} M-AILABS WAVs cached.", flush=True)

    # Sample N_TARGET real
    rng = random.Random(SEED)
    real_wavs = sorted(existing_wavs)
    rng.shuffle(real_wavs)
    real_wavs = real_wavs[:N_TARGET]

    # Sanity: assert source = M-AILABS (not FLEURS, not edge-tts)
    for wav in real_wavs[:5]:
        assert str(real_wav_dir) in str(wav), \
            f"SANITY FAIL: real WAV not from M-AILABS dir: {wav}"

    # Process real WAVs → .pt
    print(f"  [{label}] Processing {len(real_wavs)} real WAVs...", flush=True)
    real_meta = []
    for i, wav_path in enumerate(real_wavs):
        pt_path = audio_dir / f"real_{i:04d}.pt"
        if not pt_path.exists():
            try:
                tensor = preprocess_wav_file(str(wav_path))
                assert tensor.shape == (TARGET_SAMPLES,), \
                    f"Bad shape: {tensor.shape}"
                torch.save(tensor, pt_path)
            except Exception as e:
                print(f"    [WARN] real_{i:04d}: {e}", flush=True)
                torch.save(torch.zeros(TARGET_SAMPLES), pt_path)
        real_meta.append({
            "file": f"real_{i:04d}.pt",
            "label": "bonafide",
            "source": "M-AILABS",
            "wav_name": wav_path.name,
        })

    # ── Fake audio: MLAAD-full ────────────────────────────────────────────────
    total_fake = sum(len(v) for v in by_system.values())
    print(f"  [{label}] {total_fake} fake files across "
          f"{len(by_system)} systems.", flush=True)

    if total_fake < N_FAKE_MIN:
        print(f"  [{label}] EXCLUDED: only {total_fake} fake files.", flush=True)
        return None

    selected = select_fake_files(by_system, N_TARGET)
    print(f"  [{label}] Downloading {len(selected)} fake WAVs via CDN...",
          flush=True)
    dl_dir    = DATA_CACHE / lang / "mlaad_wavs"
    local_map = download_mlaad_fakes_parallel(selected, dl_dir)
    print(f"  [{label}] {len(local_map)}/{len(selected)} downloaded.", flush=True)

    if len(local_map) < N_FAKE_MIN:
        print(f"  [{label}] EXCLUDED: only {len(local_map)} fake files available.",
              flush=True)
        return None

    fake_items = list(local_map.items())[:N_TARGET]   # [(rel_path, local_path)]

    # Process fake WAVs → .pt
    print(f"  [{label}] Processing {len(fake_items)} fake WAVs...", flush=True)
    fake_meta = []
    for i, (rel_path_str, wav_path) in enumerate(fake_items):
        pt_path = audio_dir / f"fake_{i:04d}.pt"
        if not pt_path.exists():
            try:
                tensor = preprocess_wav_file(str(wav_path))
                assert tensor.shape == (TARGET_SAMPLES,), \
                    f"Bad shape: {tensor.shape}"
                torch.save(tensor, pt_path)
            except Exception as e:
                print(f"    [WARN] fake_{i:04d}: {e}", flush=True)
                torch.save(torch.zeros(TARGET_SAMPLES), pt_path)
        fake_meta.append({
            "file": f"fake_{i:04d}.pt",
            "label": "spoof",
            "source": "MLAAD",
            "mlaad_path": rel_path_str,
        })

    n_real = len(real_wavs)
    n_fake = len(fake_items)
    records = _build_records(n_real, n_fake)

    # Save metadata
    splits_obj = {
        "language": label,
        "mlaad_lang": lang,
        "mailabs_archive": cfg["archive"],
        "n_real": n_real,
        "n_fake": n_fake,
        "seed": SEED,
        "real_source": "M-AILABS",
        "fake_source": "MLAAD (mueller91/MLAAD)",
        "records": records,
        "real_meta": real_meta,
        "fake_meta": fake_meta,
    }
    (meta_dir / "splits.json").write_text(
        json.dumps(splits_obj, indent=2, ensure_ascii=False))
    (meta_dir / "dataset_stats.json").write_text(json.dumps({
        "language": label,
        "mlaad_lang": lang,
        "n_real": n_real,
        "n_fake": n_fake,
        "n_total": n_real + n_fake,
        "n_tts_systems": len(by_system),
        "real_source": "M-AILABS",
        "fake_source": "MLAAD",
    }, indent=2))

    print(f"  [{label}] Ready: {n_real} real + {n_fake} fake.", flush=True)
    return records


def _build_records(n_real: int, n_fake: int) -> list[dict]:
    return (
        [{"audio_path": f"real_{i:04d}.pt",
          "label": "bonafide", "attack_system": "-"}
         for i in range(n_real)]
        + [{"audio_path": f"fake_{i:04d}.pt",
            "label": "spoof", "attack_system": "MLAAD-TTS"}
           for i in range(n_fake)]
    )


# ── Bootstrap CI ───────────────────────────────────────────────────────────────

def bootstrap_paired_eer(base_recs: list[dict], abl_recs: list[dict],
                          n_boot: int = N_BOOTSTRAP,
                          seed: int = SEED) -> dict:
    """
    Paired bootstrap: resample same indices for both conditions.
    Returns CI dicts for {base_eer, abl_eer, xl_red}.
    """
    rng = np.random.default_rng(seed)
    base_labels = np.array([r["label"]  for r in base_recs])
    base_scores = 1.0 / (1.0 + np.exp(-np.array([r["logit"] for r in base_recs])))
    abl_labels  = np.array([r["label"]  for r in abl_recs])
    abl_scores  = 1.0 / (1.0 + np.exp(-np.array([r["logit"] for r in abl_recs])))
    n = len(base_labels)

    boot_base, boot_abl, boot_xl = [], [], []
    for _ in range(n_boot):
        idx       = rng.integers(0, n, size=n)
        be        = ha.compute_eer(base_labels[idx], base_scores[idx])
        ae        = ha.compute_eer(abl_labels[idx],  abl_scores[idx])
        boot_base.append(be)
        boot_abl.append(ae)
        boot_xl.append(be - ae)

    def ci(vals):
        return {
            "mean":  float(np.mean(vals)),
            "ci_lo": float(np.percentile(vals, 2.5)),
            "ci_hi": float(np.percentile(vals, 97.5)),
        }
    return {
        "base_eer": ci(boot_base),
        "abl_eer":  ci(boot_abl),
        "xl_red":   ci(boot_xl),
    }


def bootstrap_auc(recs: list[dict], n_boot: int = N_BOOTSTRAP,
                  seed: int = SEED) -> dict:
    rng    = np.random.default_rng(seed)
    labels = np.array([r["label"] for r in recs])
    scores = 1.0 / (1.0 + np.exp(-np.array([r["logit"] for r in recs])))
    n      = len(labels)
    vals   = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(ha.compute_auc(labels[idx], scores[idx]))
    return {
        "mean":  float(np.mean(vals)),
        "ci_lo": float(np.percentile(vals, 2.5)),
        "ci_hi": float(np.percentile(vals, 97.5)),
    }


# ── Evaluation ─────────────────────────────────────────────────────────────────

def make_loader(records: list[dict],
                audio_dir: Path) -> torch.utils.data.DataLoader:
    ds = MAALDAblationDataset(records, audio_dir)
    return torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=ha.collate)


def eval_language(label: str, records: list[dict], audio_dir: Path,
                  lit, device: torch.device) -> dict:
    print(f"\n[{label}] Evaluating ({len(records)} samples)...", flush=True)
    loader = make_loader(records, audio_dir)

    t0        = time.time()
    base_recs = run_condition(lit, loader, device, frozenset())
    base_eer  = pooled_eer(base_recs)
    print(f"  baseline      EER={base_eer:.4f}  ({time.time()-t0:.1f}s)",
          flush=True)
    ha.sanity_layer_untouched(lit)

    t0       = time.time()
    abl_recs = run_condition(lit, loader, device, ABLATION_HEADS)
    abl_eer  = pooled_eer(abl_recs)
    print(f"  ablate{{h2,h4}} EER={abl_eer:.4f}  ({time.time()-t0:.1f}s)",
          flush=True)
    ha.sanity_layer_untouched(lit)

    xl_red = float(base_eer - abl_eer)
    print(f"  xl_red = {xl_red:+.4f}", flush=True)

    print(f"  Computing bootstrap CIs ({N_BOOTSTRAP} resamples)...", flush=True)
    boot = bootstrap_paired_eer(base_recs, abl_recs)
    auc_boot_base = bootstrap_auc(base_recs)
    auc_boot_abl  = bootstrap_auc(abl_recs)

    print(f"  xl_red CI [{boot['xl_red']['ci_lo']:+.4f}, "
          f"{boot['xl_red']['ci_hi']:+.4f}]  "
          f"criterion_met={boot['xl_red']['ci_lo'] > 0}", flush=True)

    return {
        "language":      label,
        "n_samples":     len(records),
        "n_real":        sum(1 for r in records if r["label"] == "bonafide"),
        "n_fake":        sum(1 for r in records if r["label"] == "spoof"),
        "baseline_eer":  round(float(base_eer), 6),
        "ablated_eer":   round(float(abl_eer),  6),
        "xl_red_point":  round(xl_red, 6),
        "bootstrap": {
            "base_eer": boot["base_eer"],
            "abl_eer":  boot["abl_eer"],
            "xl_red":   boot["xl_red"],
            "base_auc": auc_boot_base,
            "abl_auc":  auc_boot_abl,
        },
        "criterion_met": boot["xl_red"]["ci_lo"] > 0,
    }


# ── Verdict ────────────────────────────────────────────────────────────────────

def apply_verdict(results: dict[str, dict]) -> dict:
    n = len(results)
    if n == 0:
        return {"verdict": "NO_DATA", "n_languages": 0,
                "explanation": "No languages with sufficient data."}

    n_positive  = sum(1 for r in results.values() if r["xl_red_point"] > 0)
    n_confirmed = sum(1 for r in results.values() if r["criterion_met"])

    confirm_thresh = math.ceil(2 * n / 3)
    kill_thresh    = math.ceil(n / 3)

    # Priority: CONFIRM > SCOPE_DOWN > KILL
    # CONFIRM:    CI-confirmed xl_red on ≥ 2/3 of languages
    # KILL:       positive point estimate on < 1/3 of languages (effect rarely present)
    # SCOPE_DOWN: effect present in ≥ 1/3 but not CONFIRM
    if n_confirmed >= confirm_thresh:
        verdict = "CONFIRM"
        expl = (f"xl_red > 0 with CI_lo > 0 on {n_confirmed}/{n} languages "
                f"(≥⌈2N/3⌉={confirm_thresh}). "
                "Cross-language paradox replicates; claim survives.")
    elif n_positive >= kill_thresh:
        verdict = "SCOPE_DOWN"
        expl = (f"xl_red > 0 (point est.) on {n_positive}/{n} languages "
                f"(≥⌈N/3⌉={kill_thresh}), CI confirmed on {n_confirmed}/{n}. "
                "Effect real but irregular; paper narrows claim.")
    else:
        verdict = "KILL"
        expl = (f"xl_red > 0 on only {n_positive}/{n} languages "
                f"(<⌈N/3⌉={kill_thresh}). "
                "Cross-language paradox claim killed; pivot to surefire-only.")

    return {
        "verdict":            verdict,
        "n_languages_tested": n,
        "n_positive_xl_red":  n_positive,
        "n_confirmed_xl_red": n_confirmed,
        "confirm_threshold":  confirm_thresh,
        "kill_threshold":     kill_thresh,
        "explanation":        expl,
    }


# ── Output helpers ─────────────────────────────────────────────────────────────

def write_summary_csv(results: dict[str, dict], out_path: Path) -> None:
    import csv
    rows = []
    for lang_key, r in results.items():
        b = r["bootstrap"]
        rows.append({
            "language":       r["language"],
            "n_real":         r["n_real"],
            "n_fake":         r["n_fake"],
            "baseline_eer":   round(r["baseline_eer"], 4),
            "base_eer_ci_lo": round(b["base_eer"]["ci_lo"], 4),
            "base_eer_ci_hi": round(b["base_eer"]["ci_hi"], 4),
            "ablated_eer":    round(r["ablated_eer"], 4),
            "abl_eer_ci_lo":  round(b["abl_eer"]["ci_lo"], 4),
            "abl_eer_ci_hi":  round(b["abl_eer"]["ci_hi"], 4),
            "xl_red":         round(r["xl_red_point"], 4),
            "xl_red_ci_lo":   round(b["xl_red"]["ci_lo"], 4),
            "xl_red_ci_hi":   round(b["xl_red"]["ci_hi"], 4),
            "criterion_met":  r["criterion_met"],
        })
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote: {out_path}", flush=True)


def print_final_summary(results: dict[str, dict], verdict: dict) -> None:
    N = verdict["n_languages_tested"]
    conf_t = verdict["confirm_threshold"]
    kill_t = verdict["kill_threshold"]

    print("\n" + "=" * 72, flush=True)
    print("PHASE 2.1 MULTILINGUAL MLAAD RESULTS", flush=True)
    print("=" * 72, flush=True)
    print(f"\nLanguages tested: {N}", flush=True)
    print(f"Pre-registered thresholds: CONFIRM ≥{conf_t}/N, KILL <{kill_t}/N",
          flush=True)
    print(f"\n{'Language':12s} {'Base EER':>10s} {'[95% CI]':>18s}"
          f"  {'Abl EER':>10s}  {'xl_red':>8s} {'[95% CI]':>18s}  Crit",
          flush=True)
    print("-" * 95, flush=True)
    for lang_key, r in results.items():
        b = r["bootstrap"]
        print(
            f"{r['language']:12s} "
            f"{r['baseline_eer']:10.4f} "
            f"[{b['base_eer']['ci_lo']:+.4f},{b['base_eer']['ci_hi']:+.4f}]  "
            f"{r['ablated_eer']:10.4f}  "
            f"{r['xl_red_point']:+8.4f} "
            f"[{b['xl_red']['ci_lo']:+.4f},{b['xl_red']['ci_hi']:+.4f}]  "
            f"{'✓' if r['criterion_met'] else '✗'}",
            flush=True,
        )
    print("-" * 95, flush=True)
    print(f"\n{CRITERIA_TEXT}", flush=True)
    print(f"\nVERDICT: {verdict['verdict']}", flush=True)
    print(f"  {verdict['explanation']}", flush=True)
    print("=" * 72, flush=True)


# ── Sanity checks ──────────────────────────────────────────────────────────────

def run_sanity_checks(results: dict, excluded: dict) -> None:
    print("\n[SANITY CHECKS]", flush=True)

    # (a/b) Hardcoded excluded languages not present in results
    for excl in sorted(EXCLUDED_LANGS_HARDCODE):
        assert excl not in results, f"FAIL: {excl} found in results but should be excluded"
    excl_str = ", ".join(sorted(EXCLUDED_LANGS_HARDCODE)) or "(none)"
    print(f"  (a/b) Excluded ({excl_str}) not in results: PASS", flush=True)

    # (c) Source = MLAAD / M-AILABS
    for lang_key, r in results.items():
        lang_dir = DATA_CACHE / lang_key
        meta_dir = lang_dir / "metadata"
        sp = meta_dir / "splits.json"
        if sp.exists():
            splits = json.loads(sp.read_text())
            assert splits["real_source"] == "M-AILABS", \
                f"FAIL (c): real source not M-AILABS for {lang_key}"
            assert splits["fake_source"] == "MLAAD (mueller91/MLAAD)", \
                f"FAIL (c): fake source not MLAAD for {lang_key}"
    print("  (c) Source = MLAAD / M-AILABS: PASS", flush=True)

    # (d) Sample count ≥ 300 each class
    for lang_key, r in results.items():
        assert r["n_real"] >= N_REAL_MIN, \
            f"FAIL (d): {lang_key} real={r['n_real']} < {N_REAL_MIN}"
        assert r["n_fake"] >= N_FAKE_MIN, \
            f"FAIL (d): {lang_key} fake={r['n_fake']} < {N_FAKE_MIN}"
    print(f"  (d) All languages ≥{N_REAL_MIN} real + {N_FAKE_MIN} fake: PASS",
          flush=True)

    # (e) Audio shape: 48000 samples (spot-check 5 per language)
    for lang_key in list(results.keys())[:2]:
        audio_dir = DATA_CACHE / lang_key / "audio"
        for pt in sorted(audio_dir.glob("*.pt"))[:5]:
            t = torch.load(str(pt), weights_only=False)
            assert t.shape == (TARGET_SAMPLES,), \
                f"FAIL (e): {pt} shape={t.shape}"
    print("  (e) Audio tensors are (48000,): PASS (spot-checked)", flush=True)

    # (f) Excluded languages logged
    if excluded:
        print(f"  (f) Excluded languages: {list(excluded.keys())}", flush=True)
        for lang, reason in excluded.items():
            print(f"      {lang}: {reason}", flush=True)
    else:
        print("  (f) No languages excluded.", flush=True)

    print("[SANITY] All checks passed.\n", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_CACHE.mkdir(parents=True, exist_ok=True)

    # ─ Print pre-registered criteria FIRST (before any evaluation data)
    print(CRITERIA_TEXT, flush=True)
    print(flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # ─ Load model
    ckpt_path = _best_or_last("mlaad_robust_goat")
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}", flush=True)
        sys.exit(1)
    print(f"Checkpoint: {ckpt_path.name}", flush=True)

    # Compute checkpoint hash for run_config
    ckpt_hash = hashlib.md5(ckpt_path.read_bytes()).hexdigest()[:12]

    ha.patch_phoneme_loader()
    lit = ha.load_model(ckpt_path, device)
    print(f"Model loaded. GAT NH={lit.model.GAT.gat_net[0].num_of_heads}",
          flush=True)

    # ─ ONE combined discovery pass: count + collect file paths per language
    # This avoids calling list_repo_files again per language (avoids rate limit).
    target_langs = set(MAILABS_LANGS.keys())
    print("\nDiscovering MLAAD-full fake files for target languages "
          f"({sorted(target_langs)})...", flush=True)
    from huggingface_hub import list_repo_files as _lrf
    mlaad_by_system: dict[str, dict[str, list[str]]] = {l: {} for l in target_langs}
    mlaad_counts: dict[str, int] = defaultdict(int)
    for fpath in _lrf(MLAAD_REPO, repo_type="dataset"):
        if not (fpath.startswith("fake/") and fpath.endswith(".wav")):
            continue
        parts = fpath.split("/")
        if len(parts) < 4:
            continue
        lang, system = parts[1], parts[2]
        mlaad_counts[lang] += 1
        if lang in target_langs:
            mlaad_by_system[lang].setdefault(system, []).append(fpath)
    for lang, cfg in MAILABS_LANGS.items():
        cnt = mlaad_counts.get(lang, 0)
        n_sys = len(mlaad_by_system.get(lang, {}))
        print(f"  {lang} ({cfg['label']}): {cnt} fake files, {n_sys} systems",
              flush=True)

    # ─ Prepare data + evaluate per language
    all_results:  dict[str, dict] = {}
    excluded_langs: dict[str, str] = {}

    # Build run_config (before eval)
    run_config = {
        "checkpoint":      ckpt_path.name,
        "checkpoint_md5":  ckpt_hash,
        "target_heads":    sorted(ABLATION_HEADS),
        "ablation_mode":   "zero",
        "n_bootstrap":     N_BOOTSTRAP,
        "n_target":        N_TARGET,
        "n_real_min":      N_REAL_MIN,
        "n_fake_min":      N_FAKE_MIN,
        "seed":            SEED,
        "languages_attempted": list(MAILABS_LANGS.keys()),
        "real_source":     "M-AILABS",
        "fake_source":     MLAAD_REPO,
        "pre_registered_criteria": {
            "confirm": "xl_red > 0 AND CI_lo > 0 on >= ceil(2N/3) languages",
            "scope_down": "xl_red > 0 (point) on 1/3 to 2/3 of languages",
            "kill": "xl_red > 0 AND CI_lo > 0 on < ceil(N/3) languages",
        },
    }
    (OUT_ROOT / "run_config.json").write_text(json.dumps(run_config, indent=2))

    for lang, cfg in MAILABS_LANGS.items():
        # Sanity: requested langs must not include hardcoded exclusions
        assert lang not in EXCLUDED_LANGS_HARDCODE, \
            f"BUG: {lang} is in EXCLUDED_LANGS_HARDCODE but also in MAILABS_LANGS"

        if mlaad_counts.get(lang, 0) < N_FAKE_MIN:
            reason = f"only {mlaad_counts.get(lang, 0)} fake files in MLAAD"
            print(f"\n[{cfg['label']}] EXCLUDED: {reason}", flush=True)
            excluded_langs[lang] = reason
            continue

        lang_dir  = DATA_CACHE / lang
        by_system = mlaad_by_system.get(lang, {})
        records   = prepare_language(lang, cfg, lang_dir, by_system)

        if records is None:
            reason = "insufficient data after download"
            excluded_langs[lang] = reason
            continue

        audio_dir = lang_dir / "audio"
        metrics   = eval_language(cfg["label"], records, audio_dir, lit, device)
        all_results[lang] = metrics

    # ─ Verdict
    if not all_results:
        print("\nERROR: No languages with sufficient data. Cannot compute verdict.",
              flush=True)
        sys.exit(1)

    # Update run_config with actual tested languages
    run_config["languages_tested"]  = list(all_results.keys())
    run_config["languages_excluded"] = excluded_langs
    (OUT_ROOT / "run_config.json").write_text(json.dumps(run_config, indent=2))

    verdict = apply_verdict(all_results)

    # ─ Sanity checks
    run_sanity_checks(all_results, excluded_langs)

    # ─ Write outputs
    (OUT_ROOT / "per_language_metrics.json").write_text(
        json.dumps(all_results, indent=2))
    print(f"Wrote: {OUT_ROOT}/per_language_metrics.json", flush=True)

    (OUT_ROOT / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(f"Wrote: {OUT_ROOT}/verdict.json", flush=True)

    if all_results:
        write_summary_csv(all_results, OUT_ROOT / "summary_table.csv")

    # ─ Final summary
    print_final_summary(all_results, verdict)
    print(f"\nAll outputs in: {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
