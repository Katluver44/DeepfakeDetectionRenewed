#!/usr/bin/env python3
"""
e9_attn_cache.py — Extract and cache A_agg matrices for E9 mechanism analysis.

For each E9 checkpoint, runs 50 held-out samples per system through the model
and saves all-layer attention artifacts, mirroring the format of
gat_attn_graphs/all_layers_artifacts.pt used by the original E2–E7 scripts.

Usage:
  venv/bin/python3 experiments/e9_attn_cache.py --seed 1
  venv/bin/python3 experiments/e9_attn_cache.py --seed 2

Outputs:
  experiments/results/e9_mixed_training/seed{N}/all_layers_artifacts.pt
  experiments/results/e9_mixed_training/seed{N}/eer_eval.json
"""
from __future__ import annotations

import argparse, csv, io, json, os, random, sys, time, types, warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.transforms as T_audio
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--seed",        type=int, default=1)
parser.add_argument("--n_per_class", type=int, default=50,
                    help="Target samples per system for mechanism eval")
parser.add_argument("--also_asvspoof", type=int, default=1,
                    help="Also evaluate on ASVspoof 2019 LA eval set (condition B)")
args = parser.parse_args()

SEED = args.seed
N_PER = args.n_per_class
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

REPO_ROOT   = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CKPT_PATH   = REPO_ROOT / "models" / f"e9_seed{SEED}.ckpt"
MANIFEST    = REPO_ROOT / "experiments" / "results" / "e9_mixed_training" / f"seed{SEED}_train_manifest.csv"
OUT_DIR     = REPO_ROOT / "experiments" / "results" / "e9_mixed_training" / f"seed{SEED}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR      = 16_000
TARGET_SAMPLES = 3 * TARGET_SR
NF_PER_SAMPLE  = TARGET_SAMPLES // 320 - 1
BATCH_SIZE     = 8
MIN_NODES      = 3

# ── torch.load compat ────────────────────────────────────────────────────────
_orig = torch.load
def _patched(*a, **kw): kw.setdefault("weights_only", False); return _orig(*a, **kw)
torch.load = _patched

from argparse import Namespace
try:
    from pandas import Series as _PS
    from ay2.tools.text._phonemes import Phonemer_Tokenizer_Recombination as _PTR
    torch.serialization.add_safe_globals([Namespace, _PS, _PTR])
except Exception:
    torch.serialization.add_safe_globals([Namespace])

import torchaudio
from loader import _crop_policy, _ensure_sr


# ── Load training manifest to find held-out files ───────────────────────────

def load_manifest(manifest_path: Path) -> set[str]:
    """Return set of training (non-val) file paths."""
    train_paths = set()
    if not manifest_path.exists():
        print(f"[WARN] Manifest not found: {manifest_path}. "
              "Cannot exclude training files from eval set.")
        return train_paths
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") == "train":
                if row["path"] != "__hf__":
                    train_paths.add(row["path"])
    print(f"Loaded manifest: {len(train_paths)} training file paths to exclude from eval")
    return train_paths


# ── Sample eval files per source ─────────────────────────────────────────────

def sample_wavefake_eval(wavefake_root: Path, n_per_system: int,
                         excluded: set[str], seed: int) -> list[dict]:
    """n_per_system samples per WaveFake system, excluding training files."""
    audio_root = wavefake_root / "generated_audio"
    if not audio_root.exists():
        raise RuntimeError(f"WaveFake not extracted at {audio_root}")

    by_system: dict[str, list[str]] = defaultdict(list)
    for sys_dir in sorted(audio_root.iterdir()):
        if not sys_dir.is_dir(): continue
        sys_id = sys_dir.name
        if sys_id.startswith("jsut"): continue   # Japanese
        for fpath in sys_dir.rglob("*.wav"):
            p = str(fpath)
            if p not in excluded:
                by_system[sys_id].append(p)

    rng = random.Random(seed)
    records = []
    for sys_id, paths in sorted(by_system.items()):
        rng.shuffle(paths)
        chosen = paths[:n_per_system]
        for p in chosen:
            records.append({
                "path": p, "label": 1,
                "system_id": sys_id, "source": "wavefake"
            })
        if len(chosen) < n_per_system:
            print(f"  [WARN] {sys_id}: only {len(chosen)}/{n_per_system} held-out files")
    print(f"WaveFake eval: {len(records)} records, "
          f"{len(by_system)} systems")
    return records


def sample_vcc2020_eval(vcc2020_root: Path, n_per_system: int,
                        excluded: set[str], seed: int) -> list[dict]:
    """50 samples per VCC2020 system (T01–T33), excluding training files."""
    manifest = vcc2020_root / "manifest_task1.json"
    with open(manifest) as f:
        m = json.load(f)

    rng = random.Random(seed)
    records = []
    for sys_id, paths in sorted(m.items()):
        available = [p for p in paths if os.path.exists(p) and p not in excluded]
        rng.shuffle(available)
        chosen = available[:n_per_system]
        for p in chosen:
            records.append({
                "path": p, "label": 1,
                "system_id": f"VCC2020_{sys_id}", "source": "vcc2020"
            })
        if len(chosen) < n_per_system:
            print(f"  [WARN] VCC2020_{sys_id}: only {len(chosen)}/{n_per_system} held-out files")
    print(f"VCC2020 eval: {len(records)} records, "
          f"{len(m)} systems")
    return records


def sample_librispeech_eval(n_bonafide: int, excluded_hf_indices: set[int],
                             seed: int, token=None) -> tuple[list[dict], object]:
    """n_bonafide samples from LibriSpeech train-clean-100, excluding training indices."""
    from datasets import load_dataset, Audio as HFAudio
    ds = load_dataset("openslr/librispeech_asr", "clean",
                      split="train.100",
                      cache_dir=str(REPO_ROOT / "data" / "librispeech"),
                      token=token)
    ds = ds.cast_column("audio", HFAudio(sampling_rate=TARGET_SR))

    available = [i for i in range(len(ds)) if i not in excluded_hf_indices]
    rng = random.Random(seed)
    rng.shuffle(available)
    chosen = available[:n_bonafide]

    records = [
        {"path": "__hf__", "hf_idx": i, "label": 0,
         "system_id": "librispeech", "source": "librispeech"}
        for i in chosen
    ]
    print(f"LibriSpeech eval: {len(records)} bonafide records")
    return records, ds


# ── Dataset wrapper ───────────────────────────────────────────────────────────

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict], hf_ds=None):
        self.records = records
        self.hf_ds   = hf_ds

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        if rec["path"] == "__hf__":
            ex  = self.hf_ds[rec["hf_idx"]]
            wav = torch.tensor(ex["audio"]["array"], dtype=torch.float32).unsqueeze(0)
            assert ex["audio"]["sampling_rate"] == TARGET_SR
        else:
            wav, sr = torchaudio.load(rec["path"])
            wav = _ensure_sr(wav, sr)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
        wav = _crop_policy(wav, "eval")
        return {
            "audio":     wav,
            "label":     torch.tensor(rec["label"]).long(),
            "system_id": rec["system_id"],
        }


def collate(batch):
    return {
        "audio":     torch.stack([b["audio"]  for b in batch]),
        "label":     torch.stack([b["label"]  for b in batch]),
        "system_id": [b["system_id"] for b in batch],
    }


# ── Model loading ─────────────────────────────────────────────────────────────

def patch_phoneme_loader():
    import phoneme_GAT.modules as mm
    import phoneme_GAT.phoneme_model as pm
    from phoneme_GAT.phoneme_model import BaseModule, network_param, optim_param

    def _load(network_name="wavlm", pretrained_path=None, total_num_phonemes=198):
        network_param.network_name    = network_name
        network_param.pretrained_name = (
            "microsoft/wavlm-base" if network_name.lower() == "wavlm"
            else "facebook/wav2vec2-base-960h")
        network_param.vocab_size = total_num_phonemes
        if pretrained_path and Path(pretrained_path).exists():
            return BaseModule.load_from_checkpoint(
                str(pretrained_path), network_param=network_param,
                optim_param=optim_param, tokenizer=None,
                total_num_phonemes=total_num_phonemes, weights_only=False).cpu()
        return BaseModule(network_param, optim_param, tokenizer=None,
                          total_num_phonemes=total_num_phonemes)

    pm.load_phoneme_model = _load
    mm.load_phoneme_model = _load


def load_model(ckpt_path: Path, device: torch.device):
    from phoneme_GAT.modules import Phoneme_GAT_lit
    cfg = Namespace(PhonemeGAT=Namespace(
        backbone="wavlm", use_raw=False, use_GAT=True,
        n_edges=10, use_aug=True, use_pool=True, use_clip=True))
    lit = Phoneme_GAT_lit.load_from_checkpoint(
        str(ckpt_path), cfg=cfg, map_location=device, strict=True)
    lit.to(device); lit.eval(); lit.freeze()
    return lit


# ── PhonemeCapture (identical to gat_attn_graphs.py) ─────────────────────────

class PhonemeCapture:
    def __init__(self, gat_model):
        self.node_phoneme_ids   = None
        self.node_sample_idx    = None
        self.reduced_num_frames = None
        cap  = self
        orig = gat_model.encoder_and_GAT.__func__

        def _patched(self_inner, hidden_states, num_frames, phoneme_ids,
                     profiler=None, use_encoder=True, ground_truth_labels=None):
            result = orig(self_inner, hidden_states, num_frames, phoneme_ids,
                          profiler=profiler, use_encoder=use_encoder,
                          ground_truth_labels=ground_truth_labels)
            rids = result[2].detach().cpu()
            rnf  = result[3].detach().cpu()
            flat_ids, flat_samp = [], []
            for i in range(len(rnf)):
                n = int(rnf[i].item())
                flat_ids.append(rids[i, :n])
                flat_samp.append(torch.full((n,), i, dtype=torch.long))
            cap.node_phoneme_ids   = torch.cat(flat_ids)
            cap.node_sample_idx    = torch.cat(flat_samp)
            cap.reduced_num_frames = rnf
            return result

        gat_model.encoder_and_GAT = types.MethodType(_patched, gat_model)


def run_frozen_frontend(audio, gm, device):
    x = audio
    if x.ndim == 3 and x.size(1) == 1:
        x = x[:, 0, :]
    with torch.no_grad():
        feat1 = gm.transformer_in_phoneme_model.feature_extractor(x).transpose(1, 2)
        hidden, _ = gm.transformer_in_phoneme_model.feature_projection(feat1)
        phoneme_feat = gm.transformer_in_phoneme_model.encoder(hidden)[0]
        phoneme_logits = gm.phoneme_model.model.model.lm_head(phoneme_feat)
        phoneme_ids = torch.argmax(phoneme_logits, dim=-1)
    return hidden, phoneme_ids


# ── Extraction (same logic as gat_attn_graphs.extract_all_layers) ─────────────

def extract_all_layers(lit, loader, device) -> tuple[list[dict], int]:
    gm = lit.model
    n_layers = len(gm.GAT.gat_net)
    assert n_layers == 3

    for i in range(n_layers):
        gm.GAT.gat_net[i].log_attention_weights = True

    buf = {f"attn_{i}": None for i in range(n_layers)}
    buf.update({f"eidx_{i}": None for i in range(n_layers)})

    hooks = []
    for li in range(n_layers):
        def _make_hook(idx):
            def _hook(module, inp, out):
                buf[f"eidx_{idx}"] = out[1].detach().cpu()
                if module.attention_weights is not None:
                    buf[f"attn_{idx}"] = module.attention_weights.squeeze(-1).detach().cpu()
            return _hook
        hooks.append(gm.GAT.gat_net[li].register_forward_hook(_make_hook(li)))

    phoneme_cap = PhonemeCapture(gm)
    records: list[dict] = []
    n_degen = 0
    sanity_done = False
    total = len(loader.dataset)

    try:
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                audio   = batch["audio"].to(device)
                labels  = batch["label"].tolist()
                sys_ids = batch["system_id"]
                B       = len(labels)
                num_f   = torch.full((B,), NF_PER_SAMPLE, device=device)

                hidden, phoneme_ids = run_frozen_frontend(audio, gm, device)

                with torch.no_grad():
                    gm.encoder_and_GAT(hidden, num_f, phoneme_ids)

                if not sanity_done:
                    for li in range(1, n_layers):
                        assert torch.equal(buf["eidx_0"], buf[f"eidx_{li}"])
                    print(f"  [OK] Edge indices identical across all {n_layers} layers")
                    sanity_done = True

                edge_index_global = buf["eidx_0"]
                attn_global       = [buf[f"attn_{li}"] for li in range(n_layers)]

                if any(a is None for a in attn_global) or edge_index_global is None:
                    print(f"  [WARN] batch {bi}: hook missing, skipping")
                    continue

                node_pids     = phoneme_cap.node_phoneme_ids
                node_samp_idx = phoneme_cap.node_sample_idx
                rnf           = phoneme_cap.reduced_num_frames

                node_offsets = [0]
                for i in range(B - 1):
                    node_offsets.append(node_offsets[-1] + int(rnf[i].item()))

                base_id = len(records)
                for i in range(B):
                    n_nodes_i = int(rnf[i].item())
                    offset_i  = node_offsets[i]
                    src_global = edge_index_global[0]
                    edge_mask  = (node_samp_idx[src_global] == i)

                    ei_global_i = edge_index_global[:, edge_mask]
                    ei_local    = ei_global_i - offset_i
                    is_adj      = (ei_local[1] - ei_local[0]) == 1
                    pids_i      = node_pids[node_samp_idx == i]
                    is_deg      = n_nodes_i < MIN_NODES
                    if is_deg: n_degen += 1

                    rec = {
                        "sample_id":        base_id + i,
                        "label":            labels[i],
                        "system_id":        sys_ids[i],
                        "node_phoneme_ids": pids_i.clone(),
                        "edge_index":       ei_local.clone(),
                        "edge_type_adj":    is_adj.clone(),
                        "n_nodes":          n_nodes_i,
                        "n_edges":          ei_local.shape[1],
                        "is_degenerate":    is_deg,
                    }
                    for li in range(n_layers):
                        rec[f"attn_l{li}"] = attn_global[li][edge_mask].clone()
                    records.append(rec)

                for li in range(n_layers):
                    buf[f"attn_{li}"] = None
                    buf[f"eidx_{li}"] = None

                if (bi + 1) % 10 == 0 or (bi + 1) == len(loader):
                    print(f"  {min((bi+1)*BATCH_SIZE, total)}/{total}")

    finally:
        for h in hooks: h.remove()

    print(f"  Extracted {len(records)} samples ({n_degen} degenerate)")
    return records, n_degen


def records_to_artifact(records: list[dict]) -> dict:
    """Pack records list into a single dict of stacked tensors (same format as
    gat_attn_graphs/all_layers_artifacts.pt)."""
    keys_list  = ["sample_ids", "labels", "system_ids", "n_nodes", "n_edges",
                  "is_degenerate"]
    artifact = {
        "sample_ids":    [r["sample_id"]  for r in records],
        "labels":        [r["label"]       for r in records],
        "system_ids":    [r["system_id"]   for r in records],
        "n_nodes":       [r["n_nodes"]     for r in records],
        "n_edges":       [r["n_edges"]     for r in records],
        "is_degenerate": [r["is_degenerate"] for r in records],
        "node_phoneme_ids": [r["node_phoneme_ids"] for r in records],
        "edge_index":       [r["edge_index"]       for r in records],
        "edge_type_adj":    [r["edge_type_adj"]     for r in records],
        "attn_l0": [r["attn_l0"] for r in records],
        "attn_l1": [r["attn_l1"] for r in records],
        "attn_l2": [r["attn_l2"] for r in records],
    }
    return artifact


# ── EER evaluation ────────────────────────────────────────────────────────────

def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    from scipy.interpolate import interp1d
    thresholds = np.sort(np.unique(scores))
    fprs, fnrs = [], []
    for t in thresholds:
        pred = (scores >= t).astype(int)
        fp = ((pred == 1) & (labels == 0)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        tn = ((pred == 0) & (labels == 0)).sum()
        tp = ((pred == 1) & (labels == 1)).sum()
        fpr = fp / (fp + tn + 1e-10)
        fnr = fn / (fn + tp + 1e-10)
        fprs.append(fpr); fnrs.append(fnr)
    fprs, fnrs = np.array(fprs), np.array(fnrs)
    diff = np.abs(fprs - fnrs)
    idx  = np.argmin(diff)
    return float((fprs[idx] + fnrs[idx]) / 2)


def eval_eer(lit, loader, device) -> dict:
    all_logits, all_labels = [], []
    lit.eval()
    with torch.no_grad():
        for batch in loader:
            audio  = batch["audio"].to(device)
            labels = batch["label"].tolist()
            B      = len(labels)
            num_f  = torch.full((B,), NF_PER_SAMPLE, device=device)
            gm     = lit.model
            hidden, phoneme_ids = run_frozen_frontend(audio, gm, device)
            _, _, _, _, _, logit = gm.encoder_and_GAT(hidden, num_f, phoneme_ids)
            all_logits.extend(logit.cpu().tolist())
            all_labels.extend(labels)
    scores = np.array(all_logits)
    labels = np.array(all_labels)
    eer = compute_eer(scores, labels)
    return {"eer": eer, "n": len(labels),
            "n_bonafide": int((labels == 0).sum()),
            "n_spoof":    int((labels == 1).sum())}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    assert CKPT_PATH.exists(), f"Checkpoint not found: {CKPT_PATH}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token  = (REPO_ROOT / "secret.txt").read_text().strip() \
             if (REPO_ROOT / "secret.txt").exists() else None

    print(f"=== E9 Attention Cache  seed={SEED}  n_per={N_PER} ===")
    patch_phoneme_loader()
    lit = load_model(CKPT_PATH, device)
    print("Model loaded.")

    # Read training manifest to exclude training files
    excluded = load_manifest(MANIFEST)
    excluded_hf_idx = set()  # LibriSpeech HF indices used in training
    if MANIFEST.exists():
        with open(MANIFEST, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("split") == "train" and row["path"] == "__hf__":
                    excluded_hf_idx.add(int(row["hf_idx"]))

    # ── Build eval records ─────────────────────────��──────────────────────────
    wf_records  = sample_wavefake_eval(
        REPO_ROOT / "data" / "wavefake", N_PER, excluded, SEED)
    vcc_records = sample_vcc2020_eval(
        REPO_ROOT / "data" / "vcc2020",  N_PER, excluded, SEED)
    ls_records, librispeech_ds = sample_librispeech_eval(
        n_bonafide=len(wf_records) + len(vcc_records),
        excluded_hf_indices=excluded_hf_idx,
        seed=SEED, token=token)

    all_records = wf_records + vcc_records + ls_records
    print(f"\nTotal eval records: {len(all_records)}")

    dataset = EvalDataset(all_records, hf_ds=librispeech_ds)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, collate_fn=collate)

    # ── EER sanity check ──────────────────────────────────────────────��──────
    print("\nComputing EER on new-domain eval set (condition A)...")
    eer_info = eval_eer(lit, loader, device)
    print(f"EER = {eer_info['eer']*100:.2f}%  "
          f"(n={eer_info['n']}, bona={eer_info['n_bonafide']}, spoof={eer_info['n_spoof']})")
    if eer_info["eer"] > 0.20:
        print(f"\n[GATE FAIL] EER={eer_info['eer']*100:.2f}% > 20%. "
              "Mechanism analysis on a poorly-trained model is invalid.")
        print("Stopping. Investigate training before continuing.")
        json.dump(eer_info, open(OUT_DIR / "eer_eval.json", "w"), indent=2)
        sys.exit(1)
    else:
        print("[GATE PASS] EER ≤ 20%. Proceeding to mechanism analysis.")

    with open(OUT_DIR / "eer_eval.json", "w") as f:
        json.dump(eer_info, f, indent=2)

    # ── Extract all-layer attention ───────────────────────────────────────────
    print("\nExtracting all-layer attention artifacts...")
    records, n_degen = extract_all_layers(lit, loader, device)

    artifact = records_to_artifact(records)
    out_path = OUT_DIR / "all_layers_artifacts.pt"
    torch.save(artifact, out_path)
    print(f"\nSaved: {out_path}  ({len(records)} samples, {n_degen} degenerate)")

    # ── Also evaluate on ASVspoof 2019 LA (condition B) ──────────────────────
    if args.also_asvspoof:
        print("\n" + "─"*60)
        print("Condition B: ASVspoof 2019 LA evaluation (cross-protocol)")
        from datasets import load_dataset, Audio as HFAudio
        from collections import Counter

        hf = load_dataset("Bisher/ASVspoof_2019_LA", split="test",
                          cache_dir=str(REPO_ROOT / "data" / "asvspoof_2019_la"),
                          token=token)
        hf = hf.cast_column("audio", HFAudio(sampling_rate=TARGET_SR))

        by_sys: dict[str, list[int]] = defaultdict(list)
        for i in range(len(hf)):
            by_sys[hf[i].get("system_id", "?")].append(i)

        rng = random.Random(SEED)
        asv_records_b = []
        for sys_id, idxs in sorted(by_sys.items()):
            rng.shuffle(idxs)
            for idx in idxs[:N_PER]:
                asv_records_b.append({
                    "path": "__hf__", "hf_idx": idx,
                    "label": int(hf[idx]["key"]),
                    "system_id": sys_id, "source": "asvspoof"
                })

        asv_ds_b = EvalDataset(asv_records_b, hf_ds=hf)
        asv_loader_b = torch.utils.data.DataLoader(
            asv_ds_b, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=4, collate_fn=collate)

        asv_eer = eval_eer(lit, asv_loader_b, device)
        print(f"Cross-protocol EER = {asv_eer['eer']*100:.2f}%  "
              f"(n={asv_eer['n']})")
        with open(OUT_DIR / "eer_eval_asvspoof.json", "w") as f:
            json.dump(asv_eer, f, indent=2)

    print(f"\n=== Done. Artifacts: {OUT_DIR} ===")


if __name__ == "__main__":
    main()
