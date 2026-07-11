"""C3(b) — replicate the laughter-insertion evasion + worst-window defense on
official AASIST (spectro-temporal, non-WavLM), reusing the SAME clean vs
laughter-inserted ASVspoof2019-LA eval sets used for the WavLM-GAT result
(data/eval_asv19, data/eval_asv19_aug).

AASIST loading follows experiments/scripts/j5_aasist_crossfamily.py exactly:
  from models.AASIST import Model as AASIST
  conf = json.load(baselines/aasist/config/AASIST.conf)["model_config"]
  model.load_state_dict(torch.load(baselines/aasist/models/weights/AASIST.pth))
  model(x) -> (hidden, output) ; output[:, 1] = bona-fide logit, output[:, 0] = spoof logit
  (2-way softmax head; j5 orients so higher = spoof empirically per-run)

Input: raw 16kHz waveform, nb_samp=64600 (~4.04s), tile-pad if short / crop if long
(official AASIST eval convention: pad_random / take from the start; we use the
SAME center-crop-or-tile policy as score_itw_detector.py for apples-to-apples
methodology with the WavLM-GAT pipeline, and also expose a sliding-window
worst-window scorer analogous to d1_defense.py).

Writes:
  rerun_2026/concerns/tables/c3_aasist_insertion.csv      (paired evasion table, d1-style)
  rerun_2026/concerns/tables/c3_aasist_defense.csv        (mean/max/p90 worst-window defense, d1_defense-style)
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import roc_curve

LAUGHSMI = Path("/home/sagemaker-user/DeepfakeDetectionRenewed/laughsmi")
REPO = LAUGHSMI.parent
AASIST_DIR = REPO / "baselines" / "aasist"
sys.path.insert(0, str(AASIST_DIR))
sys.path.insert(0, str(LAUGHSMI / "scripts"))

from models.AASIST import Model as AASISTModel  # noqa: E402
from score_itw_detector import load_wav_mono_16k  # noqa: E402

SR = 16000
NB_SAMP = 64600  # official AASIST fixed input length (~4.04s)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_aasist():
    conf = json.loads((AASIST_DIR / "config" / "AASIST.conf").read_text())
    model = AASISTModel(conf["model_config"]).to(DEVICE)
    sd = torch.load(AASIST_DIR / "models" / "weights" / "AASIST.pth", map_location=DEVICE, weights_only=False)
    model.load_state_dict(sd)
    model.eval()
    return model


def crop_or_tile_center(wav: np.ndarray, target_len: int = NB_SAMP) -> np.ndarray:
    T = len(wav)
    if T == 0:
        return np.zeros(target_len, dtype=np.float32)
    if T < target_len:
        reps = -(-target_len // T)
        return np.tile(wav, reps)[:target_len]
    start = (T - target_len) // 2
    return wav[start:start + target_len]


@torch.no_grad()
def raw_score_batch(model, wavs, bs=16):
    """Returns np array of RAW AASIST col-1 scores (official convention: col 1 =
    bona-fide log-prob), UNORIENTED. Orientation (sign) is determined empirically
    per-run from the clean labeled set, exactly as experiments/scripts/
    j5_aasist_crossfamily.py does (`orient = +1 if median(spoof) > median(bona)
    else -1`), rather than hardcoding a sign convention that could be backwards."""
    out = []
    for i in range(0, len(wavs), bs):
        xb = torch.as_tensor(np.stack(wavs[i:i + bs]), dtype=torch.float32, device=DEVICE)
        _, o = model(xb)
        out.append(o.float().cpu().numpy())
    logits = np.concatenate(out, axis=0)  # (N, 2)
    return logits[:, 1]


ORIENT = 1.0  # set by main() after computing on the clean labeled set; higher*ORIENT = more spoof-like


def score_batch(model, wavs, bs=16):
    return ORIENT * raw_score_batch(model, wavs, bs=bs)


def window_scores(model, wav, hop_s=1.0, bs=16):
    hop = int(hop_s * SR)
    T = len(wav)
    if T <= NB_SAMP:
        return score_batch(model, [crop_or_tile_center(wav)], bs=bs)
    starts = list(range(0, T - NB_SAMP + 1, hop))
    if starts[-1] != T - NB_SAMP:
        starts.append(T - NB_SAMP)
    wins = [crop_or_tile_center(wav[s:s + NB_SAMP]) for s in starts]
    return score_batch(model, wins, bs=bs)


def aggregate(ws):
    return {"mean": float(ws.mean()), "max": float(ws.max()), "p90": float(np.percentile(ws, 90))}


def score_dir_single(model, audio_dir, meta_rows):
    """Single center-crop score (the attack surface, matches score_itw_detector default)."""
    rows = []
    for r in meta_rows:
        wav = load_wav_mono_16k(str(audio_dir / r["file"]))
        wav_c = crop_or_tile_center(wav)
        s = score_batch(model, [wav_c])[0]
        rows.append({"file": Path(r["file"]).name, "label": r["label"], "score": float(s)})
    return rows


def score_dir_windowed(model, audio_dir, meta_rows):
    rows = []
    for r in meta_rows:
        wav = load_wav_mono_16k(str(audio_dir / r["file"]))
        agg = aggregate(window_scores(model, wav))
        rows.append({"file": Path(r["file"]).name, "label": r["label"], **agg})
    return rows


def eer_thr(scores, labels):
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float(thr[i]), float((fpr[i] + fnr[i]) / 2)


def rank_biserial_from_wilcoxon(deltas):
    d = deltas[deltas != 0]
    if len(d) == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    r_plus = ranks[d > 0].sum()
    r_minus = ranks[d < 0].sum()
    total = ranks.sum()
    return float((r_plus - r_minus) / total)


def main():
    global ORIENT
    model = load_aasist()
    print("[c3_aasist] official AASIST loaded on", DEVICE)

    clean_dir = LAUGHSMI / "data" / "eval_asv19"
    aug_dir = LAUGHSMI / "data" / "eval_asv19_aug"
    clean_meta = list(csv.DictReader(open(clean_dir / "meta.csv")))
    aug_meta_all = list(csv.DictReader(open(aug_dir / "meta.csv")))
    manifest = {Path(r["file"]).name: r for r in csv.DictReader(open(aug_dir / "manifest.csv"))}

    # ---- determine score orientation empirically on the clean labeled set
    # (same approach as j5_aasist_crossfamily.py: raw col-1 score, orient so
    # that higher = spoof by comparing medians) ----
    print("[c3_aasist] determining score orientation on clean eval_asv19 ...")
    raw_probe = []
    for r in clean_meta:
        wav = load_wav_mono_16k(str(clean_dir / r["file"]))
        raw_probe.append((raw_score_batch(model, [crop_or_tile_center(wav)])[0], r["label"]))
    raw_spoof = np.array([s for s, l in raw_probe if l == "spoof"])
    raw_bona = np.array([s for s, l in raw_probe if l == "bona-fide"])
    ORIENT = 1.0 if np.median(raw_spoof) > np.median(raw_bona) else -1.0
    print(f"[c3_aasist] ORIENT = {ORIENT:+.0f}  (raw col-1 median spoof={np.median(raw_spoof):.3f} "
          f"bona={np.median(raw_bona):.3f})")

    # ---- single-center-crop scoring (the attack surface) ----
    print("[c3_aasist] scoring clean eval_asv19 (single center-crop) ...")
    clean_single = score_dir_single(model, clean_dir, clean_meta)
    print("[c3_aasist] scoring aug eval_asv19_aug (single center-crop) ...")
    aug_single = score_dir_single(model, aug_dir, aug_meta_all)

    clean_by_id = {r["file"]: r for r in clean_single}
    aug_by_id = {r["file"]: r for r in aug_single}

    # clean-EER threshold from clean set
    sc = np.array([r["score"] for r in clean_single])
    yl = np.array([0 if r["label"] == "bona-fide" else 1 for r in clean_single])
    thr, eer = eer_thr(sc, yl)
    print(f"[c3_aasist] clean EER = {eer*100:.2f}%  thr = {thr:.4f}")

    aug_ids = [k for k, r in manifest.items() if r["augmented"] == "1"]
    base_s = np.array([clean_by_id[k]["score"] for k in aug_ids])
    aug_s = np.array([aug_by_id[k]["score"] for k in aug_ids])
    delta = aug_s - base_s

    W = wilcoxon(base_s, aug_s) if np.any(delta != 0) else None
    rb = rank_biserial_from_wilcoxon(delta)
    base_catch = float(np.mean(base_s >= thr))
    aug_catch = float(np.mean(aug_s >= thr))
    evasion_rate = float(np.mean(aug_s < thr))

    by_pos = {}
    for pos in ["start", "mid", "end"]:
        idx = [i for i, k in enumerate(aug_ids) if manifest[k]["position"] == pos]
        if idx:
            by_pos[pos] = float(np.mean(delta[idx]))

    insertion_res = {
        "dataset": "ASVspoof19 (AASIST, official pretrained)",
        "n_aug_fakes": len(aug_ids),
        "clean_eer_pct": round(eer * 100, 2),
        "thr_at_eer": round(thr, 4),
        "base_spoof_score_mean": round(float(base_s.mean()), 4),
        "aug_spoof_score_mean": round(float(aug_s.mean()), 4),
        "delta_mean": round(float(delta.mean()), 4),
        "delta_median": round(float(np.median(delta)), 4),
        "wilcoxon_p": (round(float(W.pvalue), 6) if W else None),
        "rank_biserial": round(rb, 3),
        "base_catch_rate": round(base_catch, 3),
        "aug_catch_rate": round(aug_catch, 3),
        "evasion_rate_after": round(evasion_rate, 3),
        "delta_start": round(by_pos.get("start", float("nan")), 4) if "start" in by_pos else None,
        "delta_mid": round(by_pos.get("mid", float("nan")), 4) if "mid" in by_pos else None,
        "delta_end": round(by_pos.get("end", float("nan")), 4) if "end" in by_pos else None,
    }

    tables = LAUGHSMI / "rerun_2026" / "concerns" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    with open(tables / "c3_aasist_insertion.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(insertion_res.keys()))
        w.writeheader()
        w.writerow(insertion_res)
    print("\n=== AASIST insertion result ===")
    for k, v in insertion_res.items():
        print(f"  {k}: {v}")
    print(f"wrote {tables / 'c3_aasist_insertion.csv'}")

    # ---- worst-window defense (sliding 64600-sample windows, hop 1s, over full waveform) ----
    print("\n[c3_aasist] worst-window defense: scoring clean set with sliding windows ...")
    clean_win = score_dir_windowed(model, clean_dir, clean_meta)
    print("[c3_aasist] worst-window defense: scoring augmented-fakes with sliding windows ...")
    aug_win_meta = [r for r in aug_meta_all if Path(r["file"]).name in manifest and manifest[Path(r["file"]).name]["augmented"] == "1"]
    aug_win = score_dir_windowed(model, aug_dir, aug_win_meta)

    clean_win_by_id = {r["file"]: r for r in clean_win}

    print(f"\n{'agg':<6}{'cleanEER':>10}{'thr':>8}{'baseCatch':>11}{'augCatch':>10}{'evasion':>9}")
    defense_results = []
    for agg in ["mean", "max", "p90"]:
        sc2 = np.array([r[agg] for r in clean_win])
        yl2 = np.array([0 if r["label"] == "bona-fide" else 1 for r in clean_win])
        thr2, eer2 = eer_thr(sc2, yl2)
        aug_ids2 = [r["file"] for r in aug_win]
        base_fake = np.array([clean_win_by_id[i][agg] for i in aug_ids2])
        aug_fake = np.array([r[agg] for r in aug_win])
        base_catch2 = float((base_fake >= thr2).mean())
        aug_catch2 = float((aug_fake >= thr2).mean())
        evasion2 = float((aug_fake < thr2).mean())
        print(f"{agg:<6}{eer2*100:>9.2f}%{thr2:>8.3f}{base_catch2:>11.2f}{aug_catch2:>10.2f}{evasion2:>9.2f}")
        defense_results.append({
            "agg": agg, "clean_eer_pct": round(eer2 * 100, 2), "thr": round(thr2, 4),
            "base_catch": round(base_catch2, 3), "aug_catch": round(aug_catch2, 3),
            "evasion_rate": round(evasion2, 3),
        })

    with open(tables / "c3_aasist_defense.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(defense_results[0].keys()))
        w.writeheader()
        w.writerows(defense_results)
    print(f"\nwrote {tables / 'c3_aasist_defense.csv'}")


if __name__ == "__main__":
    main()
