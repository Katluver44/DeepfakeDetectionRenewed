"""Explain the D1 evasion effect. Two probes on the ASV19 augmented fakes:

(1) DOSE-RESPONSE: Δscore (aug-base) vs. the fraction of the model's 3s center
    crop that is occupied by the inserted laughter. The scorer center-crops to
    48000 samples; compute how much of that crop is laughter given insert
    position/duration, and correlate with Δscore.

(2) SPEECH-INSERT CONTROL: re-augment the same fakes but splice in a random
    *bona-fide LibriSpeech* speech clip (not laughter) at the same positions,
    re-score. If speech insertion ALSO drops the spoof score similarly, the
    mechanism is "any inserted non-synthetic (real-audio) segment dilutes the
    continuous-synthetic-texture evidence" and laughter is just a natural
    carrier. If laughter drops it MORE than speech, laughter is special.
"""
from __future__ import annotations
import csv, random, sys
from pathlib import Path
import numpy as np
import soundfile as sf

L = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(L / "scripts"))
from score_itw_detector import load_detector, crop_or_tile_center, score_batch, load_wav_mono_16k

SR = 16000
CROP = 3 * SR


def laughter_fraction_in_crop(host_len, ins_len, position):
    """Fraction of the center 3s crop occupied by the inserted segment."""
    total = host_len + ins_len
    if position == "start":
        ins_s, ins_e = 0, ins_len
    elif position == "end":
        ins_s, ins_e = total - ins_len, total
    else:
        mid = host_len // 2
        ins_s, ins_e = mid, mid + ins_len
    if total <= CROP:
        # tiled/padded — whole signal present, fraction ~ ins/total
        return ins_len / total
    cs = (total - CROP) // 2
    ce = cs + CROP
    overlap = max(0, min(ce, ins_e) - max(cs, ins_s))
    return overlap / CROP


def main():
    device = "cuda"
    lit = load_detector(str(L / "models" if False else L.parent / "models" / "asv19-wavlm-gat-full.ckpt"), device)
    base = {Path(r["file_id"]).name: float(r["score"]) for r in csv.DictReader(open(L/"detector_out"/"base_asv19.csv"))}
    aug = {Path(r["file_id"]).name: float(r["score"]) for r in csv.DictReader(open(L/"detector_out"/"aug_asv19.csv"))}
    man = [r for r in csv.DictReader(open(L/"data"/"eval_asv19_aug"/"manifest.csv")) if r["augmented"] == "1"]

    # (1) dose-response
    fracs, deltas = [], []
    for r in man:
        k = Path(r["file"]).name
        host_len = int(round(float(r["host_dur_s"]) * SR))
        ins_len = int(round(float(r["insert_dur_s"]) * SR))
        f = laughter_fraction_in_crop(host_len, ins_len, r["position"])
        fracs.append(f); deltas.append(aug[k] - base[k])
    fracs, deltas = np.array(fracs), np.array(deltas)
    from scipy.stats import spearmanr, pearsonr
    rho, prho = spearmanr(fracs, deltas)
    print("=== (1) DOSE-RESPONSE: laughter fraction of 3s crop vs Δscore ===")
    print(f"  Spearman rho={rho:.3f} (p={prho:.2e}), Pearson r={pearsonr(fracs,deltas)[0]:.3f}")
    for lo, hi in [(0, .25), (.25, .5), (.5, .75), (.75, 1.01)]:
        m = (fracs >= lo) & (fracs < hi)
        if m.sum():
            print(f"  crop-laughter {int(lo*100)}-{int(hi*100)}%: n={m.sum():3d} meanΔ={deltas[m].mean():+.3f}")

    # (2) speech-insert control: splice a random LibriSpeech bona clip instead
    rng = random.Random(20260710)
    lib = [r for r in csv.DictReader(open(L/"data"/"eval_mlaad"/"meta.csv")) if r["label"] == "bona-fide"]
    lib_wavs = [load_wav_mono_16k(str(L/"data"/"eval_mlaad"/r["file"])) for r in rng.sample(lib, 40)]

    def splice(host, ins, pos):
        if pos == "start": return np.concatenate([ins, host])
        if pos == "end": return np.concatenate([host, ins])
        mid = len(host)//2; return np.concatenate([host[:mid], ins, host[mid:]])

    wavs, ids = [], []
    for r in man:
        host = load_wav_mono_16k(str(L/"data"/"eval_asv19"/r["file"]))
        ins = rng.choice(lib_wavs)
        # match insert length to the laughter insert length used for this file
        ins_len = int(round(float(r["insert_dur_s"]) * SR))
        ins = np.tile(ins, -(-ins_len//len(ins)))[:ins_len] if len(ins) < ins_len else ins[:ins_len]
        # peak-norm then level-match to host rms
        ins = ins/(np.max(np.abs(ins))+1e-9) * (np.sqrt(np.mean(host**2))+1e-9)
        w = splice(host, ins.astype(np.float32), r["position"])
        wavs.append(crop_or_tile_center(np.clip(w,-1,1).astype(np.float32)))
        ids.append(Path(r["file"]).name)

    sp = []
    for i in range(0, len(wavs), 16):
        sp.extend(p for _, p in score_batch(lit, wavs[i:i+16], device))
    sp = np.array(sp)
    b = np.array([base[i] for i in ids]); a = np.array([aug[i] for i in ids])
    print("\n=== (2) SPEECH-INSERT CONTROL (same fakes, real speech spliced in) ===")
    print(f"  base spoof-score      = {b.mean():.3f}")
    print(f"  +laughter             = {a.mean():.3f}  (Δ {a.mean()-b.mean():+.3f})")
    print(f"  +real-speech          = {sp.mean():.3f}  (Δ {sp.mean()-b.mean():+.3f})")
    print("  interp: if speech-insert Δ ≈ laughter Δ -> effect is 'insert any real audio';")
    print("          if laughter Δ >> speech Δ -> laughter specifically is the stronger evasion carrier")


if __name__ == "__main__":
    main()
