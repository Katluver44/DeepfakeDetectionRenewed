#!/usr/bin/env python3
"""
px_common.py — shared, audited infrastructure for the channel-robustness experiments
====================================================================================
P1 (test-time channel-axis projection), P2 (causal minimal augmentation), P3 (paired
invariance adapter), P4 (blind channel-aware threshold calibration).

Everything here is built to ONE invariant: the detector score is the *full-model*
decision logit  gm(x, num_frames, profiler=None, use_aug=False, stage="val")["logit"]
(NOT the encoder_and_GAT[5] intermediate — that bug is documented in memory). Sign
convention: higher logit = more "spoof" (BCEWithLogitsLoss, spoof label = 1).

Shared pieces:
  - load_detector()            -> Phoneme_GAT_lit.model (gm), on DEVICE, eval+frozen
  - detector_logits(gm, wavs)  -> np.array of full-model logits
  - CHANNELS / apply_channel   -> RMS-matched recording-channel degradations (E13)
  - load_itw_waves()           -> cached ITW waveforms (48000) + labels + speakers
  - load_mlaad_waves()         -> MLAAD test-split bona/spoof waveforms (48000)
  - compute_eer / metrics_at_threshold
  - AxisProjector              -> the test-time intervention hook (P1/P3 reuse)

Audited choices are flagged with  # AUDIT:  comments.
"""
from __future__ import annotations
import os, sys, json, warnings
from argparse import Namespace
from pathlib import Path
import numpy as np
import torch
from scipy import signal
from sklearn.metrics import roc_curve

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).resolve().parents[2]
EXP_DIR  = Path(__file__).resolve().parents[1]
SCRIPTS  = Path(__file__).resolve().parent
for _p in (str(BASE), str(EXP_DIR), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PROC_DIR  = EXP_DIR / "data" / "mlaad_tiny_processed"
TEST_JSON = EXP_DIR / "results" / "mlaad" / "baseline_eval" / "test_in_distribution.json"
CKPT      = EXP_DIR / "checkpoints" / "mlaad_robust_goat.ckpt"
ITW_CACHE = BASE / "data" / "in_the_wild"
META_CSV  = (ITW_CACHE / "downloads" / "extracted" /
             "c3c93f2f54ac2d261fa7010629351505bd6e05597ea22fd4a35c92dda590a3bf" /
             "release_in_the_wild" / "meta.csv")
WAVE_CACHE = BASE / "outputs" / "px_wave_cache"
WAVE_CACHE.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR, TLEN = 16000, 48000
NF_PER   = TLEN // 320 - 1
BATCH    = 16
SEED     = 42

# torch 2.6 weights_only default — trusted local checkpoints contain pandas/Namespace globals
_torch_load = torch.load
def _trusted_load(*a, **k):
    k.setdefault("weights_only", False); return _torch_load(*a, **k)
torch.load = _trusted_load


# ── basic dsp ─────────────────────────────────────────────────────────────────
def _fix(y):
    y = np.asarray(y, np.float32)
    if len(y) < TLEN: y = np.pad(y, (0, TLEN - len(y)))
    return y[:TLEN]

def rms(x): return float(np.sqrt(np.mean(x**2)) + 1e-12)
def match_rms(y, ref): return (y * (rms(ref) / rms(y))).astype(np.float32)
def cosv(a, b): return float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))


# ── channel degradations (identical to E13; RMS-matched, length-TLEN) ──────────
def deg_telephone(x, sr=SR):
    sos = signal.butter(6, [300, 3400], btype="band", fs=sr, output="sos")
    return signal.sosfilt(sos, x)

def deg_lowpass4k(x, sr=SR):
    sos = signal.butter(8, 3400, btype="low", fs=sr, output="sos")
    return signal.sosfilt(sos, x)

def deg_mulaw(x, sr=SR):
    mu = 255.0
    m = np.max(np.abs(x)) + 1e-9; xn = x / m
    comp = np.sign(xn) * np.log1p(mu*np.abs(xn)) / np.log1p(mu)
    q = np.round((comp*0.5 + 0.5) * 255) / 255 * 2 - 1
    exp = np.sign(q) * (1/mu) * ((1+mu)**np.abs(q) - 1)
    return exp * m

def deg_reverb(x, sr=SR, t60=0.35, decay=0.08, rng=None):
    rng = rng or np.random.default_rng()
    t = np.arange(int(t60*sr)); rir = np.exp(-t/(decay*sr)) * rng.standard_normal(len(t))
    rir[0] = 1.0; rir /= np.sqrt(np.sum(rir**2))
    return signal.fftconvolve(x, rir)[:len(x)]

_mp3 = None
def deg_mp3(x, sr=SR, bitrate=16):
    global _mp3
    if _mp3 is None:
        from audiomentations import Mp3Compression
        _mp3 = Mp3Compression(min_bitrate=bitrate, max_bitrate=bitrate, p=1.0)
    return _mp3(samples=np.asarray(x, np.float32), sample_rate=sr)

def deg_wild(x, sr=SR, rng=None):
    return deg_reverb(deg_mp3(deg_telephone(x, sr), sr), sr, rng=rng)

CHANNELS = {"clean": lambda x: x, "telephone": deg_telephone, "lowpass3.4k": deg_lowpass4k,
            "mulaw8bit": deg_mulaw, "mp3_16k": deg_mp3, "reverb": deg_reverb, "wild_chain": deg_wild}

def apply_channel(x, fn, rng=None):
    try:
        y = fn(x, rng=rng) if "rng" in getattr(fn, "__code__", _fix.__code__).co_varnames else fn(x)
        y = _fix(y)
    except Exception as e:
        print(f"    [channel error] {e}"); y = _fix(x)
    return match_rms(y, x)


# ── detector ──────────────────────────────────────────────────────────────────
def _patch_phoneme_loader():
    import phoneme_GAT.modules as mm, phoneme_GAT.phoneme_model as pm
    from phoneme_GAT.phoneme_model import BaseModule, network_param, optim_param
    def _load(network_name="wavlm", pretrained_path=None, total_num_phonemes=198):
        network_param.network_name = network_name
        network_param.pretrained_name = "microsoft/wavlm-base"
        network_param.vocab_size = total_num_phonemes
        if pretrained_path and Path(pretrained_path).exists():
            return BaseModule.load_from_checkpoint(str(pretrained_path), network_param=network_param,
                optim_param=optim_param, tokenizer=None, total_num_phonemes=total_num_phonemes,
                weights_only=False).cpu()
        return BaseModule(network_param, optim_param, tokenizer=None, total_num_phonemes=total_num_phonemes)
    pm.load_phoneme_model = _load; mm.load_phoneme_model = _load

def load_detector(ckpt=CKPT, device=DEVICE):
    """Returns the inner Phoneme_GAT model (gm), eval+frozen, on device."""
    _patch_phoneme_loader()
    try:
        from pandas import Series as _PS
        from ay2.tools.text._phonemes import Phonemer_Tokenizer_Recombination as _PTR
        torch.serialization.add_safe_globals([Namespace, _PS, _PTR])
    except Exception:
        torch.serialization.add_safe_globals([Namespace])
    from phoneme_GAT.modules import Phoneme_GAT_lit
    ck = torch.load(str(ckpt), weights_only=False)
    cfgh = ck.get("hyper_parameters", {}).get("cfg", None)
    n_edges = getattr(getattr(cfgh, "PhonemeGAT", None), "n_edges", 10) if cfgh else 10
    cfg = Namespace(PhonemeGAT=Namespace(backbone="wavlm", use_raw=False, use_GAT=True,
                    n_edges=int(n_edges), use_aug=True, use_pool=True, use_clip=True))
    lit = Phoneme_GAT_lit.load_from_checkpoint(str(ckpt), cfg=cfg, map_location=device, strict=True)
    lit.to(device); lit.eval(); lit.freeze()
    return lit.model

@torch.no_grad()
def detector_logits(gm, wavs, device=DEVICE, batch=BATCH):
    """Full-model decision logit per waveform. wavs: list/array of (TLEN,) float32."""
    out = []
    for i in range(0, len(wavs), batch):
        xb = torch.as_tensor(np.stack([_fix(w) for w in wavs[i:i+batch]]),
                             dtype=torch.float32, device=device)
        nf = torch.full((xb.shape[0],), NF_PER, device=device)
        lg = gm(xb, nf, profiler=None, use_aug=False, stage="val")["logit"]
        out.extend(lg.detach().cpu().float().numpy().ravel().tolist())
    return np.asarray(out)


# ── test-time channel-axis projection hook (P1 / P3) ──────────────────────────
class AxisProjector:
    """
    Projects out (or captures) a channel direction in the detector's *trainable-encoder
    output* frame space — the exact (B,T,768) tensor that feeds phoneme-pool -> GAT ->
    cls_head (modules.py:472). Gated by gm._tap_on so it only fires inside encoder_and_GAT
    (NOT the frozen phoneme-id path that shares the same encoder module, modules.py:576).
    """
    def __init__(self, gm):
        self.gm = gm
        self.v = None            # (768,) unit channel axis (torch, on device)
        self.alpha = 0.0
        self.capture = False
        self._buf = []
        gm._tap_on = False
        # wrap encoder_and_GAT to toggle the gate around the trainable-encoder call
        import types
        orig = gm.encoder_and_GAT.__func__
        def wrapped(self_inner, *a, **k):
            self_inner._tap_on = True
            try:
                return orig(self_inner, *a, **k)
            finally:
                self_inner._tap_on = False
        gm.encoder_and_GAT = types.MethodType(wrapped, gm)
        self._handle = gm.encoder.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        if not getattr(self.gm, "_tap_on", False):
            return None
        h = out[0]                                   # (B,T,768)
        if self.capture:
            self._buf.append(h.detach().float().mean(1).cpu().numpy())   # per-utt mean frame
            return None
        if self.v is None or self.alpha == 0.0:
            return None
        v = self.v.to(h.dtype)
        proj = torch.einsum("btd,d->bt", h, v).unsqueeze(-1) * v          # (B,T,768)
        h2 = h - self.alpha * proj
        return (h2,) + tuple(out[1:])

    def set_axis(self, v_np, alpha=1.0):
        v = np.asarray(v_np, np.float32); v = v / (np.linalg.norm(v) + 1e-12)
        self.v = torch.as_tensor(v, device=DEVICE); self.alpha = float(alpha)

    def start_capture(self):
        self.capture = True; self._buf = []
    def collect(self):
        self.capture = False
        arr = np.concatenate(self._buf, 0) if self._buf else np.zeros((0, 768))
        self._buf = []
        return arr
    def disable(self):
        self.alpha = 0.0; self.v = None; self.capture = False
    def remove(self):
        self._handle.remove()


# ── frozen pretrained WavLM (analysis encoder) ────────────────────────────────
def load_frozen_wavlm(device=DEVICE):
    """
    WavLMModel with the pos_conv weight-norm parametrization REPAIRED.
    transformers 4.36 + torch 2.6 silently leaves
    encoder.pos_conv_embed.conv.parametrizations.weight.original{0,1} randomly
    initialized (checkpoint stores legacy weight_g/weight_v). We copy them in
    explicitly; shapes verified identical (g:(1,1,128), v:(768,48,128)).
    """
    from transformers import WavLMModel
    from transformers.utils import cached_file
    torch.manual_seed(0)
    wl = WavLMModel.from_pretrained("microsoft/wavlm-base")
    sd = _torch_load(cached_file("microsoft/wavlm-base", "pytorch_model.bin"),
                     map_location="cpu", weights_only=False)
    g = sd.get("encoder.pos_conv_embed.conv.weight_g")
    v = sd.get("encoder.pos_conv_embed.conv.weight_v")
    conv = wl.encoder.pos_conv_embed.conv
    if g is not None and hasattr(conv, "parametrizations"):
        with torch.no_grad():
            conv.parametrizations.weight.original0.copy_(g)
            conv.parametrizations.weight.original1.copy_(v)
        print("[px_common] repaired WavLM pos_conv weight-norm weights from checkpoint")
    return wl.to(device).eval()


# ── metrics ───────────────────────────────────────────────────────────────────
def compute_eer(y, s):
    """y in {0=bona,1=spoof}, s=score (higher=spoof). Returns (eer, thr_at_eer)."""
    fpr, tpr, thr = roc_curve(y, s, pos_label=1)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])

def metrics_at_threshold(y, s, thr):
    y = np.asarray(y); s = np.asarray(s)
    pred = (s >= thr).astype(int)
    bona = y == 0; spoof = y == 1
    fpr = float(pred[bona].mean()) if bona.any() else np.nan      # bona->spoof (false positive)
    fnr = float((1 - pred[spoof]).mean()) if spoof.any() else np.nan  # spoof->bona (miss)
    acc = float((pred == y).mean())
    bal = float(0.5*((1-fpr) + (1-fnr)))
    return {"FPR": fpr, "FNR": fnr, "acc": acc, "bal_acc": bal}


# ── data caching ──────────────────────────────────────────────────────────────
def _center_crop_pad(wav, tlen=TLEN):
    wav = np.asarray(wav, np.float32)
    if len(wav) >= tlen:
        s = (len(wav) - tlen) // 2
        return wav[s:s+tlen]
    return np.pad(wav, (0, tlen - len(wav)))

def load_itw_waves(n_per_class=3000, seed=SEED, force=False):
    """
    Cached balanced ITW waveforms (same selection convention as E5: SEED, n_per_class,
    sorted indices). Returns dict: waves (N,TLEN) float32, labels (N,) {0,1}, speakers,
    names. Order = sorted HF index (mixes classes); use labels to split.
    """
    cache = WAVE_CACHE / f"itw_waves_n{n_per_class}_seed{seed}.npz"
    if cache.exists() and not force:
        z = np.load(cache, allow_pickle=True)
        return {"waves": z["waves"].astype(np.float32), "labels": z["labels"],
                "speakers": z["speakers"], "names": z["names"]}
    import pandas as pd
    from datasets import load_dataset, Audio as HFAudio
    assert META_CSV.exists(), f"meta.csv not found at {META_CSV}"
    meta = pd.read_csv(META_CSV); meta.columns = [c.strip().lower() for c in meta.columns]
    file_col  = next(c for c in meta.columns if "file" in c or "path" in c or "name" in c)
    spk_col   = next((c for c in meta.columns if "speaker" in c), None)
    label_col = next(c for c in meta.columns if c not in (file_col, spk_col))
    meta["_b"] = meta[file_col].apply(lambda p: os.path.splitext(os.path.basename(str(p)))[0])
    b2l = dict(zip(meta["_b"], meta[label_col])); b2s = dict(zip(meta["_b"], meta[spk_col])) if spk_col else {}
    ds = load_dataset("mueller91/In-The-Wild", cache_dir=str(ITW_CACHE),
                      token=os.environ.get("HF_TOKEN"))
    ds = ds[list(ds.keys())[0]]
    nod = ds.cast_column("audio", HFAudio(decode=False))
    bn = [os.path.splitext(os.path.basename(nod[i]["audio"]["path"]))[0] for i in range(len(nod))]
    REAL = {"bona-fide","bonafide","real","genuine","0",0}; FAKE = {"spoof","fake","synthetic","1",1}
    def l2i(v):
        if v is None: return None
        vl = str(v).lower().strip()
        return 0 if vl in REAL else (1 if vl in FAKE else None)
    labs = [l2i(b2l.get(b)) for b in bn]; spks = [str(b2s.get(b,"unknown")) for b in bn]
    rng = np.random.default_rng(seed)
    ri = [i for i,l in enumerate(labs) if l==0]; fi = [i for i,l in enumerate(labs) if l==1]
    nr, nf = min(n_per_class,len(ri)), min(n_per_class,len(fi))
    sel = sorted(rng.choice(ri,nr,replace=False).tolist() + rng.choice(fi,nf,replace=False).tolist())
    dsb = ds.select(sel).cast_column("audio", HFAudio(sampling_rate=SR))
    waves = np.zeros((len(sel), TLEN), np.float16)
    for j in range(len(sel)):
        waves[j] = _center_crop_pad(dsb[j]["audio"]["array"]).astype(np.float16)
        if (j+1) % 500 == 0: print(f"    ITW cache {j+1}/{len(sel)}", flush=True)
    out = {"waves": waves, "labels": np.array([labs[i] for i in sel]),
           "speakers": np.array([spks[i] for i in sel]), "names": np.array([bn[i] for i in sel])}
    np.savez_compressed(cache, **out)
    return {"waves": out["waves"].astype(np.float32), "labels": out["labels"],
            "speakers": out["speakers"], "names": out["names"]}

def load_mlaad_waves(n_per_class=600, seed=SEED, force=False):
    """Cached MLAAD test-split waveforms (.pt tensors). dict: bona (Nb,TLEN), spoof (Ns,TLEN)."""
    cache = WAVE_CACHE / f"mlaad_waves_n{n_per_class}_seed{seed}.npz"
    if cache.exists() and not force:
        z = np.load(cache, allow_pickle=True)
        return {"bona": z["bona"].astype(np.float32), "spoof": z["spoof"].astype(np.float32)}
    recs = json.loads(TEST_JSON.read_text())
    def _wav(r): return _fix(np.asarray(torch.load(PROC_DIR / r["audio_path"]), np.float32))
    bona  = [r for r in recs if not str(r["label"]).lower().startswith("spoof")]
    spoof = [r for r in recs if str(r["label"]).lower().startswith("spoof")]
    rng = np.random.default_rng(seed); rng.shuffle(bona); rng.shuffle(spoof)
    def grab(rs):
        out = []
        for r in rs:
            if len(out) >= n_per_class: break
            try: out.append(_wav(r))
            except Exception: pass
        return np.stack(out).astype(np.float16)
    out = {"bona": grab(bona), "spoof": grab(spoof)}
    np.savez_compressed(cache, **out)
    return {"bona": out["bona"].astype(np.float32), "spoof": out["spoof"].astype(np.float32)}


if __name__ == "__main__":
    # smoke test / build caches
    print(f"[px_common] DEVICE={DEVICE}")
    mw = load_mlaad_waves(); print(f"  MLAAD bona={mw['bona'].shape} spoof={mw['spoof'].shape}")
    iw = load_itw_waves(); print(f"  ITW waves={iw['waves'].shape} "
                                 f"bona={(iw['labels']==0).sum()} spoof={(iw['labels']==1).sum()}")
    gm = load_detector(); print("  detector loaded")
    lb = detector_logits(gm, mw['bona'][:32]); ls = detector_logits(gm, mw['spoof'][:32])
    print(f"  MLAAD logit bona med={np.median(lb):+.2f} spoof med={np.median(ls):+.2f}")
