"""Consolidate all synthesis methods into the LaughFake dataset + rebuild a
multi-method D3 inventory (real vs each synthetic method) so the SSL analysis
spans every technique, not just Bark.

Scans: data/laugh_bank_bark (+bark_probe laughter_only/speech_laugh) and every
data/laugh_oss/<method>/ dir (each with a manifest.csv). Writes:
  data/laughter_dataset/manifest.csv        (all methods, unified schema)
  embeddings/d3_multi_inventory.csv         (laugh-real + laugh-<method> groups
                                             + speech-real anchor)
"""
from __future__ import annotations
import csv, random, shutil
from pathlib import Path
import soundfile as sf

L = Path(__file__).resolve().parents[1]
DS = L / "data" / "laughter_dataset"


def dur(p):
    w, sr = sf.read(str(p)); return len(w) / sr, sr


def main():
    rng = random.Random(20260710)
    rows = []
    (DS / "all").mkdir(parents=True, exist_ok=True)

    # Bark (laughter-only bank + probe laughter_only + probe speech_laugh)
    bark_srcs = [
        (L/"data"/"laugh_bank_bark", "bark", "bark_laughter_token", "laughter_only"),
        (L/"data"/"bark_probe"/"laughter_only", "bark", "bark_laughter_token", "laughter_only"),
        (L/"data"/"bark_probe"/"speech_laugh", "bark", "bark_laughs_inline", "speech_with_laugh"),
    ]
    for d, src, method, cat in bark_srcs:
        for p in sorted(Path(d).glob("*.wav")):
            du, sr = dur(p)
            rows.append({"file": str(p.resolve()), "source": src, "method": method,
                         "category": cat, "dur_s": round(du, 3), "sr": 16000})

    # every OSS method dir
    oss = L / "data" / "laugh_oss"
    if oss.exists():
        for md in sorted(oss.iterdir()):
            if not md.is_dir():
                continue
            man = md / "manifest.csv"
            method_name = md.name
            wavs = sorted(md.glob("*.wav"))
            for p in wavs:
                du, sr = dur(p)
                rows.append({"file": str(p.resolve()), "source": "oss", "method": method_name,
                             "category": "laughter_only", "dur_s": round(du, 3), "sr": 16000})

    with open(DS / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "source", "method", "category", "dur_s", "sr"])
        w.writeheader(); w.writerows(rows)

    from collections import Counter
    bym = Counter(r["method"] for r in rows)
    total_min = sum(r["dur_s"] for r in rows) / 60
    print(f"LaughFake dataset: {len(rows)} clips, {total_min:.1f} min, methods:")
    for m, n in bym.most_common():
        print(f"  {m}: {n}")

    # ---- multi-method D3 inventory ----
    inv = []
    # real laughter (VocalSound) 300
    vs = list(csv.DictReader(open(L/"data"/"vocalsound"/"laughter_list.csv")))
    rng.shuffle(vs)
    for r in vs[:300]:
        inv.append({"group": "laugh-real", "file_path": str(L/"data"/"vocalsound"/r["file"]),
                    "start_s": 0, "end_s": -1, "pair_id": "", "speaker": r["speaker_id"]})
    # speech anchor 150
    lib = [r for r in csv.DictReader(open(L/"data"/"eval_mlaad"/"meta.csv")) if r["label"] == "bona-fide"][:150]
    for r in lib:
        inv.append({"group": "speech-real", "file_path": str(L/"data"/"eval_mlaad"/r["file"]),
                    "start_s": 0, "end_s": -1, "pair_id": "", "speaker": r["speaker"]})
    # each synthetic method as its own group (cap 100/method for balance)
    by_method = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r["file"])
    for method, files in by_method.items():
        rng.shuffle(files)
        for fp in files[:100]:
            inv.append({"group": f"laugh-{method}", "file_path": fp,
                        "start_s": 0, "end_s": -1, "pair_id": "", "speaker": method})
    with open(L/"embeddings"/"d3_multi_inventory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "file_path", "start_s", "end_s", "pair_id", "speaker"])
        w.writeheader(); w.writerows(inv)
    print(f"\nwrote embeddings/d3_multi_inventory.csv ({len(inv)} rows, "
          f"{len(set(r['group'] for r in inv))} groups)")


if __name__ == "__main__":
    main()
