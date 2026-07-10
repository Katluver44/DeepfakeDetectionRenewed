"""D1 explanation, cont. — how is laughter different from words (speech) in
WavLM space? Reuses the D3 mean-embedding extraction (embeddings/d3/*.npy +
embeddings/d3_features.parquet), which has laugh-real, laugh-bark, speech-real.

Quantifies:
  - centroid cosine distance: laughter vs speech, at L9 and L12
  - within-group vs between-group cosine spread (is laughter its own island?)
  - linear-probe AUC laughter-vs-speech (should be ~1.0 => trivially separable)
  - C/T geometry contrast laughter vs speech
This supports the D1 mechanism: laughter frames sit off the speech manifold the
detector's WavLM front-end associates with (synthetic) speech, so inserting them
contributes non-spoof-looking, out-of-distribution frames that dilute the pooled
score — the same as inserting any real non-speech / real speech.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

L = Path(__file__).resolve().parents[1]


def cos(a, b):
    a = a / (np.linalg.norm(a) + 1e-9); b = b / (np.linalg.norm(b) + 1e-9)
    return 1 - float(a @ b)


def mean_pairwise_cos(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    S = Xn @ Xn.T
    n = len(X)
    return float((1 - S)[np.triu_indices(n, 1)].mean())


def main():
    df = pd.read_parquet(L / "embeddings" / "d3_features.parquet")
    out = []
    for layer, emb_path in [("L9", "mean_emb_layer9.npy"), ("L12", "mean_emb_layer12.npy")]:
        emb = np.load(L / "embeddings" / "d3" / emb_path)
        g = df.group.to_numpy()
        ok = ~np.isnan(emb).any(axis=1)
        laugh = emb[(g == "laugh-real") & ok]
        speech = emb[(g == "speech-real") & ok]
        bark = emb[(g == "laugh-bark") & ok]

        c_laugh, c_speech, c_bark = laugh.mean(0), speech.mean(0), bark.mean(0)
        d_laugh_speech = cos(c_laugh, c_speech)
        d_bark_speech = cos(c_bark, c_speech)
        d_laugh_bark = cos(c_laugh, c_bark)
        within_laugh = mean_pairwise_cos(laugh)
        within_speech = mean_pairwise_cos(speech)

        # probe laughter(real) vs speech
        X = np.vstack([laugh, speech]); y = np.r_[np.ones(len(laugh)), np.zeros(len(speech))]
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        auc = cross_val_score(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc").mean()

        print(f"=== {layer} ===")
        print(f"  centroid cos-dist  laughter<->speech = {d_laugh_speech:.3f}")
        print(f"  centroid cos-dist  bark-laugh<->speech = {d_bark_speech:.3f}")
        print(f"  centroid cos-dist  real-laugh<->bark-laugh = {d_laugh_bark:.3f}")
        print(f"  within-group spread laughter={within_laugh:.3f} speech={within_speech:.3f}")
        print(f"  probe laughter-vs-speech AUC (5-fold) = {auc:.3f}")
        # C/T contrast
        sub = df[ok]
        for feat in ["C_layer12", "T_layer9"]:
            lv = sub[sub.group == "laugh-real"][feat].mean()
            sv = sub[sub.group == "speech-real"][feat].mean()
            print(f"  {feat}: laughter={lv:.3f} speech={sv:.3f}")
        out.append({"layer": layer, "d_laugh_speech": round(d_laugh_speech, 3),
                    "d_bark_speech": round(d_bark_speech, 3), "d_laugh_bark": round(d_laugh_bark, 3),
                    "within_laugh": round(within_laugh, 3), "within_speech": round(within_speech, 3),
                    "probe_laugh_vs_speech_auc": round(float(auc), 3)})
    with open(L / "tables" / "table_d1_laughter_vs_words.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print("\nwrote tables/table_d1_laughter_vs_words.csv")


if __name__ == "__main__":
    main()
