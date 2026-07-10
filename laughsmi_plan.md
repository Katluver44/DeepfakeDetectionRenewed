# LaugHSMI 2026 — Laughter & Audio Deepfake Detection: Agent-Executable Research Plan

**Target:** LaugHSMI Workshop @ ICMI 2026 (Naples). Submission via OpenReview, double-blind.
**Deadline:** July 10, 2026, 23:59 AoE (TOMORROW). Camera-ready Aug 1 — polish can happen post-acceptance.
**Format:** SHORT PAPER — 4 pages excluding references, ACM ICMI template, anonymized. Do NOT attempt the 8-page format; short papers explicitly allow work-in-progress.
**Paper working title:** "Does Laughter Fool Deepfake Detectors? Genuine vs. Synthetic Laughter in Self-Supervised Speech Representations"

---

## 0. Framing rules (read before doing anything)

The workshop is an ICMI affective-computing/HCI venue, NOT an anti-spoofing venue. Every result must be framed as **laughter analysis** first, security second:

- Say: "genuine vs. synthetic laughter," "laughter as a social signal whose authenticity matters," "spontaneous vs. generated laughter."
- Avoid leading with: EER tables, ASVspoof jargon, countermeasure terminology. These appear in Results but do not headline the abstract.
- Explicit CFP hooks to cite in the intro: (a) automatic detection of laughter in multimodal data, (b) acoustic features of laughter, (c) spontaneous vs. posed laughter, (d) datasets/annotation, (e) ethical considerations (deepfake laughter = new ethics angle).
- Double-blind: no author names, no "our previous geometric hardness paper (citation to self)" phrasing — cite the geometric work in third person if it is public; if unpublished, describe the C/T features inline as if introduced here.

## 1. Pre-registered hypotheses (write into paper §1)

- **H1 (prevalence asymmetry):** In the In-the-Wild (ITW) dataset, laughter occurs at a substantially higher rate in bona fide audio than in spoofed audio → laughter presence is a spurious shortcut cue AND a future blind spot.
- **H2 (geometric distinctness):** Laughter segments in WavLM embedding space have higher trajectory irregularity (velocity entropy, layer 9) and lower compactness (higher radius of gyration, layer 12) than fluent speech from the same speakers.
- **H3 (conditional detector behavior):** Detector performance/score distributions differ on laughter-containing vs. laughter-free utterances, and frame-level scores diverge over laughter regions.
- **H4 (synthetic laughter probe, optional):** TTS-generated laughter ([laughs] tokens / voice clones) lands off the genuine-laughter manifold in WavLM space, yet current detectors [do / do not] flag it.

Each stage below is independently publishable. If time runs out, ship whatever stages are complete.

---

## 2. Environment & repo setup (30 min)

```bash
mkdir -p laughsmi/{data,detector_out,embeddings,figures,tables,paper}
cd laughsmi
python -m venv venv && source venv/bin/activate
pip install torch torchaudio transformers librosa pandas numpy scipy scikit-learn umap-learn matplotlib soundfile tqdm tgt
git clone https://github.com/jrgillick/laughter-detection.git
pip install -r laughter-detection/requirements.txt   # NOTE: old pins; if conflicts, install tensorboardX + pyloudnorm manually and skip strict pins
```

Known pitfalls:
- jrgillick repo is Python 3.6-era. If librosa API errors (`librosa.output` removed), patch to `soundfile.write`. If torch checkpoint load fails, add `map_location='cpu'` and `weights_only=False`.
- The pretrained checkpoint is in the repo (`checkpoints/in_use/resnet_with_augmentation`). Verify it exists after clone; if not, download link is in the repo README / Colab notebook.
- GPU strongly preferred for WavLM extraction; laughter detector runs fine on CPU (~1–2 s per 10 s audio).

## 3. Data acquisition (parallel with §2)

### 3.1 In-the-Wild (Müller et al. 2022) — REQUIRED
- Source: https://deepfake-total.com/in_the_wild (Fraunhofer AISEC) or the Hugging Face mirror. ~38k utterances (~20k bona fide, ~12k spoof), celebrities, podcast/interview provenance. Comes with `meta.csv` (file, speaker, label).
- Verify: count files, confirm label column values {bona-fide, spoof}.

### 3.2 Genuine laughter corpus — REQUIRED (pick first available, in order)
1. **VocalSound** (preferred: direct public download, no registration): https://github.com/YuanGongND/vocalsound — 21,024 clips of laughter/sighs/coughs/etc. from 3,365 speakers. Use ONLY the laughter class (~3.4k clips). This is the "genuine laughter" anchor set.
2. **AudioSet laughter subset** (fallback): labels via jrgillick repo annotations; requires YouTube downloads — SKIP unless VocalSound is unreachable.
3. **MAHNOB / Haha-Pod** (fallback): require registration/request — SKIP for this deadline.
- If the user supplies "MuLaugh" (unverified name; possibly a workshop-affiliated corpus), slot it here as the genuine-laughter anchor and/or synthetic-laughter source; the pipeline is corpus-agnostic.

### 3.3 MLAAD — OPTIONAL (only for §6.3 spoof-side laughter scan)
- Full MLAAD is huge. Download only a stratified sample: ≤50 files per TTS system, English subset first. If bandwidth is a problem, SKIP — ITW spoof files already give the spoof-side laughter rate for H1.

### 3.4 Synthetic laughter probe set — CONDITIONAL (Decision D2)
- Generate with **Bark** (suno-ai/bark, local, supports `[laughter]`/`[laughs]` tokens):
```python
from bark import SAMPLE_RATE, generate_audio, preload_models
preload_models()
# 3 conditions x 20 prompts x N speaker presets ≈ 60-120 clips:
#  (a) speech-only text, (b) same text + [laughs] mid-utterance, (c) laughter-only "[laughter]"
```
- Target: 60–120 clips, 16 kHz mono. Label as spoof-laugh. Frame in paper as a "pilot probe," not a dataset contribution.
- Fallback if Bark fails: any locally runnable TTS with non-verbal tokens; if none works within 1 hour, drop H4 and note as future work.

---

## 4. Stage 1 — Laughter detection over ITW (2 h wall-clock; start FIRST, runs unattended)

Batch-run the Gillick detector over ALL ITW files at threshold 0.5, min_length 0.2 s; also record per-file max frame probability so 0.3/0.7 operating points can be derived WITHOUT re-running:

```bash
python scripts/run_laughter_batch.py \
  --audio_dir data/in_the_wild --meta data/in_the_wild/meta.csv \
  --threshold 0.5 --min_length 0.2 --out detector_out/itw_laughter.csv
```

`itw_laughter.csv` schema (one row per file):
`file_id,label,speaker,dur_s,n_laugh_segs,laugh_dur_s,laugh_ratio,max_prob,seg_starts,seg_ends`

**Decision D1 — does ITW contain laughter?**
- Let B = # bona fide files with ≥1 segment at thr 0.5, S = same for spoof.
- **D1a (expected): B ≥ 100.** Proceed with full plan (Stages 2–4 on ITW laughter + VocalSound).
- **D1b: 20 ≤ B < 100.** Proceed, but report bootstrap CIs everywhere; merge H3's utterance-level EER split into score-distribution analysis only (EER on <100 files is unstable).
- **D1c: B < 20.** ITW laughter analysis is descriptive only ("laughter is essentially absent from the benchmark" — itself a finding). Pivot the paper's empirical core to: VocalSound genuine laughter vs. Bark synthetic laughter vs. ITW speech, i.e., H2 + H4 become the main results. This is the user's stated fallback and it is fully executable.

**QC (mandatory, 20 min):** listen to 20 random detected segments (10 bona, 10 spoof). Record precision estimate. Known false alarms: applause, crowd noise, music stings — ITW is noisy. Report precision in paper §Limitations.

**Deliverable — Table 1:** laughter prevalence by class at thr ∈ {0.3, 0.5, 0.7}: % files with laughter, mean laugh_ratio, Fisher exact test bona vs. spoof, odds ratio + CI.

## 5. Stage 2 — WavLM geometry of laughter vs. speech (3–4 h)

### 5.1 Segment inventory
- **laugh-bona:** ITW bona fide laughter segments (from Stage 1) — pad to ≥0.5 s.
- **speech-bona (paired):** for each laugh segment, a same-file, same-duration speech segment ≥1 s away from any laughter (controls speaker + channel — this pairing is the key confound control).
- **laugh-vs (external):** 300 random VocalSound laughter clips.
- **speech-spoof:** 300 random ITW spoof segments (duration-matched).
- **laugh-spoof:** any spoof-side detections from Stage 1 + Bark probe clips if D2 executed.

### 5.2 Features (reuse existing geometric-hardness code)
Extract WavLM-Large frame embeddings, layers 9 and 12 (`microsoft/wavlm-large`, `output_hidden_states=True`). Per segment compute:
- **C:** negative radius of gyration at layer 12 (compactness).
- **T:** velocity entropy at layer 9 (trajectory irregularity) — entropy of the distribution of frame-to-frame Δ-norms.
- Mean frame-to-frame cosine distance (sanity metric).
- Segment-mean embedding (for UMAP).

### 5.3 Analyses
1. Paired Wilcoxon signed-rank, laugh-bona vs. speech-bona, on C and T. Report effect size (rank-biserial r). → **H2 verdict.**
2. UMAP (n_neighbors=30, min_dist=0.1, cosine) of segment-mean embeddings, colored by the 4–5 groups above; two panels (layer 9, layer 12). → **Figure 1.**
3. If laugh-spoof exists: distance of laugh-spoof centroid to laugh-bona vs. speech manifolds (report cosine distances + a simple linear probe genuine-vs-synthetic-laughter AUC, 5-fold). → **H4 verdict.**

**Deliverables:** Figure 1 (UMAP panels), Table 2 (C/T stats + tests).

## 6. Stage 3 — Conditional detector inference (2 h; parallel with Stage 2)

Use the existing trained WavLM→BiLSTM→GAT detector checkpoint (the SAME checkpoint reported in the geometric-hardness work, for consistency).

1. Score all ITW files. Split by laughter presence (thr 0.5). Report AUC/EER per subset **only if** each subset ≥100 files per class (D1a); else report score distributions + Mann-Whitney U.
2. Frame-level: within laughter-containing utterances, compare per-frame detector scores on laughter frames vs. speech frames (paired by utterance). → the "cool SSL finding" candidate: does the detector's confidence collapse or spike during laughter?
3. Masking probe: re-score laughter-containing utterances with laughter regions spliced out; report per-file Δscore distribution. Δ >> 0 ⇒ detector leans on laughter as a cue.
4. Score the Bark probe clips (if D2 done): report detection rate at the detector's ITW-EER operating threshold. → headline number for H4.

**Deliverables:** Table 3 (conditional performance / score shifts), optional Figure 2 (frame-score traces over 2–3 example utterances with laughter spans shaded — very workshop-friendly figure).

## 7. Stage 4 — DO NOT retrain

No fine-tuning on laughter data before this deadline. Write it as future work: "training-time laughter augmentation" motivated by H1. If reviewers want it, camera-ready (Aug 1) or a follow-up.

---

## 8. Paper assembly (4 pages, ACM ICMI template, anonymized)

| Section | Content | Source |
|---|---|---|
| 1 Intro (0.75 p) | Generators now target laughter/emotion (cite EmoFake as emotion precedent, Bark as laughter-capable TTS); laughter = social signal whose authenticity matters; H1–H4 | framing §0–1 |
| 2 Related (0.5 p) | Laughter detection (Gillick 2021), laughter corpora (VocalSound, MAHNOB), deepfake benchmarks (ITW, MLAAD, ASVspoof one-liner), SSL geometry | — |
| 3 Method (0.75 p) | Detector pipeline, C/T features, segment pairing protocol, Bark probe | §4–6 |
| 4 Results (1.25 p) | Table 1, Fig 1, Table 2, Table 3 (+Fig 2 if space) | §4–6 |
| 5 Discussion (0.5 p) | Laughter = shortcut today, blind spot tomorrow; geometry predicts hardness | — |
| 6 Limits/Ethics (0.25 p) | Single laughter detector + measured precision; small n; probe-scale synthesis; dual-use note (CFP lists ethics — include 2 sentences) | QC §4 |

Release plan: state that laughter annotations for ITW (the Stage-1 CSV) will be released — this doubles as a **dataset contribution** under the CFP's "Datasets and Evaluation" topic and materially strengthens fit.

## 9. Time budget (deadline T-1 day)

| Hour | Task |
|---|---|
| 0–0.5 | Env setup; kick off ITW download |
| 0.5–2.5 | Stage 1 batch running unattended → meanwhile download VocalSound, write extraction code |
| 2.5–3 | D1 decision + QC listening |
| 3–7 | Stage 2 (embeddings + stats + Figure 1) ∥ Stage 3 scoring |
| 7–8 | Bark probe (only if ahead of schedule) |
| 8–12 | Write paper into template; tables/figures final |
| 12+ | Anonymization check, OpenReview submission (submit a safety draft EARLY — you can update until the deadline) |

## 10. Risk register

| Risk | Mitigation |
|---|---|
| Gillick repo dependency rot | Patches in §2; worst case, use its Colab to process a 2k-file ITW sample instead of full set |
| ITW has no laughter (D1c) | Pivot: VocalSound vs. Bark becomes the core (H2+H4); H1 becomes "benchmarks lack laughter" |
| Bark fails / too slow | Drop H4; paper still stands on H1–H3 |
| Laughter subset too small for EER | Score-distribution stats instead (pre-specified in §6.1) |
| "MuLaugh" unresolvable | Pipeline is corpus-agnostic; VocalSound fills the slot |
| Detector precision low on noisy ITW | Report measured precision; rerun headline stats at thr 0.7 as robustness check |

## 11. Acceptance criteria (definition of done)

- [ ] `itw_laughter.csv` covers ≥95% of ITW files; QC precision estimate recorded
- [ ] Table 1 with Fisher test at 3 thresholds
- [ ] Figure 1 UMAP + Table 2 paired Wilcoxon on C and T
- [ ] Table 3 conditional scores (form per D1 branch)
- [ ] 4-page anonymized PDF in ACM format submitted on OpenReview before 23:59 AoE July 10
- [ ] Repo/CSV prepared for anonymous release link (e.g., anonymous.4open.science)
