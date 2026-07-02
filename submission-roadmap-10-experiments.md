# Submission Roadmap — 10 Decisive Experiments, Confirmed-Results Assessment, Paper Overhaul

**Date:** 2026-07-02. **Deadlines:** IEEE WIFS **July 15**, ACM MM 2nd Deepfake Workshop **July 16** (13–14 days); NeurIPS workshops (~Aug–Sept); ICASSP 2027 (~Sept). One A10 GPU.

**Already in flight** (approved rectification plan, Sonnet agents running): ASVspoof21-DF roster recon, multilingual MLAAD manifest recon, ITW fusion rectification (E1/E2), channel-confound + spread-factor controls (E6/E7). Experiments below absorb and extend those.

---

## Part 1 — Ten decisive experiments

Venue key: **W**=WIFS, **M**=ACM MM deepfake wkshp, **N**=NeurIPS wkshp, **I**=ICASSP.

| # | Experiment | Decides | Venues | Cost | Status |
|---|---|---|---|---|---|
| X1 | Multilingual MLAAD law replication (pre-reg) | Law generality, n 61→130+ | all | <1 day | recon running |
| X2 | ASVspoof21-DF full-roster confirmation (pre-reg) | Held-out confirmation at real power | W M I | 1–2 days | recon running |
| X3 | Backbone generality (wav2vec2, HuBERT) | "WavLM-only" limitation | N I | 1 day | new |
| X4 | Layer-wise law profile + channel partialling | Mechanism vs artifact | N | ½ day | E6 running, extend |
| X5 | Channel-perturbation axis-rotation test | Is rotation channel or semantics? | N M | 1–2 days | new |
| X6 | Headroom curve (7 detector budgets) | Two-point anecdote → quantitative curve | W M I | 2–3 days GPU | new |
| X7 | ITW rectification + adaptation-budget curve | Honest domain-shift story | M W | done-ish | E1/E2 running |
| X8 | Axis-guided evasion attack + countermeasure | Security stakes of the law | **W** M | 1–2 days | new |
| X9 | TTS vs VC scoping + spread-inheritance | Law's boundary, new mechanism | M N | 1 day | new |
| X10 | Pooled meta-analysis, reliability-corrected | One unifying figure | all | ½ day | after X1/X2 |

### X1 — Multilingual MLAAD held-out law replication (pre-registered)
**Design:** per-language corpus-internal LOSO centroid axis on de/ru/pl (+fr if completable pre-freeze) → per-system sd_along → hardness (1−AUC, mlaad_robust_goat) → pooled within-language-z Spearman, one-sided perm p, frozen spec + pushed git tag before scoring. n≈69–100 held-out non-English systems.
**Novelty/value:** converts the law from "one English corpus" to "replicates across 4+ languages on genuinely held-out systems" — the single highest-leverage upgrade for every venue; also the honest answer to the failed d_heldout confirmation (new data, new pre-registration).
**Potential bugs:** (1) `.pt`→system manifest mapping errors (recon agent verifying); (2) per-language bona (M-AILABS) has language-specific channel — pooling *without* within-language z-scoring would let language-level channel differences masquerade as the law; (3) same TTS model appearing in multiple languages → pseudo-replication (report model-family–clustered sensitivity); (4) detector is English-trained → per-language hardness under domain shift (fine, but state it; don't swap detectors post hoc); (5) min-utts too low (~10/system) inflates sd_along noise — report the min-utts sensitivity row as done for English.
**Audit:** tag-before-scoring timestamp check; permute-within-language null; LOSO axis leakage assertion; audit11 recomputes ρ/p from cached embeddings; model-family jackknife.

### X2 — ASVspoof2021-DF full-roster held-out confirmation (pre-registered)
**Design:** roster = all DF eval systems excluding the spent source==asvspoof A07–A19 (vcc2018/vcc2020/HUB/SPO/team rosters); primary predictor = **sd_along** (the actual law — the failed test used axis *position*), secondary = P3-position with Holm; hardness vs DF's own bona pool; frozen quotas; one-sided perm p<0.05.
**Novelty/value:** at n=13 the failed test needed ρ≥0.47; at n≥40 systems, ρ≈0.3 suffices — this is the difference between "underpowered failure" and a real answer. A pass gives the paper its held-out confirmation; a fail is genuinely informative (law may be MLAAD/TTS-scoped) and must be reported either way.
**Potential bugs:** (1) **attack labels may be hidden for DF eval trials** — recon must confirm which metadata field identifies the system before the spec is written (blocking risk); (2) codec mixture confounds hardness (fix condition to nocodec or stratify, frozen in spec); (3) bona pool source heterogeneity (vcc vs asvspoof bona differ in channel — use matched-source bona or state the choice); (4) quota sampling (40/system) variance; (5) shard download failures mid-scoring → cache all audio before unlock.
**Audit:** recon/scoring separation (recon computed no statistics); tag precedes scoring; exact perm recomputation; no-retry clause; global BH-FDR ledger updated with this p.

### X3 — Backbone generality: wav2vec 2.0 + HuBERT
**Design:** embed the cached MLAAD n=61 utterance set with frozen wav2vec2-base and hubert-base; recompute LOSO axis, sd_along, same hardness labels; report the **full layer sweep** for each backbone (no layer cherry-picking), with BH across the backbone×layer family; headline = layer-matched (L12-analog) ρ.
**Novelty/value:** kills the paper's own stated limitation ("generality to wav2vec2/HuBERT untested"); if the law holds across three SSL objectives it's a property of SSL speech geometry, not WavLM — a real upgrade for NeurIPS-workshop/ICASSP framing. Cheap: ~1846 utts × 2 models on the A10.
**Potential bugs:** (1) layer-selection cheating (pre-declare the analysis: all layers reported, correction applied); (2) feature normalization / sample-rate mismatches silently degrading embeddings (sanity-check bona/spoof separability per layer first); (3) hardness labels come from a WavLM-based detector → cross-backbone test is axis-side only, say so; (4) mean-pooling window mismatch vs the WavLM pipeline.
**Audit:** assert embedding cache shapes/hashes; replicate one WavLM number end-to-end through the new code path as a positive control; perm p per layer + family correction verified by audit11.

### X4 — Layer-wise law profile + channel partialling (extends running E6)
**Design:** ρ(sd_along@L, hardness) for L=0..12 on MLAAD; partial out layer-0 velocity/RMS channel proxies from the L12 law; report raw and partial side by side.
**Novelty/value:** a rising layer profile (chance at L0 → peak at L9–12) is the cleanest "this is learned structure, not recording channel" figure the paper can have; directly answers the reviewer objection the paper currently concedes ("cannot fully exclude channel").
**Potential bugs:** (1) over-partialling — if channel genuinely mediates part of the effect, partialling it is conservative, not wrong: present both; (2) L0 proxies are channel+speaker mixtures (label them "channel/low-level proxy"); (3) collinearity between proxies → use rank-based partials with permutation inference, not OLS t-tests.
**Audit:** permutation nulls for partial statistics; confirm proxy features regenerate from cached embeddings; sign/magnitude cross-check against audit6's ITW vmean0 finding.

### X5 — Channel-perturbation axis-rotation test
**Design:** take MLAAD subset, apply controlled channel transforms (MP3/opus @ 2 bitrates, reverb, additive noise, bandlimit); re-embed; re-estimate the internal axis per condition; measure cos(w_clean, w_perturbed) against the within-corpus split-half ceiling (0.93) and the cross-corpus floor (≈0); also re-test the law under perturbation.
**Novelty/value:** the decisive experiment for the paper's biggest open question — if heavy channel perturbation barely rotates the axis (cos stays ≫ 0.5) while corpus change rotates it to ~0, rotation is *synthesis/content semantics*, not channel; if channel alone reproduces the rotation, the honest story changes and the field learns the axis is a channel-entangled object. Either outcome is publishable and quotable.
**Potential bugs:** (1) codec implementations (ffmpeg settings) leaking resampling artifacts — fix a documented transform recipe; (2) comparing axes across perturbations requires the same standardization frame (audit4's frame-dependence caveat — compute cosines in a single fixed frame); (3) perturbing only spoof or only bona breaks the centroid-difference symmetry (perturb both); (4) reusing the same utterances across conditions → correlated noise (fine for cosines, note for the law re-test).
**Audit:** split-half cosine per condition as a noise floor; seed-varied subsets; audit11 recomputes cosines from cached perturbed embeddings.

### X6 — Headroom curve: fusion gain vs detector strength
**Design:** train WavLM-GAT heads at budgets {125, 250, 500, 1k, 2k, 4k, full} × 2–3 seeds on ASVspoof2019-LA (head is ~13M params, minutes–hours per run on the A10); apply the identical unsupervised axis fusion to each; plot ΔEER(fusion) vs detector-alone EER with per-fold CIs; fit and test a monotone relationship.
**Novelty/value:** upgrades the current two-point contrast (mini_goat −0.030 vs robust_goat +0.004) into a *quantitative headroom law* — the most practically useful figure for WIFS/MM/ICASSP ("when should a practitioner bother with the free lever? here is the curve"). This is the strongest new applied result available in the time budget.
**Potential bugs:** (1) budget subsampling must be attack-stratified and seed-controlled or small budgets get degenerate attack coverage; (2) early stopping on the eval set (use a val split disjoint from eval folds); (3) λ selection per budget on train folds only; (4) at tiny budgets EER estimates are noisy → seeds and CIs mandatory; (5) don't let "full" quietly be a different checkpoint than the published robust_goat (reuse the existing checkpoints for 500/full anchor points to stay consistent with published numbers).
**Audit:** attack-disjoint fold assertions; verify the 500-utt and full anchors reproduce the published mini_goat/robust_goat numbers; isotonic/Spearman trend test with cluster bootstrap; training logs committed.

### X7 — ITW rectification + adaptation-budget curve (running: E1/E2)
**Design (running):** generalized fusion weights (axis-only allowed), residualized/channel-cleaned detector variants, strict speaker-disjoint folds, cluster-bootstrap CIs; logistic label-light combiner; then assemble the **adaptation-budget curve**: EER vs #labels {0 (unsupervised axis) … 250 (j3 LDA 0.192)} on ITW.
**Novelty/value:** the honest domain-shift product: "an unsupervised corpus-internal axis (0.301) beats a fine-tuned detector (0.363) on ITW; naive fusion can't help (correlated, channel-driven errors); with k labels you get this curve." Practical, clean, and immune to the audit findings that killed the old claim.
**Potential bugs:** speaker leakage (only strict_identity folds count — audit5's "strict" still leaked); weight-selection optimism (selection on train folds, claims on held-out + CI); the ITW axis itself may partially encode channel (report vmean0 correlation next to it); j3 reference numbers used a different fold RNG (recompute under the same folds for the curve).
**Audit:** audit11 re-runs disjointness assertions and bootstrap; the word "zero-cost" banned for anything using labels.

### X8 — Axis-guided evasion attack + countermeasure (the WIFS experiment)
**Design:** threat model: attacker has a TTS system + frozen public WavLM (no detector access). Attack = **selection/rejection sampling**: generate/score candidate utterances by axis projection (toward bona) and keep the most bona-like fraction f ∈ {50%, 25%, 10%}; measure detector miss-rate/EER vs f, on WavLM-GAT and AASIST (transferability). Countermeasure: does axis-fused or axis-aware scoring resist the selection? Implement by re-ranking existing MLAAD/ASVspoof utterances per system (no new synthesis needed).
**Novelty/value:** converts the interpretability law into a **security result** — "the geometry that predicts hardness is also a zero-knowledge evasion lever, and here is the defense" — exactly WIFS's remit and strong for the MM deepfake workshop. No one has shown a detector-blind, frozen-SSL-guided selection attack of this form.
**Potential bugs:** (1) selection shrinks n → compare at matched n via random-selection null; (2) results specific to one detector → both architectures required; (3) attacker "bona centroid" must come from attacker-side bona (e.g., LibriSpeech), not the defender's corpus, or the threat model is inconsistent; (4) this is *selection*, not synthesis modification — frame precisely, don't overclaim "adversarial audio"; (5) EER under class-imbalanced selection needs care (report full ROC).
**Audit:** pre-declared selection rule and fractions; random-selection permutation null; both-detector replication; audit11 recomputes miss-rates from cached scores. Ethics note in paper (defensive framing, published corpora only).

### X9 — TTS vs VC scoping + spread-inheritance mechanism
**Design:** split all corpora's systems by type (TTS vs VC — MLAAD metadata, ASVspoof A17–A19 etc.); report the law per type with per-type n; regenerate the currently-uncited "ρ≈0 on VC" claim from artifacts. Mechanism probe: for VC systems, does the *source speech's* spread predict hardness (spread inheritance)?
**Novelty/value:** the paper already claims the TTS/VC boundary in Limitations with no artifact behind it — this makes the boundary a supported *finding* and, if spread-inheritance shows anything, a new mechanism paragraph. Cheap and reviewer-proofing.
**Potential bugs:** small VC n (report exact n, no significance theater); type labels wrong in metadata (hand-verify the roster); VC bona-similarity confound (VC retains source prosody — that's the hypothesis, not a bug, but keep it explicit).
**Audit:** the ρ≈0 VC number must regenerate from a committed script (Workstream-A discipline); permutation within type.

### X10 — Pooled cross-corpus meta-analysis, reliability-corrected
**Design:** pool per-system (sd_along, hardness) from MLAAD-en (61) + X1 (69–100) + X2 roster + ASVspoof-19/21 (13) with within-corpus z-scoring; Stouffer combination of per-corpus one-sided p's; report disattenuated ρ using the 0.86 hardness reliability; leave-one-corpus-out sensitivity. Explicitly exploratory.
**Novelty/value:** one figure — "the law across ~180 systems, 5+ corpora, 4+ languages" — the money plot for any venue; the disattenuation makes the "32% of explainable variance" claim rigorous.
**Potential bugs:** corpora share detectors → non-independence in Stouffer (use the per-corpus-detector structure honestly, label exploratory); z-scoring erases genuine between-corpus effect-size differences (also show per-corpus ρ forest plot); double-counting the 13 ASVspoof attacks if X2's roster overlaps (it won't — A07–A19 excluded, assert it).
**Audit:** leave-one-corpus-out; per-corpus ρ CIs; exploratory label enforced; numbers regenerate from one script.

**Recommended cut for July 15/16 deadlines:** X1, X2, X7 (in flight) + **X8** and **X6** (the two decisive *new* ones for WIFS/MM) + X3/X4 if time (each ≤1 day). X5, X9, X10 slot into the NeurIPS-workshop/ICASSP version.

---

## Part 2 — What is actually confirmed, and is it MOSS-publishable?

### The confirmed core (survives the 10-audit red team)
1. **The law (P1):** per-system spread along the corpus-internal natural↔synthetic axis in frozen WavLM-L12 predicts detection hardness. MLAAD n=61, LOSO R²=0.277, ρ=0.60, the only 1 of 83 tests surviving *global* BH-FDR; strengthens as measurement noise falls (0.60→0.64→0.69); rank-order replicates on a different architecture (AASIST-FT, ρ=0.35).
2. **The axis rotates (P3):** estimated almost noiselessly within-corpus (split-half cos 0.93–0.98) yet cross-corpus cosines sit at the chance floor or negative (MLAAD↔ASVspoof21 −0.14 to −0.21). A transferred axis is uninformative or backwards.
3. **Headroom-conditional free lever (P5 + mini_goat):** unsupervised axis fusion cuts a weak detector's EER ~21% (all folds, all seeds; MLAAD in-domain 0.272→0.163) and does nothing for a near-ceiling detector (p≈0.56–0.80).
4. **Method hygiene as a result:** hardness reliability ceiling quantified (0.86); a pre-registered, externally-timestamped held-out confirmation was run and **failed honestly** (ρ=+0.33, p=0.135, n=13); a red-team audit corrected 13 claims. Plus (ITW, once E1/E2 land): an unsupervised corpus-internal axis beats a fine-tuned detector under domain shift.

### Is it common sense?
Partly — and the partition matters:
- **The near-common-sense part:** "systems whose fakes overlap the real-speech region are harder to detect" is close to the definition of AUC (class overlap ⇒ low separability). And there is a partial-circularity exposure: hardness is measured with a detector built on the *same* frozen encoder whose geometry supplies the predictor. If sd_along were just "variance along the detector's decision direction," the law would be a tautology.
- **Why it isn't a tautology:** (a) the naive form of the intuition — axis *position* (closer to bona = harder) — is exactly what **failed** (P1-position predictors on ASVspoof, the failed held-out test); the survivor is *spread*, an unsupervised second-moment quantity, which is not the obvious guess. (b) The axis is a two-centroid unsupervised direction, not the detector's boundary, and the rank-order law transfers to AASIST, a detector not built on WavLM — the circularity objection has a concrete, tested answer (it should be made louder in the paper, and X3 strengthens it further). (c) Nothing in "class overlap" intuition predicts that the direction carrying this structure is *corpus-specific to the point of orthogonality* — the literature's working assumption (shared vocoder artifacts, universal one-class cues) implies the opposite. (d) "Fusion always helps weak scores" is ensemble folklore, but the *conditionality* (helps at headroom, exactly zero at ceiling, hurts when errors are correlated and channel-driven, as on ITW) is a measured boundary, not folklore.
- **Verdict on novelty:** the law alone = moderate novelty with a real circularity question attached; the law **+ rotation + measured boundary conditions + audit discipline** = a genuine contribution. The paper's current title has it right: the contribution is the *map of where the cheap signal works and where it doesn't*.

### Is it relevant to the field?
Yes, on the field's most painful axis: **cross-domain generalization failure is the central unsolved problem of audio deepfake detection** (every ASVspoof/ITW paper laments it). This work gives (i) a geometric *explanation* (the discriminative direction is corpus-local — detectors learn a direction that doesn't exist elsewhere), (ii) a cheap *diagnostic* (estimate the internal axis, read off which systems will evade), and (iii) a cheap *lever* with an honest applicability map. It is *not* relevant as a SOTA detector contribution — EERs here are far from leaderboard numbers — and it should never be framed as one.

### MOSS@COLM verdict
**Yes — publishable at a methods-at-small-scale workshop, and a strong fit,** with two caveats. Fit: single GPU, frozen ~95M encoder, training-free/label-light interventions, insight-per-FLOP framing, an 83-test FDR ledger, a pre-registered failure reported as a failure — this is precisely the methodology culture such venues exist to reward; the honest-negative-result and audit apparatus is itself a MOSS-style contribution. Caveats: (1) the venue itself is unconfirmed (the only verified MOSS is ICML's; the COLM-2026 analog's deadline has passed) — resolve the target before polishing; (2) the current single-corpus law is the weak point — **X1 (multilingual) is the one experiment that most changes the paper's acceptance odds**, and X2 turns the reported failure into a properly-powered answer. If X1 and X2 both fail, the honest paper is still submittable (law + rotation + boundary map on MLAAD, with two pre-registered negatives), but it drops from "strong workshop paper" to "solid workshop paper"; I would *not* stop — the rotation and headroom results carry weight independent of the confirmations.

---

## Part 3 — main.pdf analysis, diagram overhaul (Opus agent), headlines

### Bugs found in `main (1).pdf` (fix all; each must trace to a regenerating script)
1. **95M vs 100M parameter mismatch (confirmed):** Abstract says "frozen ∼100M-parameter encoder"; Setup and Reproducibility say "∼95M". WavLM-base is 94.7M — standardize on "∼95M (WavLM-base, 94.7M)" everywhere.
2. **Reliability inconsistency:** Setup says "split-half reliability **0.76** (R² ceiling ≈ 0.86)"; the audit (audit9) established split-half reliability **0.86**, and the ceiling *is* the reliability. Either 0.76 is a typo or it's a different (held-out) split statistic — verify against audit9 outputs and make the two numbers consistent with a stated definition.
3. **Fig 2b annotation conflates observed cosine with chance floor:** figure annotates "chance 0.08" while the text says the observed cos ≈ −0.08 against a chance floor ≈ 0.03. The 0.08 on the figure appears to be the *observed* |cos| mislabeled as chance. Redraw with three clearly distinct reference lines: within-corpus 0.93, observed −0.08, chance band ±0.03.
4. **Fig 2c labeling (the user-flagged "weird WavLM-GAT labelings"):** bars labeled "WavLM-GAT (500 utt) / (full) / (MLAAD)" mix detector strength and corpus in one unlabeled dimension, and the y-axis says EER while the annotations are ΔEER. Redraw as paired bars (detector alone vs +fusion) grouped by regime: "weak detector, ASVspoof19 (500-utt)", "strong detector, ASVspoof19 (near ceiling)", "mid detector, MLAAD (in-domain)"; annotate ΔEER with CIs.
5. **Strong-detector EER:** text says 0.078; `c_asvspoof_fusion` artifacts say robust_goat 0.071. Regenerate and pick the artifact number.
6. **Omission of the failed pre-registered confirmation (integrity issue):** §7 says cross-dataset prospective prediction "remains unconfirmed and is left as future work" — but a pre-registered, externally-timestamped test was run and **failed** (ρ=0.33, p=0.135). The HEADLINE_STORY reports it proudly; the paper hides it. Add 2–3 sentences reporting the failure explicitly (it *strengthens* the paper at an audit-culture venue) — and update again when X2 lands.
7. **Uncited VC claim:** "on voice-conversion attacks all predictors give ρ ≈ 0" has no artifact/regeneration path — X9 fixes this; until then, soften or cite the artifact.
8. **ITW absent entirely:** the domain-shift section would be materially stronger with the (rectified) ITW result — axis-alone 0.301 vs fine-tuned detector 0.363, fusion can't help because errors are correlated/channel-driven (E1/E2 output). Currently the paper's "where it does not help" story has no in-the-wild evidence.
9. Minor: "T raining-Free"/"T raining" kerning artifacts in title and §5 heading (LaTeX spacing in the .sty or a literal typo — check source); verify "83 quantities" still matches the final ledger after new experiments (it will change); ensure "≥8 utterances" matches audit2's actual min-utts filter.

### Diagram regeneration plan (execute via **Opus agent**, using paper/figures/_style.py + the dataviz skill)
- **Fig 1 (architecture):** keep but tighten: consistent frozen/trainable color coding, mark *where* geometry is read out (L12 tap) visually, drop the CTC detail into the caption.
- **Fig 2a (law scatter):** add the reliability-ceiling band and per-point min-utts encoding (size); annotate 2–3 named systems (easiest/hardest) for narrative.
- **Fig 2b (rotation):** redraw per bug #3; better: a small **cosine matrix heatmap** (MLAAD, ASVspoof19, ASVspoof21, ITW; within-corpus split-half on the diagonal) — the rotation story in one glance, all values already exist in audit4 outputs.
- **Fig 2c (headroom):** redraw per bug #4; replace outright with the **X6 headroom curve** when it lands (ΔEER vs detector-alone EER with CIs — a far stronger figure than 3 bars).
- **New Fig 3 (quartile mechanism):** i7 per-quartile ΔAUC (+0.04/+0.08/+0.14/+0.22) — shows the fusion gain lands exactly on the hardest systems, the law-as-mechanism plot; data in `i7_quartiles.csv`.
- **New Fig 4 (domain shift / ITW):** adaptation-budget curve from X7 (EER vs #labels, unsupervised point at 0 labels, detector-alone and axis-alone reference lines).
- **New Fig 5 (if X1/X2 land):** X10 forest/pooled plot — per-corpus ρ with CIs + pooled estimate.
- All figures regenerated from committed scripts into `paper/figures/` with a `make figures` entry; no hand-edited numbers.

### Headline options (title + section heads)
Current title is honest but flat. Candidates, in order:
1. **"The Axis Rotates: Corpus-Local Geometry Predicts — and Bounds — Deepfake Detection"**
2. **"Spread Predicts Evasion: A Training-Free Hardness Law in Frozen Speech Representations"**
3. **"One Direction per Dataset: Why Deepfake Detectors Don't Transfer, and What a Frozen Encoder Tells You for Free"**
4. Keep current title, sharpen the subtitle: "…Where a Training-Free Hardness Signal Helps — and Where It Provably Does Not".
Section heads: prefer claim-shaped heads ("§3 Spread along a frozen axis predicts which systems evade", "§4 The axis is corpus-local: the geometry of transfer failure", "§5 A free corrective signal, paid only under headroom", "§6 What an adversarial audit changed").

### Tips to make this a genuinely strong MOSS-class submission (my honest take: it has the potential — proceed)
1. **Land X1 before anything else.** A held-out multilingual replication converts the single point of fragility (one corpus) into the paper's strongest sentence.
2. **Make the audit a first-class contribution**, not an apologia: a table of "claim → audit outcome → correction", the 83-test FDR ledger as a figure, and the failed pre-registration reported with its timestamp. At a methods venue this *is* novelty; nobody else does it.
3. **Answer circularity loudly and early** (§3): the position-form failed, the spread-form survived; the axis is unsupervised and cross-architecture; X3/X4 layer profiles if available. One paragraph, kills the most likely reject reason.
4. **State the negative results as findings with mechanisms**, not caveats: rotation (with X5's channel-vs-semantics answer), ITW fusion failure (correlated channel-driven errors), ceiling-detector null. The paper's identity is the *map*, so the "does not work" regions need the same evidential care as the wins.
5. **One quantitative artifact per claim**: every number in the paper regenerates from one script (Workstream-A discipline) — say so in the reproducibility statement and mean it.
6. **Keep EER-leaderboard framing out.** The moment a reviewer reads this as a detector paper, it loses; the takeaway sentence should be the transferable lesson about cheap geometric directions in frozen SSL models.
7. **Venue triage:** WIFS wants X8+X6 (security + operational curve); MM deepfake workshop wants X1+X7+X6 (generality + honest domain shift); NeurIPS workshop wants X3+X4+X5+X10 (mechanism); MOSS-style wants exactly what you have + X1/X2 (method-per-FLOP + audit culture). Same core, four different lead paragraphs — do not send the same framing to all four.

### Execution
- Diagram + paper-fix work: **Opus agent** (figures, LaTeX fixes, headline variants) — launch after E1/E2/X1-recon results land so figures use rectified numbers.
- New experiments X3–X6, X8–X10: Sonnet coding agents, one per experiment, audited by audit11 pattern (same as the running wave).
- Order for the July 15/16 deadlines: X2 spec freeze → X1 scoring → X8 → X6 → paper overhaul (Opus) → X3/X4 as buffer allows.
