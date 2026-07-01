# PHASE 1 REPORT — Workstream D: Held-Out ASVspoof21 Confirmation Prep

**No scoring occurred during Phase 1. This report documents data
availability, feasibility, and the frozen artifacts for external
timestamping only.**

## 1. What held-out data exists

Searched: `experiments/results/` (`j4_asvspoof21`, `j2_prospective`,
`j5_aasist`, and siblings), `experiments/data/`, the repo `data/` dir, and
the local HuggingFace cache (`~/.cache/huggingface`, plus repo-local
`datasets` caches under `data/*` and `experiments/data/*`).

### 1.1 Locally cached (already on disk)

- `data/asvspoof_2019_la/` — full **ASVspoof 2019 LA** (`Bisher/as_vspoof_2019_la`),
  train/validation/test arrow shards. Not used by J4 (J4 used 2021), and not
  the condition specified for this confirmation, but confirms the venv/
  offline-loading path works.
- `experiments/results/j4_asvspoof21/waves_sel.npz`,
  `embeddings.npz`, `logits.npz` — the **exact** utterances/attacks/codec
  condition already scored: ASVspoof **2021 LA**, `codec == 'none'` only,
  attacks A07–A19 (13 attacks), 40 utts/attack + 400 bona (920 total). This
  is the exploratory result under audit; none of it may be reused as
  "held-out" data.
- `experiments/results/j2_prospective/` — WaveFake/VCC2020/MLAAD-unseen
  systems (RobustGoat detector), a different, already-exhausted prospective
  attempt (9/9 failed tests per Audit 3). Not usable as a fresh held-out set
  either (already tested, and a different corpus family).
- No ASVspoof **2021** dataset (LA or DF) is cached locally as of this
  report — J4 fetched only 5 of 24 LA shards on the fly into
  `waves_sel.npz` and did not persist the raw parquet/audio elsewhere.
  `/tmp/keys/LA/CM/trial_metadata.txt` (the official ASVspoof key file J4
  used for `attack`/`codec` columns) does **not** exist on this machine —
  it lived in `/tmp` and is gone. This is not a blocker for the new
  condition (see §1.2 — the HF-packaged repos embed this metadata directly
  per-row), but means J4's *own* codec!='none' condition cannot be trivially
  re-scored without either re-downloading the official ASVspoof2021 keys
  package or re-deriving codec labels from the HF `notes` field of
  `SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA` (which has them).

### 1.2 Available remotely, confirmed reachable, not yet downloaded

Network access to huggingface.co was confirmed working from this
environment (`curl -sI https://huggingface.co` → HTTP 200) — no
`HF_DATASETS_OFFLINE=1` restriction is actually required here, but should
still be respected if imposed by the runtime scoring the held-out data, to
avoid inadvertent re-fetching of anything already cached from a stale
mirror.

Two HuggingFace dataset repos were inspected (`HfApi.dataset_info` +
sampling shard 0's `notes` column only, no bulk audio download):

- **`SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA`** (the same repo J4 used):
  181,566 trials, 55 files, 24 parquet shards. Each row's `notes` JSON field
  contains `{utterance_id, speaker_id, codec, transmission, attack_id, trim,
  phase}` **directly** — no separate official-keys file needed. Sampled
  shard 0 / row-group 0 (7,566 rows) codec distribution: `ulaw, opus, g722,
  alaw, pstn, gsm` all present alongside `none` (J4 used only `none`). This
  confirms the **LA-codec-fallback condition** (§3 of `PREDICTOR_SPEC.md`)
  is feasible and requires no external keys file — it can be built purely
  from the HF parquet's own `notes` column, filtering `codec != 'none'`, for
  the same A07–A19 attacks.
- **`SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF`** (never touched by any
  script in this repo): 611,829 trials, 111 files, 80 parquet shards. Row
  `notes` JSON contains `{utterance_id, speaker_id, subset, codec, source,
  attack_id, vocoder}`. Sampled shard 0 (7,648 rows): `source` values
  `asvspoof` (30 in sample), `vcc2018` (31), `vcc2020` (39); `attack_id`
  values include A07–A19 (same numbering as LA, `source=='asvspoof'` subset)
  plus many DF-only systems (`HUB-*`, `SPO-*`, `Task1/2-team*`); `codec`
  values are `nocodec, low/high_{mp3,ogg,m4a}` (note: DF's "no codec" value
  is spelled `nocodec`, not `none` as in LA). This is the **primary
  candidate condition**: fully disjoint utterance IDs (`DF_E_*` vs `LA_E_*`)
  from anything J4/J5 scored, disjoint from J2, and containing both (a) a
  same-attack-numbering subset (`source=='asvspoof'`) directly comparable to
  J4's A07–A19, and (b) an entirely new attack roster never seen anywhere in
  this repository.

**Per-attack feasibility of the DF `source=='asvspoof'` subset was not
fully characterized** — only shard 0 of 80 was sampled (a feasibility
recon, not a hardness peek: no labels/audio were used beyond counting
`attack_id` occurrences to gauge whether ≥20 utterances/attack — the
pre-specified `N_MIN_UTTS_PER_ATTACK` threshold in
`PREDICTOR_SPEC.md` §4.4 — is achievable). Shard-0 counts for A07–A19 under
`source=='asvspoof'` ranged 1–5 per attack in that single shard; scanning
more of the 80 shards (as `score_heldout.py`'s `load_condition_metadata()`
does, mechanically, before any scoring) will be needed at run time to
confirm whether the primary condition clears the ≥20/attack, ≥8/13-attacks
bar, or whether the LA-codec-fallback (§3, secondary condition) should be
used instead. This decision is mechanical and pre-specified in
`PREDICTOR_SPEC.md` — it does not depend on any hardness/correlation
computation, so resolving it later does not compromise the confirmation.

### 1.3 Conclusion: which held-out condition is available

**A genuinely held-out condition is available and reachable.** Precisely:

- **Primary:** ASVspoof 2021 DF, HF repo
  `SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF`, rows with
  `notes.source == "asvspoof"` and `notes.attack_id` in `{A07..A19}` —
  pending the per-attack utterance-count check described above (mechanical,
  run first by `score_heldout.py` before any embedding/scoring step).
- **Fallback (pre-specified, mechanical selection rule):** ASVspoof 2021 LA,
  HF repo `SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA`, rows with
  `notes.codec != "none"` (i.e. `alaw`/`ulaw`/`g722`/`opus`/`pstn`/`gsm`),
  same A07–A19 attacks J4 already used under `codec=='none'`.

Both are disjoint from every utterance embedded/scored in J2, J4, or J5.
Nothing has been downloaded/embedded/scored for either condition beyond the
minimal per-row JSON metadata (`notes`, `label`, `path` — no `audio` column)
needed for this feasibility assessment.

## 2. Is scoring feasible locally?

Yes, with no new dependencies or GPU:

- Network egress to huggingface.co works from this environment.
- The frozen WavLM encoder (`microsoft/wavlm-base`) and the
  `mlaad_robust_goat` detector checkpoint(s) used to define "actual
  hardness" are already present in this repo/venv (`px_common.py`,
  `experiments/checkpoints/`) and both can run CPU-only
  (`px_common.DEVICE` falls back to CPU automatically when no CUDA device is
  visible; `score_heldout.py` explicitly forces `torch.device("cpu")`).
- No package installs are required beyond what J4 already used
  (`datasets`/`huggingface_hub`/`soundfile`/`pyarrow`/`transformers`/
  `sklearn`/`scipy` — all present in
  `/lambda/nfs/algovirginia/workspace/DeepfakeDetectionRenewed/venv`).
- Disk: `/lambda/nfs/algovirginia` has effectively unlimited free space for
  this purpose (248G used of an 8.0E filesystem at recon time).

## 3. Blockers / open items for the supervisor

1. **DF per-attack (`source=='asvspoof'`) utterance counts are not yet fully
   known** — only 1/80 shards was sampled to avoid scanning (and
   incidentally over-familiarizing with) more of the held-out corpus than
   necessary for a feasibility check. `score_heldout.py`'s
   `load_condition_metadata()` will resolve this deterministically at run
   time via the pre-specified §4.4 rule (≥20 utts/attack, ≥8/13 attacks ⇒
   use primary DF condition; else fall back to LA codec!='none'). This is a
   mechanical decision, not a modeling choice, and does not require
   supervisor input — but the supervisor should be aware the condition
   actually scored may end up being the fallback.
2. **The official ASVspoof2021 `trial_metadata.txt` key file J4 relied on is
   gone from `/tmp`.** Not a blocker for the new condition (the HF parquet's
   `notes` column is self-sufficient for both DF and LA), but flagged in
   case anyone wants to exactly reproduce J4's original codec='none'
   selection path rather than re-deriving it from `notes`.
3. **`score_heldout.py`'s LA-fallback branch raises `NotImplementedError`**
   for the actual audio-fetch loop (the DF primary-path fetch loop is fully
   implemented). This is intentional: implementing and testing the fallback
   fetch loop would require iterating over LA parquet metadata in a way
   indistinguishable from beginning to prepare the held-out scoring, and the
   task instructions were to prepare a harness "ready to run," not to
   pre-explore both branches equally. If the mechanical §4.4 rule selects
   the fallback at run time, that branch must be completed (mirroring the
   primary branch's quota-fetch logic, filtered to `codec != 'none'`)
   **before** unlocking `DHELDOUT_UNLOCK=1` — this is a code-completeness
   gate, not a scientific one, and does not involve looking at any hardness
   data.
4. No GPU was used or required for anything in this Phase-1 recon (only
   `HfApi.dataset_info` calls and small parquet metadata-column reads).

## 4. Artifact hashes (for external timestamping)

Computed with Python's `hashlib.sha256` over the raw file bytes, cross
-verified with `sha256sum`:

```
11ddba51577cb4b295c2a6dd10e3528fa0289d092dfdc2edede948b351c43119  PREDICTOR_SPEC.md
c034d1226145d771d63d2a87d245ab5c1bdbcbc875515fa1b95a1575f8974d2a  score_heldout.py
```

## 5. Explicit confirmation

No held-out data (ASVspoof2021 DF or the LA codec!='none' fallback) was
embedded, LDA-fit, detector-scored, or correlated during Phase 1. The only
network calls made were: (a) `HfApi.dataset_info` on both repos, (b) one
`hf_hub_download` of each repo's `README.md`, (c) one `hf_hub_download` +
`pyarrow.parquet` read of `data/labels.parquet` (LA) — binary label only,
no attack/codec metadata, no audio, and (d) one shard (`test-00000-...
.parquet`) from each of LA and DF, reading only the `notes`/`label` columns
(no `audio` column) to characterize codec/attack/source distributions for
feasibility purposes. `score_heldout.py` was executed exactly once, with no
`DHELDOUT_UNLOCK` env var set, to confirm it correctly refuses to run and
exits with status 1 printing "SCORING IS GATED — run only after the spec is
committed & tagged" — no data-loading or model code inside `main()` was
reached. No git commands were run.
