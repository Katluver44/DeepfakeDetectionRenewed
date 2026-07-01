# PATCHLOG.md — infra-only fixes applied to `score_heldout.py` at scoring time

This log records every change made to the frozen harness `score_heldout.py`
between the pre-commit tag `dheldout-precommit` (commit `3225fbb`) and actual
execution. Per `PREDICTOR_SPEC.md`, only infrastructure/plumbing bugs that
block execution may be fixed — no change to P3's construction, the
permutation test, frozen constants, quotas, or the condition-selection rule
is permitted or was made. Each entry below is verbatim before/after plus the
reason.

---

## Patch 1: `HfApi.dataset_info()` does not accept `repo_type` in this environment's huggingface_hub version

**File:** `experiments/results/d_heldout/score_heldout.py`, function `load_condition_metadata()`.

**Why:** The installed `huggingface_hub` version in `./venv` is `0.20.3`. Its
`HfApi.dataset_info` signature is:

```
(self, repo_id: str, *, revision=None, timeout=None, files_metadata=False, token=None) -> DatasetInfo
```

There is no `repo_type` keyword argument — `dataset_info()` is already
dataset-scoped by construction in this version, so passing `repo_type="dataset"`
raises `TypeError: dataset_info() got an unexpected keyword argument 'repo_type'`
and the harness cannot even resolve which condition (DF vs LA fallback) is
feasible, let alone score anything. This is pure API-surface plumbing; it
does not touch P3, the test, any frozen constant, quota, or the
condition-selection rule (`n_ok >= 8` over A07–A19 is untouched).

**Before:**
```python
    api = HfApi()
    info = api.dataset_info(DF_REPO, repo_type="dataset")
```

**After:**
```python
    api = HfApi()
    info = api.dataset_info(DF_REPO)
```

**Scope check:** confirmed via `inspect.signature(HfApi.dataset_info)` before
patching. No other call sites of `dataset_info`/`repo_type` exist in the file.
