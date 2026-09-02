# EXPERIMENTS.md — Reproducibility & Experimentation Scope

> Companion to `structural-break-realtime-context.md`. That file tracks the overall
> project; this one is the authoritative, append-only record of what's been run, with
> exactly what's needed to reproduce or extend each result. If you are an agent picking
> up this project cold: read the **Scope** section before touching any code, and append
> to the **Experiment log** after every run that changes local TS-AUC — never overwrite
> a prior entry.

## 1. Scope of experimentation (read this first)

**Update 2026-09-02: Tier 3's first result is in, and it barely beat Tier 1/2.** A
`HistGradientBoostingClassifier` trained on the union of every prior method's features
(row 8) reached TS-AUC 0.5194 — only +0.0024 over the best Tier 1 result (`ks_test`
v2, 0.5170) and +0.017 over the weakest (`cusum`, 0.5024). Eight rows across three
tiers now span a total TS-AUC range of about 0.017. See §5's ranked comparison table
and the "effect-size mismatch" hypothesis, now the leading explanation rather than a
surviving one. **This is not yet grounds to declare a hard ceiling** — the `learned`
classifier used a modest feature set and `max_iter` (constrained by streaming-protocol
per-call latency, see row 8's notes) — but every additional Tier 1/2 hand-crafted
statistic tried so far has added essentially nothing beyond what `ks_test` v2 already
captured, so further hand-crafted variants are deprioritized until either Tier 3 is
pushed harder (richer features, different model) or Tier 4 is tried.

Still explicitly **out of scope** for now:
- Tier 2's non-cumulative members not yet tried (ADWIN, BOCPD, the two-state
  regime-switching model) — deprioritized; unlikely to add more than the ~0.02 ceiling
  every Tier 1/2 method so far has topped out at.
- Tier 4 (foundation-model embeddings)

## 2. Current implementation, exactly as it stands

Both scorers live inside `infer()` in `baseline_with_viz.ipynb`, selected by a single
notebook-level constant:

```python
SCORING_METHOD = "ewma"    # or "ks_test"
ALPHA    = 0.05            # EWMA decay
KAPPA    = 3.0             # EWMA tanh scale; |z|=3 -> score ~0.76
KAPPA_KS = 3.7             # KS tanh scale; calibrated to match EWMA's no-break floor (~0.2),
                           # NOT the textbook 5% critical value (1.36), which still averaged ~0.53
```

**EWMA branch** (unchanged since the original baseline):
```
mu_ewma_t = (1-ALPHA)*mu_ewma_{t-1} + ALPHA*x_t
n_eff_t   = (1-ALPHA)*n_eff_{t-1} + 1
z_t       = (mu_ewma_t - mu_H) / (sd_H / sqrt(n_eff_t))
score_t   = tanh(|z_t| / KAPPA)
```

**KS-test branch** (current, corrected version — see §4 for the version history):
```
d_t, _ = ks_2samp(x_historical, online_so_far)      # online_so_far = x_online[:t+1]
n_eff  = len(x_historical) * t / (len(x_historical) + t)
score_t = tanh(d_t * sqrt(n_eff) / KAPPA_KS)
```
`ks_test_scores()` (the plotting helper) implements the identical formula — if you
change one, change both, or they will silently drift apart.

## 3. Reproducing a result from this file

1. Open `baseline_with_viz.ipynb`.
2. Set `SCORING_METHOD` to the value in the experiment log row you want to reproduce.
3. Re-run cells top to bottom from the hyperparameters cell onward (order matters —
   `crunch_tools.test()` must run *after* the `SCORING_METHOD` change, not before).
4. The **Computing TS-AUC locally** cell prints the number to compare against the log.
5. Determinism check: re-running step 3 twice with no code changes must reproduce the
   identical TS-AUC — if it doesn't, something non-deterministic crept in (unlikely
   here, since neither scorer uses randomness, but worth stating explicitly).

Environment: exact package versions have not been pinned yet. Before the next
experiment, run `pip freeze > requirements.txt` in the working environment and commit
it — right now, reproducibility of *results* is tracked here, but reproducibility of
the *environment* is not, and that gap should close before Tier 2 starts.

## 4. Experiment log

Append one row per run. Never edit or delete a prior row — if a result is superseded,
say so in the Notes column of the new row.

| # | Date | SCORING_METHOD | Config | Local TS-AUC | Notes |
|---|---|---|---|---|---|
| 1 | — | `ewma` | ALPHA=0.05, KAPPA=3.0 | **not yet recorded** | Original baseline as shipped. Never reported back in this conversation — do not assume a value; re-run and log it before comparing against it. |
| 2 | — | `ks_test` (v1) | `score = 1 - p_value` from `ks_2samp` | **0.5039** | Essentially random. Flagged as suspicious given how close to exactly 0.5 it is — see §5, still open. |
| 3 | — | `ks_test` (v2, current) | `score = tanh(d*sqrt(n_eff)/KAPPA_KS)`, KAPPA_KS=3.7 | **0.5170** | Small improvement over v1. This was a *calibration* fix (matching EWMA's no-break floor), not expected to move a rank-based metric like TS-AUC much — the two scores are both ~monotonic in the same underlying KS statistic. The small nonzero gain is attributed to `ks_2samp` switching between exact and asymptotic p-value computation at small sample sizes (v1's `1-p_value` isn't purely monotonic in `d*sqrt(n_eff)` in that regime; v2's score is, by construction). Does not resolve §5. |
| 4 | 2026-09-01 | `ks_multi` | `score = max(s_raw, s_absdev, s_diff)`, all three views KS-based, KAPPA_KS=3.7 shared | **0.5046** | Runs the KS comparison on three transforms of the series — raw values (location/shape), absolute deviation from the historical mean (variance/scale), first differences (dependence/volatility structure) — and takes the max. Verified on synthetic data with a pure autocorrelation break (matched mean and variance): raw-values view alone scored ~0.39 averaged over 10 trials, first-differences view caught it at ~0.71. Executed via a clean kernel restart + top-to-bottom run (see the now-resolved stale-execution incident below); `crunch_tools.test()` took 2:53, confirming the KS-based path actually ran. Barely moves the needle over v1/v2 `ks_test` — does not resolve §5. |
| 5 | 2026-09-01 | `feature` | Rolling-window (`FEAT_WINDOW=30`) mean/std/skew/lag-1-autocorr/mean-abs-diff, each z-scored against its own historical sliding-window null distribution, combined via `max(\|z\|)`, `KAPPA_FEAT=3.0` | **0.5026** | Structurally different from the KS family: compares fixed-size trailing-window statistics against a historical window-distribution instead of doing a whole-sample KS test. Verified on synthetic data first — AUC 0.988 (mean shift), 0.991 (variance shift), 0.964 (autocorrelation shift), so the scoring logic itself discriminates real regime changes well. On the real dataset it lands at the same ~0.50 floor as every other method. Ruled out as a pipeline artifact by three checks: (1) oracle scoring (using ground-truth `target` as the score) gives a perfect TS-AUC of 1.0, proving the merge/metric plumbing is correct; (2) predictions have real variance (std 0.19, full [0,1] range), not a collapsed constant; (3) excluding each series' first 30 warm-up steps (forced-zero region) barely moves the number (0.5029). Per-series inspection shows the detector does ramp up correctly 30-60 steps after the true `tau` (e.g. id 10002: 0.0 at tau, ~0.49 by tau+60) — the failure is specifically in **cross-sectional ranking at fixed time_online**, not in per-series signal extraction. Four structurally different detectors (EWMA, ks_test, ks_multi, feature) now agree at ~0.50-0.52 — see updated §5. |
| 6 | 2026-09-01 | `cusum` (Tier 2) | Two-sided CUSUM on z-scored residuals vs. historical mean/std, slack `CUSUM_K=0.5`, `score = tanh(max(g+,g-)/KAPPA_CUSUM)`, `KAPPA_CUSUM=5.0` | **0.5024** | First Tier-2 (cumulative, non-forgetting) detector tried, specifically to test the "cumulative vs. instant evidence" hypothesis from §5. Verified on synthetic data first: AUC 0.995 (mean shift), 0.970 (variance shift), 0.815 (autocorrelation shift), no-break floor 0.195 — all sensible and, notably, *better* than `feature`'s synthetic numbers on the harder autocorrelation case despite CUSUM being a much simpler statistic. `crunch_tools.test()` ran in 30s (O(1) per step, as expected, versus multi-minute KS/feature runs), confirming the intended method executed. On the real dataset: **same ~0.50 ceiling as every prior method.** This is the first real evidence against the "cumulative vs. instant" hypothesis — a genuinely non-forgetting detector does no better than the windowed ones, which shifts weight toward the "effect-size mismatch" hypothesis in §5 instead. |
| 7 | 2026-09-01 | `page_hinkley` (Tier 2) | Classical Page-Hinkley on running online mean (no historical reference), allowed drift `PH_DELTA=0.5`, `score = tanh((PH_t - min PH)/KAPPA_PH)`, `KAPPA_PH=5.0` | **0.5058** | Second Tier-2 detector. Structurally different from CUSUM in one key way: it has no historical reference at all, comparing the stream only to its own running mean, so it should in principle be *more* sensitive to slow drift and *less* sensitive to shifts that don't move the mean (variance/autocorrelation-only breaks). Verified on synthetic data: AUC 0.996 (mean shift, as expected — this is the classical mean-shift test), 0.747 (variance shift, weaker than CUSUM's 0.970 since PH only tracks the mean), 0.590 (autocorrelation shift, weaker still), no-break floor 0.094 (tighter than CUSUM's 0.195). `crunch_tools.test()` ran in 23s (O(1) per step, as expected). On the real dataset: marginally the *best* of all six methods tried so far, but still solidly in the ~0.50 band — not a meaningful improvement. |
| 8 | 2026-09-02 | `learned` (Tier 3) | `HistGradientBoostingClassifier(max_iter=100, max_depth=6, learning_rate=0.08, l2_regularization=1.0)` over 14 causal features (5 rolling-window stats + their historical z-scores + `z_raw` + `cusum` + `page_hinkley` + `t_idx`), trained on every training series' online segment with `label_t = 1 iff t >= tau` | **0.5194** | First Tier-3 result: a learned classifier over the union of every prior method's features, instead of one hand-tuned formula. **Two real bugs found and fixed before this number is trustworthy:** (1) the first implementation buffered a series' entire `x_online` up front to batch `predict_proba` calls and compute a `t_frac` (fraction-of-online-length-elapsed) feature — this crashes the local tester with `ProtocolError: previous value not yield-ed`, since `x_online` is a lazy generator gated on `infer()` yielding a score before it releases the next point (exactly what the notebook's own `infer()` docstring warns about). Fixed by consuming one point at a time and calling `predict_proba` per point. (2) `t_frac` had to be dropped entirely, not just recomputed per-point — it fundamentally needs the total online length in advance, which the streaming contract never provides; keeping it would have been training on information a real submission cannot have. Retraining without `t_frac` dropped the training-set validation TS-AUC from an inflated 0.6518 down to 0.536-0.545 — `t_frac` had been carrying a large share of that number, almost certainly by exploiting a structural correlation between `t_frac` and where `y_test`'s reduced scoring window happens to start relative to `tau` (see the `tau_offset_in_window` analysis under row 5), not genuine signal. `max_iter` was also reduced from a first attempt at 300 to 100, since `infer()` must call `predict_proba` once per streamed point (batching isn't possible under the fixed protocol) and per-call latency scales with `max_iter` — 300 pushed local testing past a 30-minute cell timeout; 100 completed in 5:56 (train 2:37, infer 3:04, determinism check 14s) for an estimated ~0.01 TS-AUC cost. Permutation importance: `roll_std`/`z_std` dominate by a wide margin (variance shift remains the standout learnable signal, matching every prior method's own weak point being autocorrelation/shape), `t_idx` a distant third, `cusum`/`page_hinkley`/autocorrelation features contribute almost nothing. **The real, protocol-correct result (0.5194) is barely distinguishable from `ks_test`'s 0.5170** — smaller than the training-validation split suggested (0.536-0.545), consistent with some amount of overfitting to the training series' specific structure that doesn't fully transfer to the held-out reduced test set. Still nominally the best result across all 8 rows, but not the clear win the first (buggy) run appeared to show. |
| 9 | 2026-09-02 | `learned` (Tier 3.5) | Same `HistGradientBoostingClassifier` config as row 8, extended to 20 causal features: row 8's 14 plus `wave_e0`/`wave_e1`/`wave_e2` (log-energy per DWT sub-band, `haar` wavelet, level 2, periodization mode, over the trailing `FEAT_WINDOW`-point window), `wave_entropy` (Shannon entropy of the per-band energy distribution), `spec_entropy`/`spec_centroid` (Shannon entropy and centroid of the window's Welch PSD) | **0.5245** | User asked to look beyond summary statistics toward wavelet/spectral transforms of the raw series. Rationale: every feature in row 8 describes a window's *values* (mean, std, skew) — none describe how energy is distributed *across scales or frequencies*, which a break that changes short-range dependence or noise structure (not just level/variance) could shift even when `roll_mean`/`roll_std` barely move. Used as raw values, not z-scored against a historical null (unlike the row-8 rolling stats) since `HistGradientBoostingClassifier` thresholds features directly and doesn't need pre-normalized inputs. **Verified before running on real data:** (1) vectorized (training) vs. streaming (inference) implementations matched to floating-point precision (~1e-14 max abs diff) across three scenarios — a plain random series, a series with an injected variance-only break, and a short online segment below `FEAT_WINDOW` — avoiding a repeat of row 8's original rolling-window alignment bug; (2) on the injected variance-only break, `wave_e1`/`wave_e2` visibly shifted (one run: 1.95→3.90 and 2.75→4.72) alongside the expected `roll_std` shift; (3) a 40-series smoke test of the actual `train()`/`infer()` notebook cells ran end-to-end without errors. **Parallelization attempted and reverted:** `train()` initially extracted per-series features via `joblib.Parallel(n_jobs=-1)` to cut the ~22 min serial extraction time down, but this crashes under this competition's local-test harness — `crunch`'s `monkey_patches.joblib_parallel_initializer` unconditionally patches `joblib.Parallel.__init__` so every worker process re-imports the user's code from a `main.py` file on disk; this notebook-based workflow has no `main.py`, so every worker fails immediately with `FileNotFoundError` and the whole pool dies with `TerminatedWorkerError`. This is a harness-level constraint (`crunch/monkey_patches.py`), not a bug in the feature code — reverted to serial extraction. **Result: 0.5245, +0.0051 over row 8's 0.5194** — now clearly ahead of `ks_test`'s 0.5170, not just barely tied with it. Wall-clock: `train()` 22:09, `infer()` 2:49, determinism check 13s (passed), total 25:13. **Honest attribution check, done specifically because the gain could plausibly have come from something other than the new features:** permutation importance on a 300-series held-out sample shows `t_idx` now dominates overwhelmingly (+0.178 AUC drop, an order of magnitude above everything else), followed by `z_std`/`z_mad_diff`/`roll_std` (+0.017 to +0.026) — the 6 new wavelet/spectral features all rank in the bottom half individually (+0.0002 to +0.0035 each, `wave_e1` smallest, `spec_entropy` largest). To separate "new features contribute nothing, gain is noise/`t_idx` reshuffling" from "new features contribute a small but real amount," ran a matched 5-fold `GroupKFold` ablation (1500 held-out series, groups = series so no series splits across train/test) comparing the row-8 14-feature set against the row-9 20-feature set under identical `HistGradientBoostingClassifier` settings: 14-feature mean AUC 0.6842 (fold range 0.666–0.697) vs. 20-feature mean AUC 0.6859 (fold range 0.671–0.701), a consistent, small, reproducible edge (20-feature set won or tied in all 5 folds) — **not zero, but modest, and secondary to `t_idx`/variance-family features as the main drivers.** Conclusion: the wavelet/spectral decomposition is a genuine, positive contribution, not a fluke, but it is a refinement on top of an existing signal rather than the qualitative unlock the Tier-1→Tier-2→Tier-3 jump represented. **`t_idx`'s new dominance was checked directly and is legitimate, not a leak:** computed `P(label=1 \| t_idx)` marginally (i.e. ignoring series content entirely) on a 2,000-series sample — it rises monotonically and substantially, from ~1% near `t_idx=0` to ~49% by `t_idx=800-1600`. Mechanism: `tau` (break point) is close to uniformly distributed as a *fraction* of each series' online length (`tau/n_online` mean 0.505, std 0.287, roughly uniform on [0,1]), while online-segment lengths themselves vary widely (10 to 999 points, mean 505). At any fixed absolute `t_idx`, more of the *shorter* series in the mix have already passed their `tau` than the *longer* ones have — so "how many points have I seen so far" carries real information about the prior probability of already being past a break, entirely independent of what the series values actually show. This is exactly the kind of prior a genuine streaming detector is allowed to use (it only requires knowledge of the past, unlike the removed `t_frac`, which needed the *future* total length) — `t_idx`'s growing importance here reflects the model learning a real property of how `tau` is distributed in this dataset, not a pipeline artifact. See updated §5. |

## 5. Open issue: why is even the best result still ~0.5?

**Update 2026-09-01: pipeline/harness bug ruled out. This is now believed to be a
genuine modeling ceiling for "instant" per-step detectors, not a bug.**

Four structurally different detectors — EWMA (mean-shift only), `ks_test`
(whole-sample KS), `ks_multi` (three-view whole-sample KS), and `feature`
(rolling-window statistics vs. a historical window-distribution) — all land in the
0.50-0.52 band. The `feature` row (row 5 above) closes out every diagnostic this
section originally asked for:

- [x] **Runtime check.** `ks_test`/`ks_multi`/`feature` all ran measurably slower than
      the O(1) `ewma` baseline (multi-minute runs vs. seconds), confirming the intended
      `SCORING_METHOD` was actually exercised each time. (This also uncovered and fixed
      a separate **stale-execution bug**: editing the hyperparameters cell without
      re-running it left `infer()` closed over an old `SCORING_METHOD` value from a
      prior kernel session. Fix was a plain kernel restart + top-to-bottom re-run, no
      code change. See `RUN_KS_MULTI_EXPERIMENT.md` for the incident writeup. Take away:
      always restart the kernel before trusting a `SCORING_METHOD` change locally.)
- [x] **Score variance check.** All four methods produce non-degenerate prediction
      distributions (`feature`: std 0.19, full [0,1] range) — no score collapse.
- [x] **Alignment check.** Oracle scoring (substituting the ground-truth `target` as
      the "prediction") gives a perfect TS-AUC of 1.0, which proves the `(id, time)`
      merge and the TS-AUC computation itself are correct. There is no silent
      misalignment.
- [x] EWMA is also in the ~0.50 band (implicitly, from the original bug report of a
      constant ~0.5038 across methods before the stale-execution fix) — consistent
      with the bug being upstream of any single scoring method, except now we know
      that "upstream" is not the harness; it is the *detection paradigm* itself.

**Update 2026-09-01, part 2: the "cumulative vs. instant evidence" hypothesis is now
ruled out too.** CUSUM (row 6, TS-AUC 0.5024) and Page-Hinkley (row 7, TS-AUC 0.5058)
are both genuinely non-forgetting, cumulative statistics — CUSUM's `g+`/`g-` only reset
via their `max(0, ...)` floor, and Page-Hinkley's `PH_t - min(PH)` never decays. Both
were verified on synthetic data first (CUSUM: 0.99/0.97/0.81 on mean/variance/autocorr
breaks; Page-Hinkley: 0.996/0.75/0.59) so the implementations are sound. Neither did
meaningfully better than the windowed methods on the real data. Full comparison:

| Method | Family | Real TS-AUC | Synthetic AUC range |
|---|---|---|---|
| `ewma` | instant, mean-only | not recorded | — |
| `ks_test` | instant, whole-sample | 0.5170 | — |
| `ks_multi` | instant, whole-sample (3 views) | 0.5046 | 0.39-0.71 |
| `feature` | instant, fixed window | 0.5026 | 0.96-0.99 |
| `cusum` | **cumulative** | 0.5024 | 0.82-0.99 |
| `page_hinkley` | **cumulative** | **0.5058** | 0.59-1.00 |

Six structurally different detectors, two of them explicitly non-forgetting, all land
within a 0.0046 TS-AUC band. This strongly favors the surviving hypothesis:

- [x] ~~Cumulative vs. instant evidence~~ -- ruled out. Being non-forgetting does not
      help; `cusum`/`page_hinkley` are statistically indistinguishable from the
      windowed methods.
- [x] **Effect-size mismatch vs. synthetic tests -- confirmed, and still the leading
      explanation, though row 9 softens it slightly.** Row 8's `learned` classifier
      (Tier 3) trained on the *union* of every prior method's features and only reached
      TS-AUC 0.5194, barely above `ks_test`'s 0.5170. Row 9 then added a structurally
      different feature family (wavelet/spectral decomposition, not just more summary
      statistics of the same kind) and moved the number further, to 0.5245 -- a real,
      ablation-confirmed gain, so "more/different features can still help" is not
      fully dead. But the *size* of that gain (+0.0051, and the new features rank
      lowest in permutation importance individually) is consistent with, not
      contradictory to, the effect-size-mismatch story: each additional family of
      features finds a little more signal, but the returns are small and slowing, not
      opening up a new regime. Permutation importance across every Tier 1-3.5 method
      converges on the same story: variance-type features (`roll_std`/`z_std`) and now
      `t_idx` (confirmed legitimate, not a leak -- see row 9) are consistently the
      strongest signal, autocorrelation-type and the new scale/frequency-domain
      features are consistently the weakest, regardless of what model combines them.

**What moved the needle by how much, ranked:**

| Rank | Method | Real TS-AUC | Delta vs. floor (~0.502) |
|---|---|---|---|
| 1 | `learned` (Tier 3.5, 20 features) | **0.5245** | +0.023 |
| 2 | `learned` (Tier 3, 14 features) | 0.5194 | +0.017 |
| 3 | `ks_test` v2 | 0.5170 | +0.015 |
| 4 | `page_hinkley` | 0.5058 | +0.004 |
| 5 | `ks_multi` | 0.5046 | +0.003 |
| 6 | `feature` | 0.5026 | +0.001 |
| 7 | `cusum` | 0.5024 | +0.000 |

Nine rows across three tiers (plus one sub-tier) of increasingly sophisticated
approaches have produced a total spread of about 0.023 TS-AUC. Row 9 (adding
wavelet/spectral decomposition features) is the first result that clearly separates
from `ks_test` rather than merely tying it, and the matched-ablation check in row 9's
Notes confirms the new features contribute a small, real, reproducible amount on top
of row 8's set (not just noise from retraining) — but the improvement is incremental,
not a step-change: `t_idx` and the variance-family features (`z_std`, `roll_std`,
`z_mad_diff`) remain the dominant drivers by a wide margin. **Two options going
forward, not mutually exclusive:**

- [ ] **Push Tier 3 further before concluding this is the ceiling.** Ideas not yet
      tried: multiple window sizes (not just `FEAT_WINDOW=30`) so the model can compare
      short- and long-range statistics simultaneously; a genuinely different model
      class (e.g. a small RNN/1D-CNN over the raw windowed values instead of
      hand-engineered summary features); or tuning `HistGradientBoostingClassifier`'s
      `max_iter` upward now that a real number (0.5245) exists to weigh against the
      `predict_proba` per-call latency cost documented in row 8.
- [ ] **Accept ~0.52-0.53 as close to the true ceiling for this dataset's Tier 1-3.5
      granularity and reconsider whether Tier 4 (foundation-model embeddings, per §1)
      or a fundamentally different representation of the raw series (not just
      summary statistics or scale/frequency decompositions of it) is needed to find
      more signal.**

## 6. Reproducibility checklist (carried from the context file, made concrete here)

- [ ] `requirements.txt` pinned (see §3)
- [ ] Row 1 of the experiment log filled in before drawing further comparisons
- [ ] `ks_test_scores()` and `infer()`'s `ks_test` branch verified identical after any
      future edit to either (see §2)
- [ ] No random seeds needed yet — neither scorer uses randomness; this changes once
      Tier 3 starts and must be added here at that point
