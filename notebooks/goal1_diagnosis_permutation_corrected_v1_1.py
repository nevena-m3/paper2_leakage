# Goal 1 — corrected diagnosis permutation engine
# Version 1.1
#
# Run from the LIVE Notebook 10 FINAL kernel after the earlier accelerated
# fixed-split diagnostic has demonstrated the fold-prevalence artifact:
#
#     %run -i "goal1_diagnosis_permutation_corrected_v1_1.py"
#
# This script:
#   1) preserves/labels the earlier fixed-stratified-manifest diagnosis null as
#      invalid for final inference;
#   2) uses the prespecified computational fallback of 5 folds x 3 repeats;
#   3) globally permutes diagnosis at participant level;
#   4) REGENERATES outer StratifiedKFold assignments from the permuted labels
#      using the same deterministic repeat seeds;
#   5) lets Notebook 10 regenerate inner stratified folds from the permuted
#      labels through its authoritative tune_model() path;
#   6) reruns fold-safe QCHAN, preprocessing, ridge tuning, fitting and OOF
#      scoring under every null draw;
#   7) checkpoints each completed permutation;
#   8) runs only a 4-permutation benchmark automatically.  The same function
#      can then be resumed to 1000 without losing work.
#
# IMPORTANT:
# - Primary 10-repeat observed Goal-1 OOF estimates are NOT replaced.
# - The formal diagnosis permutation p-value uses a matched 3-repeat statistic:
#   observed first 3 true-label stratified repeats versus 1000 null draws using
#   the same 3-repeat algorithm with stratification regenerated after shuffling.
# - The earlier fixed-manifest diagnosis null must never be used for inference.

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits


# -------------------------------------------------------------------------
# 0. Hard context / provenance gates
# -------------------------------------------------------------------------

_REQUIRED = [
    "RUN_MODE",
    "BASE_SEED",
    "OUTER_FOLDS",
    "INNER_FOLDS",
    "N_PERMUTATIONS",
    "OUT",
    "FAST_PERM_AUDIT",
    "fast_primary_frame",
    "PRIMARY_MODEL_SPECS",
    "model_checkpoint_paths",
    "safe_read_csv",
    "repeat_metric_table",
    "run_nested_cv",
    "split_manifest",
    "stable_hash",
    "atomic_write_csv",
    "atomic_write_json",
]

_missing = [name for name in _REQUIRED if name not in globals()]
if _missing:
    raise RuntimeError(
        "Run this script inside the live Notebook 10 kernel after the "
        "accelerated-permutation diagnostic cells. Missing globals: "
        + ", ".join(_missing)
    )

if RUN_MODE != "FINAL":
    raise RuntimeError(
        f"Corrected diagnosis permutation requires RUN_MODE='FINAL'; got {RUN_MODE!r}."
    )

if int(OUTER_FOLDS) != 5 or int(INNER_FOLDS) != 5:
    raise RuntimeError("Expected the frozen 5-fold outer / 5-fold inner contract.")

if int(N_PERMUTATIONS) != 1000:
    raise RuntimeError(
        f"Expected 1000 formal permutations; observed N_PERMUTATIONS={N_PERMUTATIONS}."
    )

CORRECTED_ENGINE = "goal1-diagnosis-permutation-regenerated-stratification-v1.1.0"
CORRECTED_REPEATS = 3
BENCHMARK_TARGET = 4

CORRECTED_ROOT = OUT / "corrected_diagnosis_permutation_v1_1"
CORRECTED_CHECKPOINT = CORRECTED_ROOT / "diagnosis_permutation_checkpoint.csv"
CORRECTED_AUDIT = CORRECTED_ROOT / "audit"
CORRECTED_TABLES = CORRECTED_ROOT / "tables"

for _d in [CORRECTED_ROOT, CORRECTED_AUDIT, CORRECTED_TABLES]:
    _d.mkdir(parents=True, exist_ok=True)

# Preserve the master split object exactly as it existed before any worker
# temporarily swaps run_nested_cv's global split_manifest.
MASTER_SPLIT_MANIFEST_FOR_AUDIT = split_manifest.copy(deep=True)

# Exact primary Core-Q specification.
CORRECTED_COREQ_SPEC = PRIMARY_MODEL_SPECS["Core-Q"]
if CORRECTED_COREQ_SPEC.get("kind") != "ridge":
    raise RuntimeError("Primary Core-Q model is unexpectedly not ridge.")

# Formal one-row diagnosis population.
_corrected_dx = fast_primary_frame("diagnosis").copy()
_corrected_dx["participant_id"] = (
    _corrected_dx["participant_id"].astype(str).str.strip()
)
_corrected_dx = (
    _corrected_dx.sort_values("participant_id")
    .reset_index(drop=True)
)

if len(_corrected_dx) != 224:
    raise RuntimeError(
        f"Expected 224 diagnosis participants; found {len(_corrected_dx)}."
    )
if not _corrected_dx["participant_id"].is_unique:
    raise RuntimeError("Diagnosis permutation population is not one row per participant.")

_dx_y = _corrected_dx["y"].to_numpy(int)
_dx_counts = pd.Series(_dx_y).value_counts().to_dict()
if set(_dx_counts) != {0, 1}:
    raise RuntimeError(f"Diagnosis classes are invalid: {_dx_counts}")

# Earlier diagnostic evidence is required before changing the permutation engine.
_fixed_diag_path = FAST_PERM_AUDIT / "diagnosis_fixed_fold_null_diagnostic.csv"
_fixed_prev_path = FAST_PERM_AUDIT / "diagnosis_fixed_fold_prevalence_diagnostic.csv"

if not _fixed_diag_path.exists() or not _fixed_prev_path.exists():
    raise FileNotFoundError(
        "The fixed-fold permutation validity diagnostic must be run first."
    )

_fixed_diag = pd.read_csv(_fixed_diag_path)
_fixed_prev = pd.read_csv(_fixed_prev_path)

_fixed_intercept_mean = float(
    _fixed_diag["intercept_only_fixed_fold_AUROC"].mean()
)
_fixed_coreq_mean = float(_fixed_diag["CoreQ_null_AUROC"].mean())
_fixed_prev_corr = float(
    _fixed_prev[["train_prevalence", "test_prevalence"]]
    .corr()
    .iloc[0, 1]
)

if not (
    _fixed_intercept_mean < 0.46
    and _fixed_prev_corr < -0.95
):
    raise RuntimeError(
        "The recorded diagnostic does not reproduce the fixed-fold "
        "prevalence artifact strongly enough to justify this correction."
    )


# -------------------------------------------------------------------------
# 1. Exact label-dependent outer split constructor
# -------------------------------------------------------------------------

def corrected_outer_manifest(frame: pd.DataFrame, repeats: int = CORRECTED_REPEATS):
    """
    Recreate the Phase-0 diagnosis splitting algorithm after the supplied
    outcome labels have been assigned.

    Participant ordering and repeat seeds match the Phase-0 contract:
      sorted participant_id
      StratifiedKFold(5, shuffle=True, random_state=BASE_SEED + repeat_index)
    """
    participant = (
        frame[["participant_id", "y"]]
        .copy()
        .assign(
            participant_id=lambda d: d["participant_id"].astype(str).str.strip()
        )
        .sort_values("participant_id")
        .reset_index(drop=True)
    )

    if not participant["participant_id"].is_unique:
        raise RuntimeError("Outer split constructor requires one row per participant.")

    labels = participant["y"].to_numpy(int)
    counts = pd.Series(labels).value_counts()
    if counts.min() < OUTER_FOLDS:
        raise RuntimeError(
            f"Insufficient class count for stratified outer CV: {counts.to_dict()}"
        )

    rows = []

    for repeat_index in range(int(repeats)):
        repeat_seed = int(BASE_SEED + repeat_index)

        splitter = StratifiedKFold(
            n_splits=int(OUTER_FOLDS),
            shuffle=True,
            random_state=repeat_seed,
        )

        for outer_fold, (_, test_idx) in enumerate(
            splitter.split(
                participant["participant_id"].to_numpy(),
                labels,
            ),
            start=1,
        ):
            test = participant.iloc[test_idx]

            for row in test.itertuples(index=False):
                rows.append(
                    {
                        "participant_id": str(row.participant_id),
                        "repeat": int(repeat_index + 1),
                        "outer_fold": int(outer_fold),
                        "repeat_seed": repeat_seed,
                    }
                )

    manifest = pd.DataFrame(rows)

    expected_rows = len(participant) * int(repeats)
    if len(manifest) != expected_rows:
        raise RuntimeError(
            f"Corrected split row count mismatch: expected {expected_rows}, "
            f"got {len(manifest)}."
        )

    if not manifest.groupby(["participant_id", "repeat"]).size().eq(1).all():
        raise RuntimeError("Corrected outer split assignment is not unique.")

    return manifest


# -------------------------------------------------------------------------
# 2. Gate: on TRUE labels this split generator must reproduce the first
#    three frozen Phase-0 master repeats exactly.
# -------------------------------------------------------------------------

_true_manifest = corrected_outer_manifest(
    _corrected_dx,
    repeats=CORRECTED_REPEATS,
)

_master3 = (
    MASTER_SPLIT_MANIFEST_FOR_AUDIT.loc[
        MASTER_SPLIT_MANIFEST_FOR_AUDIT["repeat"].between(
            1, CORRECTED_REPEATS
        ),
        ["participant_id", "repeat", "outer_fold"],
    ]
    .copy()
)
_master3["participant_id"] = _master3["participant_id"].astype(str).str.strip()

_true_compare = (
    _master3.rename(columns={"outer_fold": "master_outer_fold"})
    .merge(
        _true_manifest[
            ["participant_id", "repeat", "outer_fold"]
        ].rename(columns={"outer_fold": "regenerated_outer_fold"}),
        on=["participant_id", "repeat"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
)

if not _true_compare["_merge"].eq("both").all():
    raise RuntimeError(
        "True-label regenerated split manifest does not align with the "
        "frozen master participant/repeat identities."
    )

if not (
    _true_compare["master_outer_fold"].astype(int).to_numpy()
    == _true_compare["regenerated_outer_fold"].astype(int).to_numpy()
).all():
    mismatch = _true_compare.loc[
        _true_compare["master_outer_fold"]
        != _true_compare["regenerated_outer_fold"]
    ].head(20)

    atomic_write_csv(
        mismatch,
        CORRECTED_AUDIT / "true_label_split_mismatch.csv",
    )

    raise RuntimeError(
        "True-label regenerated stratified folds do not exactly reproduce "
        "Phase-0 repeats 1-3."
    )

atomic_write_csv(
    _true_compare,
    CORRECTED_AUDIT / "true_label_split_reproduction.csv",
)


# -------------------------------------------------------------------------
# 3. Matched 3-repeat observed statistic
# -------------------------------------------------------------------------

_dx_oof_path = model_checkpoint_paths("diagnosis", "Core-Q")["oof"]

_dx_primary_oof = safe_read_csv(
    _dx_oof_path,
    required_columns=[
        "participant_id",
        "logical_recording_id",
        "y",
        "prediction",
        "repeat",
        "outer_fold",
        "model",
        "task",
    ],
)

_dx_observed3 = _dx_primary_oof.loc[
    _dx_primary_oof["repeat"].between(1, CORRECTED_REPEATS)
].copy()

_dx_observed3_repeat = repeat_metric_table(
    _dx_observed3,
    "diagnosis",
)

CORRECTED_OBSERVED_AUROC = float(
    _dx_observed3_repeat["AUROC"].mean()
)

# Keep the 10-repeat primary estimate visible but separate.
_dx_observed10_repeat = repeat_metric_table(
    _dx_primary_oof,
    "diagnosis",
)

PRIMARY_10_REPEAT_AUROC = float(
    _dx_observed10_repeat["AUROC"].mean()
)


# -------------------------------------------------------------------------
# 4. Cheap structural diagnostic for regenerated stratification
# -------------------------------------------------------------------------

def corrected_intercept_only_auc(permutation: int):
    permuted = _corrected_dx.copy()

    rng = np.random.default_rng(
        int(BASE_SEED + 10_000_000 + permutation)
    )
    permuted["y"] = rng.permutation(
        permuted["y"].to_numpy(int)
    )

    manifest = corrected_outer_manifest(
        permuted,
        repeats=CORRECTED_REPEATS,
    )

    label_lookup = dict(
        zip(
            permuted["participant_id"].astype(str),
            permuted["y"].astype(int),
        )
    )

    repeat_aucs = []

    all_ids = set(permuted["participant_id"].astype(str))

    for repeat in range(1, CORRECTED_REPEATS + 1):
        y_all = []
        p_all = []

        for outer_fold in range(1, OUTER_FOLDS + 1):
            test_ids = set(
                manifest.loc[
                    manifest["repeat"].eq(repeat)
                    & manifest["outer_fold"].eq(outer_fold),
                    "participant_id",
                ].astype(str)
            )

            train_ids = all_ids - test_ids

            y_train = np.asarray(
                [label_lookup[pid] for pid in sorted(train_ids)],
                dtype=float,
            )
            y_test = np.asarray(
                [label_lookup[pid] for pid in sorted(test_ids)],
                dtype=float,
            )

            train_prevalence = float(y_train.mean())

            y_all.extend(y_test.tolist())
            p_all.extend(
                np.full(len(y_test), train_prevalence).tolist()
            )

        repeat_aucs.append(
            float(
                roc_auc_score(
                    np.asarray(y_all, dtype=int),
                    np.asarray(p_all, dtype=float),
                )
            )
        )

    return float(np.mean(repeat_aucs))


_intercept_values = np.asarray(
    [corrected_intercept_only_auc(p) for p in range(1, 201)],
    dtype=float,
)

CORRECTED_INTERCEPT_MEAN = float(_intercept_values.mean())

# Regenerated stratification cannot make integer fold prevalences exactly
# identical with 66 controls / 158 ALS, so the intercept-only OOF AUROC is
# expected to be slightly below 0.5.  What must disappear is the severe ~0.42
# artifact from holding the original label-stratified folds fixed.
if not (0.47 <= CORRECTED_INTERCEPT_MEAN <= 0.51):
    raise RuntimeError(
        "Regenerated-stratification intercept diagnostic is still too far "
        f"from chance: {CORRECTED_INTERCEPT_MEAN:.6f}"
    )

atomic_write_csv(
    pd.DataFrame(
        {
            "permutation": np.arange(1, 201),
            "intercept_only_regenerated_stratification_AUROC": (
                _intercept_values
            ),
        }
    ),
    CORRECTED_AUDIT
    / "regenerated_stratification_intercept_diagnostic.csv",
)


# -------------------------------------------------------------------------
# 5. Provenance / correction memo
# -------------------------------------------------------------------------

CORRECTED_SIGNATURE = stable_hash(
    {
        "engine": CORRECTED_ENGINE,
        "base_seed": BASE_SEED,
        "outer_folds": OUTER_FOLDS,
        "outer_repeats_for_permutation": CORRECTED_REPEATS,
        "inner_folds": INNER_FOLDS,
        "n_permutations": N_PERMUTATIONS,
        "primary_model": "Core-Q ridge",
        "outer_rule": (
            "StratifiedKFold regenerated from the current observed/permuted "
            "diagnosis using seeds BASE_SEED + repeat_index"
        ),
        "inner_rule": (
            "authoritative Notebook 10 diagnosis inner stratification "
            "regenerated from the current outer-training permuted labels"
        ),
        "observed_statistic": (
            "mean participant-level AUROC across true-label repeats 1-3"
        ),
        "null_statistic": (
            "mean participant-level AUROC across matched regenerated "
            "stratified repeats 1-3"
        ),
    }
)

atomic_write_json(
    {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "engine": CORRECTED_ENGINE,
        "signature": CORRECTED_SIGNATURE,
        "status": "PRE_FINAL_BENCHMARK",
        "reason_for_correction": (
            "The primary master folds were created by stratifying on the true "
            "diagnosis. Holding those assignments fixed after globally "
            "permuting diagnosis leaked information from the original labels "
            "into the null split structure and induced a severe cross-fold "
            "prevalence artifact."
        ),
        "diagnostic_evidence": {
            "mean_CoreQ_null_AUROC_first8_fixed_manifest": (
                _fixed_coreq_mean
            ),
            "mean_intercept_only_AUROC_first8_fixed_manifest": (
                _fixed_intercept_mean
            ),
            "train_test_permuted_prevalence_correlation": (
                _fixed_prev_corr
            ),
            "mean_intercept_only_AUROC_regenerated_stratification_200_draws": (
                CORRECTED_INTERCEPT_MEAN
            ),
        },
        "inference_contract": {
            "participant_level_permutation": True,
            "n_permutations": int(N_PERMUTATIONS),
            "outer_folds": int(OUTER_FOLDS),
            "outer_repeats": int(CORRECTED_REPEATS),
            "inner_folds": int(INNER_FOLDS),
            "outer_stratification_regenerated_after_each_permutation": True,
            "inner_stratification_regenerated_after_each_permutation": True,
            "fold_safe_QCHAN_recomputed": True,
            "preprocessing_recomputed": True,
            "ridge_tuning_recomputed": True,
            "model_fit_recomputed": True,
            "old_fixed_manifest_null_used_for_inference": False,
        },
        "primary_10_repeat_AUROC_preserved": PRIMARY_10_REPEAT_AUROC,
        "matched_3_repeat_observed_AUROC_for_permutation_test": (
            CORRECTED_OBSERVED_AUROC
        ),
    },
    CORRECTED_AUDIT / "diagnosis_permutation_correction_contract.json",
)


# -------------------------------------------------------------------------
# 6. One corrected permutation
# -------------------------------------------------------------------------

def _single_corrected_diagnosis_permutation(permutation: int):
    permutation = int(permutation)

    permuted = _corrected_dx.copy()

    label_seed = int(
        BASE_SEED + 10_000_000 + permutation
    )
    rng = np.random.default_rng(label_seed)

    permuted["y"] = rng.permutation(
        permuted["y"].to_numpy(int)
    )

    local_manifest = corrected_outer_manifest(
        permuted,
        repeats=CORRECTED_REPEATS,
    )

    split_hash = stable_hash(
        local_manifest[
            ["participant_id", "repeat", "outer_fold", "repeat_seed"]
        ].sort_values(
            ["repeat", "outer_fold", "participant_id"]
        ).to_dict("records")
    )

    # run_nested_cv reads split_manifest from its defining global namespace.
    # Each joblib worker is a separate process, so this swap is isolated to
    # that worker.
    function_globals = run_nested_cv.__globals__
    prior_manifest = function_globals.get("split_manifest")

    try:
        function_globals["split_manifest"] = local_manifest

        with threadpool_limits(limits=1):
            null_oof, _, _, _ = run_nested_cv(
                permuted,
                f"Core-Q corrected permutation {permutation}",
                CORRECTED_COREQ_SPEC,
                "diagnosis",
                repeats=CORRECTED_REPEATS,
                record_splits=False,
            )

        metric_table = repeat_metric_table(
            null_oof,
            "diagnosis",
        )

        null_metric = float(
            metric_table["AUROC"].mean()
        )

    finally:
        function_globals["split_manifest"] = prior_manifest

    return {
        "permutation": permutation,
        "null_metric": null_metric,
        "label_permutation_seed": label_seed,
        "outer_repeat_seeds": ";".join(
            str(BASE_SEED + r)
            for r in range(CORRECTED_REPEATS)
        ),
        "split_manifest_hash": split_hash,
        "engine": CORRECTED_ENGINE,
        "signature": CORRECTED_SIGNATURE,
    }


# -------------------------------------------------------------------------
# 7. Restart-safe parallel runner
# -------------------------------------------------------------------------

def run_corrected_diagnosis_permutations(
    *,
    target: int,
    n_jobs: int = 4,
):
    """
    Run/resume corrected diagnosis permutations through `target`.

    Examples:
        benchmark, seconds, n_new = run_corrected_diagnosis_permutations(
            target=4, n_jobs=4
        )

        final, seconds, n_new = run_corrected_diagnosis_permutations(
            target=1000, n_jobs=8
        )
    """
    target = int(target)
    n_jobs = int(n_jobs)

    if target < 1 or target > int(N_PERMUTATIONS):
        raise ValueError(
            f"target must be between 1 and {N_PERMUTATIONS}."
        )
    if n_jobs < 1:
        raise ValueError("n_jobs must be >=1.")

    required = [
        "permutation",
        "null_metric",
        "label_permutation_seed",
        "outer_repeat_seeds",
        "split_manifest_hash",
        "engine",
        "signature",
    ]

    if CORRECTED_CHECKPOINT.exists():
        existing = safe_read_csv(
            CORRECTED_CHECKPOINT,
            required_columns=required,
            allow_empty=True,
        )

        if len(existing):
            if set(existing["signature"].astype(str)) != {
                CORRECTED_SIGNATURE
            }:
                raise RuntimeError(
                    "Existing corrected diagnosis permutation checkpoint has "
                    "an incompatible signature."
                )
    else:
        existing = pd.DataFrame(columns=required)

    done = set(
        pd.to_numeric(
            existing.get("permutation", pd.Series(dtype=float)),
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )

    todo = [
        p for p in range(1, target + 1)
        if p not in done
    ]

    if not todo:
        print(
            f"Corrected diagnosis permutations 1-{target} already complete."
        )
        return (
            existing.sort_values("permutation").reset_index(drop=True),
            0.0,
            0,
        )

    print(
        f"Corrected diagnosis permutation: {len(todo)} new draws | "
        f"target={target} | workers={n_jobs} | "
        f"CV={OUTER_FOLDS} folds x {CORRECTED_REPEATS} repeats"
    )

    wall_start = time.perf_counter()
    working = existing.copy()

    # Small durable batches. A crash loses at most one batch.
    batch_width = max(n_jobs, 2 * n_jobs)

    for start in range(0, len(todo), batch_width):
        batch = todo[start:start + batch_width]
        batch_start = time.perf_counter()

        with parallel_backend(
            "loky",
            inner_max_num_threads=1,
        ):
            result = Parallel(
                n_jobs=n_jobs,
                batch_size=1,
                verbose=0,
            )(
                delayed(_single_corrected_diagnosis_permutation)(p)
                for p in batch
            )

        working = (
            pd.concat(
                [working, pd.DataFrame(result)],
                ignore_index=True,
            )
            .sort_values("permutation")
            .drop_duplicates("permutation", keep="last")
            .reset_index(drop=True)
        )

        atomic_write_csv(
            working,
            CORRECTED_CHECKPOINT,
            required_columns=required,
        )

        elapsed_batch = time.perf_counter() - batch_start

        print(
            f"completed through permutation {max(batch)} | "
            f"checkpoint rows={len(working)} | "
            f"batch wall={elapsed_batch / 60:.2f} min"
        )

    total_seconds = time.perf_counter() - wall_start

    return (
        working,
        total_seconds,
        len(todo),
    )


# -------------------------------------------------------------------------
# 8. Automatically run only four corrected permutations as a benchmark.
# -------------------------------------------------------------------------

print("=" * 78)
print("GOAL 1 — CORRECTED DIAGNOSIS PERMUTATION PREFLIGHT")
print("=" * 78)
print("Old fixed-manifest Core-Q null mean (8 draws):",
      f"{_fixed_coreq_mean:.6f}")
print("Old fixed-manifest intercept-only mean:",
      f"{_fixed_intercept_mean:.6f}")
print("Old train/test prevalence correlation:",
      f"{_fixed_prev_corr:.6f}")
print("Regenerated-stratification intercept-only mean (200 draws):",
      f"{CORRECTED_INTERCEPT_MEAN:.6f}")
print("Primary Core-Q AUROC, 10 repeats (preserved):",
      f"{PRIMARY_10_REPEAT_AUROC:.6f}")
print("Matched observed AUROC, repeats 1-3 (formal permutation statistic):",
      f"{CORRECTED_OBSERVED_AUROC:.6f}")
print("True-label split reproduction, repeats 1-3: PASS")
print("Corrected engine signature:", CORRECTED_SIGNATURE[:16])
print()

_cpu = os.cpu_count() or 2
BENCHMARK_WORKERS = max(1, min(4, _cpu // 2))

(
    corrected_dx_benchmark,
    corrected_benchmark_seconds,
    corrected_benchmark_new,
) = run_corrected_diagnosis_permutations(
    target=BENCHMARK_TARGET,
    n_jobs=BENCHMARK_WORKERS,
)

display(corrected_dx_benchmark.tail(BENCHMARK_TARGET))

if corrected_benchmark_new > 0:
    seconds_per_completed_draw_wall = (
        corrected_benchmark_seconds
        / corrected_benchmark_new
    )

    projected_hours_4_workers = (
        seconds_per_completed_draw_wall
        * 1000
        / 3600
    )

    print()
    print(
        "Observed wall seconds per completed permutation at current "
        f"{BENCHMARK_WORKERS}-worker throughput:",
        f"{seconds_per_completed_draw_wall:.2f}",
    )
    print(
        "Naive projected wall time for 1000 at this throughput:",
        f"{projected_hours_4_workers:.2f} hours",
    )

print()
print("CORRECTED DIAGNOSIS PERMUTATION BENCHMARK: COMPLETE")
print("Do NOT use the old fixed-manifest null.")
print(
    "Next, inspect this timing benchmark before calling "
    "run_corrected_diagnosis_permutations(target=1000, ...)."
)
