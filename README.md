# Paper 2 — Recording-quality variation and clinical speech inference in ALS

This repository contains the frozen computational analysis for Paper 2, which studies how recording-quality characteristics relate to clinical information and acoustic-model inference in remote ALS speech recordings.

## Status

Goals 1–3 are computationally complete and frozen.

The original immutable scientific-analysis freeze is:

- tag: `paper2-analysis-freeze-v1.0.0`
- commit: `55cba2083327aa71be5ea7771a3185daefa0c979`

Post-freeze repository changes are restricted to provenance, reproducibility, and documentation corrections. They do not redefine analysis populations, estimands, models, resampling, perturbation doses, feature definitions, or manuscript-facing numerical results.

## Scientific structure

- **Phase 0 — data freeze and audit:** construct canonical participant/recording ledgers, severity matching, Q registry, deterministic row selection, participant-grouped split manifests, and input hashes.
- **Goal 1 — information availability:** test whether the frozen recording-quality representation Q contains reproducible out-of-sample information about ALS diagnosis or contemporaneous bulbar function.
- **Goal 2 — consequence for acoustic inference:** test whether inference from a frozen clinical acoustic representation A changes when Q is added or when Q-predictable variation is removed from A.
- **Goal 3 — localization and controlled sensitivity:** localize natural Q–A coupling and test direct sensitivity of Q, A, and frozen clinical predictions to prespecified same-source acquisition perturbations.

The interpretation boundary is deliberately conservative. Natural-data associations do not establish technical cause, confounding, artifact, or shortcut learning. Controlled perturbations support direct sensitivity to the imposed transformation; they do not prove that the same transformation generated an observational clinical association.

## Core validation contract

Participant is the independent unit for train/test splitting, permutation, and bootstrap resampling. No held-out participant contributes to training-fold preprocessing, tuning, QCHAN reference construction, residualization, or calibration.

Primary repeated internal validation uses 5 outer folds × 10 repeats with participant grouping and base seed `20260825`. Participant-level bootstrap confidence intervals use 2,000 replicates. Goal 1 formal permutation inference uses 1,000 participant-level permutations per task.

## Authoritative computational lineage

The repository intentionally retains historical notebook variants for auditability. Do not infer authority from filename order alone. The final lineage and supersession map are in [`docs/ANALYSIS_LINEAGE.md`](docs/ANALYSIS_LINEAGE.md), and the notebook-oriented execution map is in [`notebooks/README.md`](notebooks/README.md).

A post-freeze provenance correction adds `notebooks/35_goal3_controlled_perturbation_PRIMARY_FINAL_v1_1_1.ipynb`. This is the final Stage-E completion patch used to create the retained prediction-unavailability audit. Relative to v1.1.0, it fixes one support-summary reduction and adds explicit baseline-gated prediction-unavailability tables; it does not rerun or change the 14,792 waveform measurements, frozen perturbation doses, model states, prediction rules, or inferential estimands.

## Data and external dependencies

Raw audio, participant-level clinical data, adjudication files, and other restricted source data are not committed to this repository. Generated `outputs/` are also private by default. The public repository therefore supports code inspection and reconstruction by authorized users with the governed inputs; it is not a public raw-data release.

Goal 3 reuses the Paper-1 measurement implementation pinned at commit:

`cb31fb6886df1b2b2fedba4ffbbf8624bd56d7e8`

See [`docs/RUNTIME_PROVENANCE.md`](docs/RUNTIME_PROVENANCE.md) for the actual recorded Stage-E execution environment and the documented Python-version compatibility divergence.

## Environment

`environment.yml` is the project reconstruction environment, not a bit-for-bit historical lockfile. The exact critical Stage-E package versions recorded in the frozen outputs are listed in `requirements-goal3-stageE-historical-critical.txt`. A supported reconstruction environment for the pinned Paper-1 implementation is provided in `environment-goal3-supported.yml`.

The full frozen-output audit record is in [`docs/FROZEN_OUTPUT_AUDIT.md`](docs/FROZEN_OUTPUT_AUDIT.md). A standard-library verifier for an extracted or zipped frozen output package is provided at `tools/verify_frozen_outputs.py`.
