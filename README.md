# Paper 2 â€” Recording Quality and Clinical Inference

Analysis repository for Paper 2: **recording-quality variation and machine-learning inference from remote speech in ALS**.

## Scientific structure

- **Phase 0 â€” Data freeze and audit:** canonical participant/recording ledgers, severity matching, Q registry, deterministic row selection, split manifest, and input hashes.
- **Goal 1 â€” Information availability:** test whether frozen recording-quality features (Q) contain reproducible out-of-sample information about ALS diagnosis or contemporaneous bulbar function.
- **Goal 2 â€” Consequence for acoustic inference:** test whether inference from frozen clinical acoustic features (A) changes when Q is added, residualized, or varies naturally.
- **Goal 3 â€” Localization and perturbation:** localize Qâ€“A coupling and test controlled acquisition perturbations on held-out recordings.

The frozen Methods Reference is the analysis contract. Goal 1 can proceed after Phase 0. Goals 2â€“3 remain blocked until the acoustic registry **A** is frozen, versioned, checked for mechanical redundancy with Q, and hashed.

## Repository layout

```text
configs/        machine-readable analysis settings
data/           local inputs and derived data; sensitive data are not committed
docs/           frozen methods/specification notes
notebooks/      interactive Phase 0 and Goal 1â€“3 analyses
outputs/        generated results; private by default
src/paper2/     reusable code after notebook validation
tests/          reproducibility and leakage-safety tests
```

## Reproducibility contract

Participant is the independent unit for splitting, permutation, and bootstrap resampling. No held-out participant may contribute to preprocessing, QCHAN reference construction, tuning, residualization, or calibration. All data-dependent operations occur inside the relevant training fold.

Primary repeated internal validation: **5 outer folds Ã— 10 repeats**, participant grouped, base seed **20260825**.

## Status

Repository initialized. Next: populate local `data/` inputs and run the Phase 0 canonical audit before any Goal 1 model is fitted.
