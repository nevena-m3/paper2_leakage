# Goal 3 Stage B v1.3 targeted signal-only revision
# Run ONLY from the live Notebook 32 FINAL kernel after v1.1 completed:
#     %run -i "goal3_stageB_v1_2_targeted_revision.py"
#
# This script deliberately does NOT load diagnosis, severity, acoustic A,
# Goal 2 predictions, losses, or clinical model responses.
#
# It revises only:
#   1) amplitude-modulated QADD structure,
#   2) dynamic QGAIN dose range,
#   3) QCHAN cutoff range.
#
# Passing v1.1 transforms are locked and are never recalibrated here.
# No final perturbation manifest is written automatically. If all gates pass,
# the script writes READY_FOR_SEAL.json for a final human/scientific audit.

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# 0. Hard context gate
# ---------------------------------------------------------------------

_REQUIRED_GLOBALS = [
    "ROOT",
    "RUN_MODE",
    "PAPER1_COMMIT",
    "pilot_run",
    "fixed_qchan_references",
    "decode_source",
    "extract_all_q",
    "colored_broadband_noise",
    "apply_dynamic_attenuation",
    "apply_qchan_lowpass",
    "CORE_Q",
    "QDIST_TARGET",
    "iqr_lookup",
    "stable_hash",
    "stable_seed",
    "json_safe",
]

_missing_globals = [name for name in _REQUIRED_GLOBALS if name not in globals()]
if _missing_globals:
    raise RuntimeError(
        "This targeted revision must be run inside the live Notebook 32 kernel. "
        f"Missing globals: {_missing_globals}"
    )

if RUN_MODE != "FINAL":
    raise RuntimeError(
        f"Notebook 32 must have completed in FINAL mode; observed RUN_MODE={RUN_MODE!r}."
    )

if len(pilot_run) != 24:
    raise RuntimeError(f"Expected 24 signal-only pilot sources; found {len(pilot_run)}.")

if len(fixed_qchan_references) != 5:
    raise RuntimeError("Expected five fixed outer-training QCHAN references.")

V13_ENGINE = "goal3-stageB-targeted-revision-v1.3.0"
V13_CREATED_UTC = datetime.now(timezone.utc).isoformat()

V12_FAILED_SCREEN = (
    ROOT
    / "outputs"
    / "goal3"
    / "stageB_signal_only_calibration_v1_2"
    / "final"
    / "tables"
    / "goal3_v12_am_structure_screen_summary.csv"
)

V11_OUT = (
    ROOT
    / "outputs"
    / "goal3"
    / "stageB_signal_only_calibration_v1_1"
    / "final"
)

V11_TABLES = V11_OUT / "tables"

V13_OUT = (
    ROOT
    / "outputs"
    / "goal3"
    / "stageB_signal_only_calibration_v1_2"
    / "final"
)

V13_TABLES = V13_OUT / "tables"
V13_AUDIT = V13_OUT / "audit"
V13_CHECKPOINTS = V13_OUT / "checkpoints"

for _directory in [V13_OUT, V13_TABLES, V13_AUDIT, V13_CHECKPOINTS]:
    _directory.mkdir(parents=True, exist_ok=True)

def v13_atomic_csv(frame: pd.DataFrame, path: Path, *, allow_empty=False):
    if frame.empty and not allow_empty:
        raise ValueError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)

def v13_atomic_json(payload: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temp, path)

def v13_sha256_file(path: Path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

_REQUIRED_V11 = {
    "response": V11_TABLES / "goal3_signal_only_candidate_response.csv",
    "delta": V11_TABLES / "goal3_signal_only_candidate_delta.csv",
    "summary": V11_TABLES / "goal3_signal_only_candidate_summary.csv",
    "selected": V11_TABLES / "goal3_signal_only_selected_doses.csv",
    "acceptance": V11_TABLES / "goal3_calibration_acceptance_gates.csv",
}

for _name, _path in _REQUIRED_V11.items():
    if not _path.exists():
        raise FileNotFoundError(f"Missing v1.1 {_name}: {_path}")

v11_response = pd.read_csv(_REQUIRED_V11["response"], low_memory=False)
v11_delta = pd.read_csv(_REQUIRED_V11["delta"], low_memory=False)
v11_summary = pd.read_csv(_REQUIRED_V11["summary"], low_memory=False)
v11_selected = pd.read_csv(_REQUIRED_V11["selected"], low_memory=False)
v11_acceptance = pd.read_csv(_REQUIRED_V11["acceptance"], low_memory=False)

PROHIBITED = {
    "diagnosis",
    "Diagnosis",
    "y_dx",
    "bulbar_score",
    "ALSFRS",
    "assessment_date",
    "age_at_recording_years",
    "sex",
    "prediction",
    "loss",
    "absolute_error",
    "brier",
}

for _name, _frame in [
    ("v11_response", v11_response),
    ("v11_delta", v11_delta),
    ("v11_summary", v11_summary),
    ("v11_selected", v11_selected),
    ("v11_acceptance", v11_acceptance),
    ("pilot_run", pilot_run),
]:
    _leaked = PROHIBITED & set(_frame.columns)
    if _leaked:
        raise RuntimeError(
            f"Clinical information found in {_name}: {sorted(_leaked)}"
        )

# ---------------------------------------------------------------------
# 1. Lock the four transforms that passed v1.1.
# ---------------------------------------------------------------------

LOCKED_TRANSFORMS = [
    "stationary_colored_broadband",
    "uniform_level_shift",
    "RIR_convolution_RMS_matched",
    "symmetric_hard_clipping",
]

for _transform in LOCKED_TRANSFORMS:
    _local = v11_acceptance.loc[v11_acceptance["transform"].eq(_transform)].copy()
    if len(_local) == 0:
        raise RuntimeError(f"Missing v1.1 acceptance rows for {_transform}.")
    if not _local["passed"].astype(bool).all():
        raise RuntimeError(
            f"Cannot lock {_transform}: not all v1.1 acceptance gates passed."
        )

locked_selected = v11_selected.loc[
    v11_selected["transform"].isin(LOCKED_TRANSFORMS)
].copy()

if len(locked_selected) != 12:
    raise RuntimeError(
        f"Expected 12 locked selected-dose rows (4 transforms x 3 doses), "
        f"found {len(locked_selected)}."
    )

v13_atomic_csv(
    locked_selected,
    V13_TABLES / "goal3_v11_locked_passing_doses.csv",
)

# ---------------------------------------------------------------------
# 2. Freeze the targeted v1.3 revision BEFORE running new signal response.
# ---------------------------------------------------------------------

# AM-QADD v1.1 failed across the SNR grid and v1.2 single-sinusoid
# log-depth modulation narrowly failed the fixed direction gate:
# 30-dB p-p reached a strong median target response but only 16/23 finite
# sources moved in the expected direction (0.695652 < frozen 0.70).
#
# We DO NOT relax or round the gate. We revise only the modulation structure,
# while remaining signal-only and keeping the same QADD construct:
# amplitude-modulated colored broadband interference.
#
# v1.3 structure:
# - colored broadband carrier unchanged
# - intermittent alternating log-amplitude states
# - randomized dwell times around 0.40 s
# - 80-ms smoothing of state transitions
# - five independent realizations/source
# - representative injected SNR = 10 dB
# - screen 18/24/30 dB peak-to-peak log-amplitude depths
#
# qadd_pause_level_iqr_db is Q75-Q25 of eligible 30-ms pause-frame AC levels.
# Repeated high/low interference states directly instantiate temporal
# heterogeneity without conditioning the perturbation on speech/pause labels.
#
# Selection rule remains fixed before the v1.3 signal response:
# choose the LOWEST depth with:
# - >=12 finite pilot sources
# - >=70% source-averaged responses in expected direction
# - median target-Q increase >=0.25 natural IQR at screen SNR
#
# If none passes, stop again before any clinical unblinding.

V13_AM_SCREEN_SNR_DB = 10.0
V13_AM_DEPTH_DB_PP = [18.0, 24.0, 30.0]
V13_AM_STATE_SEC = 0.40
V13_AM_STATE_JITTER_FRACTION = 0.35
V13_AM_TRANSITION_SEC = 0.08
V13_AM_NOMINAL_MODULATION_HZ = 1.0 / (2.0 * V13_AM_STATE_SEC)
V13_AM_REALIZATIONS = 5
V13_AM_SCREEN_MIN_FINITE = 12
V13_AM_SCREEN_MIN_DIRECTION = 0.70
V13_AM_SCREEN_MIN_MEDIAN_IQR = 0.25

V13_AM_SNR_GRID_DB = [35.0, 30.0, 25.0, 20.0, 15.0, 10.0, 5.0, 0.0]

# Dynamic QGAIN showed correct monotonic direction but insufficient range.
# Keep the transformation exactly unchanged and extend only amplitude.
V13_DYNAMIC_GAIN_EXTENSION_DB = [
    10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 24.0
]

# QCHAN showed weak rolloff-95 response at 3.5 kHz.
# Keep exact Paper 1 low-pass implementation + same 3 filter orders;
# extend only the cutoff range downward.
V13_QCHAN_CUTOFF_EXTENSION_HZ = [
    3250.0, 3000.0, 2750.0, 2500.0,
    2250.0, 2000.0, 1750.0, 1500.0,
]
V13_QCHAN_FILTER_ORDERS = {1: 4, 2: 8, 3: 12}

V13_SELECTION_TARGETS = {"low": 0.5, "medium": 1.0, "high": 2.0}
V13_MIN_SELECTED_FINITE = 12
V13_MIN_DIRECTION = 0.70
V13_MIN_HIGH_IQR = 1.00

v13_contract = {
    "created_utc": V13_CREATED_UTC,
    "engine": V13_ENGINE,
    "paper1_commit": PAPER1_COMMIT,
    "clinical_outcomes_loaded": False,
    "clinical_predictions_loaded": False,
    "v12_failed_am_screen_path": (
        str(V12_FAILED_SCREEN) if V12_FAILED_SCREEN.exists() else None
    ),
    "v12_failed_am_screen_sha256": (
        v13_sha256_file(V12_FAILED_SCREEN)
        if V12_FAILED_SCREEN.exists()
        else None
    ),
    "v11_passing_transforms_locked": LOCKED_TRANSFORMS,
    "revised_transforms_only": [
        "amplitude_modulated_colored_broadband",
        "smooth_time_varying_gain",
        "upper_band_restriction",
    ],
    "am_qadd_revision": {
        "reason": (
            "v1.1 source-level median pause-level-IQR change was negative "
            "throughout the entire SNR grid; revise modulation structure."
        ),
        "screen_snr_db": V13_AM_SCREEN_SNR_DB,
        "modulation_depth_db_peak_to_peak_grid": V13_AM_DEPTH_DB_PP,
        "envelope_state_sec": V13_AM_STATE_SEC,
        "state_jitter_fraction": V13_AM_STATE_JITTER_FRACTION,
        "transition_sec": V13_AM_TRANSITION_SEC,
        "nominal_modulation_hz": V13_AM_NOMINAL_MODULATION_HZ,
        "realizations_per_source_setting": V13_AM_REALIZATIONS,
        "structure_selection": (
            "lowest depth meeting finite>=12, direction>=0.70, "
            "median target change>=0.25 natural IQR"
        ),
        "final_snr_grid_db": V13_AM_SNR_GRID_DB,
    },
    "dynamic_qgain_revision": {
        "reason": (
            "v1.1 response was ordered and directionally correct but high dose "
            "reached only ~0.30 natural IQR."
        ),
        "transformation_changed": False,
        "new_amplitude_db_grid": V13_DYNAMIC_GAIN_EXTENSION_DB,
    },
    "qchan_revision": {
        "reason": (
            "v1.1 upper-band restriction was weak on frozen rolloff95-deficit "
            "target; extend cutoff severity without changing target/reference."
        ),
        "transformation_changed": False,
        "new_cutoff_hz_grid": V13_QCHAN_CUTOFF_EXTENSION_HZ,
        "filter_orders": V13_QCHAN_FILTER_ORDERS,
    },
    "acceptance_gates_unchanged": {
        "minimum_finite_sources": V13_MIN_SELECTED_FINITE,
        "minimum_expected_direction_fraction": V13_MIN_DIRECTION,
        "minimum_high_dose_median_target_iqr": V13_MIN_HIGH_IQR,
        "strict_median_monotonicity": True,
    },
    "final_manifest_auto_write": False,
}

v13_atomic_json(
    v13_contract,
    V13_OUT / "goal3_stageB_v1_2_revision_contract.json",
)

print("=" * 78)
print("GOAL 3 STAGE B v1.3 TARGETED REVISION")
print("=" * 78)
print("Locked from v1.1:", ", ".join(LOCKED_TRANSFORMS))
print("Revising only: AM-QADD structure, dynamic-QGAIN range, QCHAN range")
print("Clinical outcomes/predictions loaded: FALSE")
print("Final perturbation manifest will NOT be written automatically.")
print()

# ---------------------------------------------------------------------
# 3. Revised AM-QADD structure
# ---------------------------------------------------------------------

def v13_am_intermittent_interference(
    native,
    source_sr,
    strict_speech_rms,
    *,
    snr_db,
    modulation_depth_db_pp,
    seed,
):
    """Intermittent amplitude-modulated colored broadband interference."""
    values = np.asarray(native, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]

    if not np.isfinite(strict_speech_rms) or strict_speech_rms <= 0:
        raise RuntimeError("Strict-speech RMS unavailable for intervention SNR.")

    n_samples = values.shape[0]

    noise = colored_broadband_noise(
        n_samples,
        int(source_sr),
        int(seed),
    )

    rng = np.random.default_rng(int(seed) ^ 0x6D2B79F5)

    half_depth = float(modulation_depth_db_pp) / 2.0
    if half_depth <= 0:
        raise ValueError("AM-QADD modulation depth must be positive.")

    envelope_db = np.empty(n_samples, dtype=np.float64)

    state = 1.0 if rng.random() >= 0.5 else -1.0
    cursor = 0

    min_dwell = V13_AM_STATE_SEC * (1.0 - V13_AM_STATE_JITTER_FRACTION)
    max_dwell = V13_AM_STATE_SEC * (1.0 + V13_AM_STATE_JITTER_FRACTION)

    while cursor < n_samples:
        dwell_sec = float(rng.uniform(min_dwell, max_dwell))
        dwell_samples = max(1, int(round(dwell_sec * source_sr)))
        stop = min(n_samples, cursor + dwell_samples)
        envelope_db[cursor:stop] = state * half_depth
        cursor = stop
        state *= -1.0

    smooth_samples = max(
        3,
        int(round(V13_AM_TRANSITION_SEC * source_sr)),
    )
    if smooth_samples % 2 == 0:
        smooth_samples += 1

    kernel = np.hanning(smooth_samples)
    if not np.isfinite(kernel).all() or kernel.sum() <= 0:
        raise RuntimeError("Invalid AM-QADD smoothing kernel.")
    kernel = kernel / kernel.sum()

    envelope_db = np.convolve(
        envelope_db,
        kernel,
        mode="same",
    )

    envelope_db -= float(np.mean(envelope_db, dtype=np.float64))

    amplitude = np.power(10.0, envelope_db / 20.0)
    noise = noise * amplitude

    noise_rms = float(
        np.sqrt(np.mean(noise * noise, dtype=np.float64))
    )
    if not np.isfinite(noise_rms) or noise_rms <= 0:
        raise RuntimeError("AM-QADD intermittent envelope produced invalid RMS.")

    noise = noise / noise_rms

    target_noise_rms = (
        float(strict_speech_rms)
        / (10.0 ** (float(snr_db) / 20.0))
    )

    noise = noise * target_noise_rms
    transformed = values + noise[:, None]

    if not np.isfinite(transformed).all():
        raise RuntimeError("AM-QADD intermittent transformation is non-finite.")

    return transformed.astype(np.float32)


# ---------------------------------------------------------------------
# 4. v1.3 checkpointed execution engine
# ---------------------------------------------------------------------

def v13_candidate_id(candidate: dict) -> str:
    return stable_hash(
        V13_ENGINE,
        candidate["family"],
        candidate["transform"],
        candidate["candidate_value"],
        candidate.get("candidate_unit", ""),
        candidate.get("exemplar", 1),
        candidate.get("modulation_depth_db_pp", ""),
        candidate.get("modulation_frequency_hz", ""),
        candidate.get("envelope_state_sec", ""),
        candidate.get("state_jitter_fraction", ""),
        candidate.get("transition_sec", ""),
        candidate.get("filter_order", ""),
    )[:20]

def v13_checkpoint_path(source_id, candidate_id):
    folder = V13_CHECKPOINTS / re.sub(
        r"[^A-Za-z0-9_.-]+", "_", str(source_id)
    )
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{candidate_id}.json"

def v13_checkpoint_signature(source_row, candidate):
    return stable_hash(
        V13_ENGINE,
        PAPER1_COMMIT,
        source_row["logical_recording_id"],
        source_row["observed_sha256"],
        json.dumps(json_safe(candidate), sort_keys=True),
    )

def v13_read_checkpoint(path, signature):
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("signature") != signature:
        return None
    if payload.get("engine") != V13_ENGINE:
        return None
    return payload.get("row")

def v13_write_checkpoint(path, signature, row):
    v13_atomic_json(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "engine": V13_ENGINE,
            "signature": signature,
            "row": row,
        },
        path,
    )

def v13_transform(source, candidate, source_row):
    transform = candidate["transform"]
    value = float(candidate["candidate_value"])

    if transform == "amplitude_modulated_colored_broadband_v3_intermittent":
        seed = stable_seed(
            "goal3_v13_am",
            source_row["logical_recording_id"],
            value,
            candidate["modulation_depth_db_pp"],
            int(candidate["exemplar"]),
        )
        waveform = v13_am_intermittent_interference(
            source["native"],
            source["source_sr"],
            source["baseline_strict_speech_rms_native"],
            snr_db=value,
            modulation_depth_db_pp=float(candidate["modulation_depth_db_pp"]),
            seed=seed,
        )
        meta = {
            "random_seed": int(seed),
            "modulation_depth_db_pp": float(candidate["modulation_depth_db_pp"]),
            "modulation_frequency_hz": float(candidate["modulation_frequency_hz"]),
            "envelope_state_sec": float(candidate.get("envelope_state_sec", np.nan)),
            "state_jitter_fraction": float(candidate.get("state_jitter_fraction", np.nan)),
            "transition_sec": float(candidate.get("transition_sec", np.nan)),
            "filter_order": np.nan,
        }

    elif transform == "smooth_time_varying_gain":
        waveform = apply_dynamic_attenuation(
            source["native"],
            source["source_sr"],
            value,
        )
        meta = {
            "random_seed": np.nan,
            "modulation_depth_db_pp": np.nan,
            "modulation_frequency_hz": float(DYNAMIC_GAIN_HZ),
            "filter_order": np.nan,
        }

    elif transform == "upper_band_restriction":
        order = int(candidate["filter_order"])
        waveform = apply_qchan_lowpass(
            source["native"],
            source["source_sr"],
            value,
            order,
        )
        meta = {
            "random_seed": np.nan,
            "modulation_depth_db_pp": np.nan,
            "modulation_frequency_hz": np.nan,
            "filter_order": order,
        }

    else:
        raise ValueError(f"Unexpected v1.3 transform: {transform}")

    if waveform.shape != np.asarray(source["native"]).shape:
        raise RuntimeError("v1.3 transformation changed waveform shape.")
    if not np.isfinite(waveform).all():
        raise RuntimeError("v1.3 transformation produced non-finite waveform.")

    return waveform, meta

def v13_run_grid(grid: pd.DataFrame, *, label: str) -> pd.DataFrame:
    if grid.empty:
        raise ValueError("v1.3 candidate grid is empty.")

    rows = []
    start = time.perf_counter()

    print(f"\n--- {label} ---")
    print("Pilot sources:", len(pilot_run))
    print("Candidate rows/source:", len(grid))

    for source_number, source_row in enumerate(
        pilot_run.sort_values("pilot_rank").to_dict("records"),
        start=1,
    ):
        source_id = str(source_row["logical_recording_id"])
        fold = int(source_row["outer_fold"])

        source = decode_source(source_row)
        qchan_reference = fixed_qchan_references[fold]

        for candidate in grid.to_dict("records"):
            candidate = dict(candidate)
            candidate["candidate_id"] = v13_candidate_id(candidate)

            variant_id = stable_hash(
                V13_ENGINE,
                source_id,
                candidate["candidate_id"],
            )[:20]

            cp = v13_checkpoint_path(source_id, candidate["candidate_id"])
            signature = v13_checkpoint_signature(source_row, candidate)
            cached = v13_read_checkpoint(cp, signature)

            if cached is not None:
                rows.append(cached)
                continue

            row_start = time.perf_counter()

            try:
                transformed, transform_meta = v13_transform(
                    source,
                    candidate,
                    source_row,
                )

                q_values, _ = extract_all_q(
                    transformed,
                    source["source_sr"],
                    logical_recording_id=f"{source_id}__goal3_v13__{variant_id}",
                    outer_fold=fold,
                    qchan_reference=qchan_reference,
                    qdist_requested=False,
                    probe=source["probe"],
                    source_path=str(source["media_path"]),
                    source_sha256=source["source_sha256"],
                    variant_id=variant_id,
                )

                row = {
                    "participant_id": str(source_row["participant_id"]),
                    "logical_recording_id": source_id,
                    "outer_fold": fold,
                    "pilot_rank": int(source_row["pilot_rank"]),
                    "variant_id": variant_id,
                    "candidate_id": candidate["candidate_id"],
                    "family": candidate["family"],
                    "transform": candidate["transform"],
                    "candidate_value": float(candidate["candidate_value"]),
                    "candidate_unit": candidate["candidate_unit"],
                    "physical_strength_order": float(
                        candidate["physical_strength_order"]
                    ),
                    "exemplar": int(candidate["exemplar"]),
                    "source_sha256": source["source_sha256"],
                    "source_sample_rate_hz": int(source["source_sr"]),
                    "execution_status": "PASS",
                    "execution_error": "",
                    "elapsed_sec": time.perf_counter() - row_start,
                    **transform_meta,
                    **q_values,
                }

            except Exception as exc:
                row = {
                    "participant_id": str(source_row["participant_id"]),
                    "logical_recording_id": source_id,
                    "outer_fold": fold,
                    "pilot_rank": int(source_row["pilot_rank"]),
                    "variant_id": variant_id,
                    "candidate_id": candidate["candidate_id"],
                    "family": candidate["family"],
                    "transform": candidate["transform"],
                    "candidate_value": float(candidate["candidate_value"]),
                    "candidate_unit": candidate["candidate_unit"],
                    "physical_strength_order": float(
                        candidate["physical_strength_order"]
                    ),
                    "exemplar": int(candidate["exemplar"]),
                    "source_sha256": source["source_sha256"],
                    "execution_status": "ERROR",
                    "execution_error": f"{type(exc).__name__}: {exc}",
                    "elapsed_sec": time.perf_counter() - row_start,
                    **{feature: np.nan for feature in CORE_Q},
                }

            v13_write_checkpoint(cp, signature, row)
            rows.append(row)

        if source_number % 4 == 0 or source_number == len(pilot_run):
            elapsed_min = (time.perf_counter() - start) / 60.0
            print(
                f"{label}: source {source_number}/{len(pilot_run)} complete | "
                f"elapsed {elapsed_min:.1f} min"
            )

    out = pd.DataFrame(rows)

    if out["execution_status"].ne("PASS").any():
        errors = out.loc[
            out["execution_status"].ne("PASS"),
            [
                "logical_recording_id",
                "transform",
                "candidate_value",
                "exemplar",
                "execution_error",
            ],
        ]
        v13_atomic_csv(
            errors,
            V13_AUDIT / f"{label}_errors.csv",
        )
        raise RuntimeError(
            f"{label}: at least one new signal-only candidate failed. "
            "See the v1.3 audit errors table."
        )

    return out

# ---------------------------------------------------------------------
# 5. Baseline pairing + source-level target summary
# ---------------------------------------------------------------------

v11_baseline = v11_response.loc[
    v11_response["transform"].eq("baseline")
].copy()

if len(v11_baseline) != 24:
    raise RuntimeError("Expected 24 v1.1 controlled baselines.")

def v13_attach_baseline_delta(response: pd.DataFrame) -> pd.DataFrame:
    baseline_cols = ["logical_recording_id", *CORE_Q]
    lookup = v11_baseline[baseline_cols].copy().rename(
        columns={
            feature: f"baseline__{feature}"
            for feature in CORE_Q
        }
    )

    out = response.merge(
        lookup,
        on="logical_recording_id",
        how="left",
        validate="many_to_one",
    )

    for feature in CORE_Q:
        out[f"delta__{feature}"] = (
            pd.to_numeric(out[feature], errors="coerce")
            - pd.to_numeric(out[f"baseline__{feature}"], errors="coerce")
        )

    return out

def v13_target_summary(
    response_delta: pd.DataFrame,
    *,
    target_q: str,
    direction: float,
    extra_group_cols=None,
):
    extra_group_cols = list(extra_group_cols or [])

    scale = float(iqr_lookup[target_q])
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError(f"Invalid natural IQR for {target_q}: {scale}")

    rows = []

    group_cols = [
        "family",
        "transform",
        "candidate_value",
        "candidate_unit",
        "physical_strength_order",
        *extra_group_cols,
    ]

    for key, group in response_delta.groupby(
        group_cols,
        dropna=False,
        sort=True,
    ):
        if not isinstance(key, tuple):
            key = (key,)

        key_map = dict(zip(group_cols, key))

        source = (
            group[
                ["logical_recording_id", f"delta__{target_q}"]
            ]
            .assign(
                _delta=lambda d: pd.to_numeric(
                    d[f"delta__{target_q}"],
                    errors="coerce",
                )
            )
            .groupby("logical_recording_id", as_index=False)
            .agg(target_delta=("_delta", "mean"))
        )

        oriented = float(direction) * pd.to_numeric(
            source["target_delta"],
            errors="coerce",
        )
        oriented = oriented[np.isfinite(oriented)]

        scaled = oriented / scale

        row = {
            **key_map,
            "target_q": target_q,
            "expected_direction_multiplier": float(direction),
            "unique_sources_total": group["logical_recording_id"].nunique(),
            "finite_target_sources": len(oriented),
            "finite_target_fraction": (
                len(oriented)
                / group["logical_recording_id"].nunique()
            ),
            "expected_direction_fraction": (
                float((oriented > 0).mean())
                if len(oriented)
                else np.nan
            ),
            "oriented_target_delta_raw_median": (
                float(oriented.median())
                if len(oriented)
                else np.nan
            ),
            "natural_target_iqr_raw": scale,
            "target_delta_natural_iqr_median": (
                float(scaled.median())
                if len(scaled)
                else np.nan
            ),
            "target_delta_natural_iqr_q25": (
                float(scaled.quantile(0.25))
                if len(scaled)
                else np.nan
            ),
            "target_delta_natural_iqr_q75": (
                float(scaled.quantile(0.75))
                if len(scaled)
                else np.nan
            ),
            "execution_error_fraction": float(
                (~group["execution_status"].eq("PASS")).mean()
            ),
        }
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["transform", "physical_strength_order"]
    ).reset_index(drop=True)

# ---------------------------------------------------------------------
# 6. AM-QADD structure screen
# ---------------------------------------------------------------------

am_screen_rows = []

for depth in V13_AM_DEPTH_DB_PP:
    for exemplar in range(1, V13_AM_REALIZATIONS + 1):
        am_screen_rows.append({
            "family": "QADD",
            "transform": "amplitude_modulated_colored_broadband_v3_intermittent",
            "candidate_value": V13_AM_SCREEN_SNR_DB,
            "candidate_unit": "dB injected SNR",
            "physical_strength_order": -V13_AM_SCREEN_SNR_DB,
            "exemplar": exemplar,
            "modulation_depth_db_pp": depth,
            "modulation_frequency_hz": V13_AM_NOMINAL_MODULATION_HZ,
            "envelope_state_sec": V13_AM_STATE_SEC,
            "state_jitter_fraction": V13_AM_STATE_JITTER_FRACTION,
            "transition_sec": V13_AM_TRANSITION_SEC,
            "filter_order": np.nan,
        })

am_screen_grid = pd.DataFrame(am_screen_rows)

v13_atomic_csv(
    am_screen_grid,
    V13_TABLES / "goal3_v13_am_structure_screen_grid.csv",
)

am_screen_response = v13_run_grid(
    am_screen_grid,
    label="am_structure_screen",
)

am_screen_delta = v13_attach_baseline_delta(am_screen_response)

am_screen_summary = v13_target_summary(
    am_screen_delta,
    target_q="qadd_pause_level_iqr_db",
    direction=+1.0,
    extra_group_cols=["modulation_depth_db_pp"],
)

am_screen_summary["screen_pass"] = (
    am_screen_summary["finite_target_sources"].ge(V13_AM_SCREEN_MIN_FINITE)
    & am_screen_summary["expected_direction_fraction"].ge(
        V13_AM_SCREEN_MIN_DIRECTION
    )
    & am_screen_summary["target_delta_natural_iqr_median"].ge(
        V13_AM_SCREEN_MIN_MEDIAN_IQR
    )
)

v13_atomic_csv(
    am_screen_summary,
    V13_TABLES / "goal3_v13_am_structure_screen_summary.csv",
)

print("\nAM-QADD REVISED STRUCTURE SCREEN")
display(am_screen_summary)

eligible_depths = (
    am_screen_summary.loc[
        am_screen_summary["screen_pass"],
        "modulation_depth_db_pp",
    ]
    .dropna()
    .astype(float)
    .sort_values()
    .unique()
)

if len(eligible_depths) == 0:
    v13_atomic_json(
        {
            "status": "NEEDS_AM_STRUCTURE_REVISION",
            "clinical_outcomes_inspected": False,
            "clinical_predictions_inspected": False,
            "screen_summary": am_screen_summary.to_dict("records"),
        },
        V13_OUT / "NEEDS_REVISION_AM_STRUCTURE.json",
    )
    raise RuntimeError(
        "No revised AM-QADD modulation depth passed the outcome-blind structure "
        "screen. Stop here; do not inspect clinical outputs."
    )

V13_AM_SELECTED_DEPTH_DB_PP = float(eligible_depths[0])

print(
    "\nAM-QADD v1.3 intermittent structure selected outcome-blind:",
    f"{V13_AM_SELECTED_DEPTH_DB_PP:g} dB peak-to-peak",
)
print(
    "Selection rule: lowest depth meeting all frozen screen gates."
)

# ---------------------------------------------------------------------
# 7. Run ONLY the necessary v1.3 extension grid
# ---------------------------------------------------------------------

extension_rows = []

# AM-QADD full SNR grid using selected structure.
for snr_db in V13_AM_SNR_GRID_DB:
    for exemplar in range(1, V13_AM_REALIZATIONS + 1):
        extension_rows.append({
            "family": "QADD",
            "transform": "amplitude_modulated_colored_broadband_v3_intermittent",
            "candidate_value": snr_db,
            "candidate_unit": "dB injected SNR",
            "physical_strength_order": -snr_db,
            "exemplar": exemplar,
            "modulation_depth_db_pp": V13_AM_SELECTED_DEPTH_DB_PP,
            "modulation_frequency_hz": V13_AM_NOMINAL_MODULATION_HZ,
            "envelope_state_sec": V13_AM_STATE_SEC,
            "state_jitter_fraction": V13_AM_STATE_JITTER_FRACTION,
            "transition_sec": V13_AM_TRANSITION_SEC,
            "filter_order": np.nan,
        })

# Dynamic QGAIN extension only.
for amplitude_db in V13_DYNAMIC_GAIN_EXTENSION_DB:
    extension_rows.append({
        "family": "QGAIN",
        "transform": "smooth_time_varying_gain",
        "candidate_value": amplitude_db,
        "candidate_unit": "dB modulation amplitude",
        "physical_strength_order": amplitude_db,
        "exemplar": 1,
        "modulation_depth_db_pp": np.nan,
        "modulation_frequency_hz": float(DYNAMIC_GAIN_HZ),
        "filter_order": np.nan,
    })

# QCHAN cutoff extension only, same three Paper-1 helper orders.
for cutoff_hz in V13_QCHAN_CUTOFF_EXTENSION_HZ:
    for exemplar, order in V13_QCHAN_FILTER_ORDERS.items():
        extension_rows.append({
            "family": "QCHAN",
            "transform": "upper_band_restriction",
            "candidate_value": cutoff_hz,
            "candidate_unit": "low-pass cutoff Hz",
            "physical_strength_order": -cutoff_hz,
            "exemplar": exemplar,
            "modulation_depth_db_pp": np.nan,
            "modulation_frequency_hz": np.nan,
            "filter_order": order,
        })

extension_grid = pd.DataFrame(extension_rows)

v13_atomic_csv(
    extension_grid,
    V13_TABLES / "goal3_v13_targeted_extension_grid.csv",
)

extension_response = v13_run_grid(
    extension_grid,
    label="targeted_extension",
)

# Combine screen + extension checkpoints, deduplicating the selected-depth
# SNR=10 AM candidates that occur in both stages.
v13_response = (
    pd.concat(
        [am_screen_response, extension_response],
        ignore_index=True,
        sort=False,
    )
    .drop_duplicates(
        ["logical_recording_id", "candidate_id"],
        keep="last",
    )
    .reset_index(drop=True)
)

v13_atomic_csv(
    v13_response,
    V13_TABLES / "goal3_v13_signal_only_candidate_response.csv",
)

v13_delta = v13_attach_baseline_delta(v13_response)

v13_atomic_csv(
    v13_delta,
    V13_TABLES / "goal3_v13_signal_only_candidate_delta.csv",
)

print("\nTARGETED v1.3 SIGNAL EXECUTION: COMPLETE")
print("Unique new/revised response rows:", len(v13_response))

# ---------------------------------------------------------------------
# 8. Build revised candidate summaries
# ---------------------------------------------------------------------

am_final_delta = v13_delta.loc[
    v13_delta["transform"].eq(
        "amplitude_modulated_colored_broadband_v3_intermittent"
    )
    & np.isclose(
        pd.to_numeric(
            v13_delta["modulation_depth_db_pp"],
            errors="coerce",
        ),
        V13_AM_SELECTED_DEPTH_DB_PP,
    )
].copy()

am_final_summary = v13_target_summary(
    am_final_delta,
    target_q="qadd_pause_level_iqr_db",
    direction=+1.0,
)

dyn_new_delta = v13_delta.loc[
    v13_delta["transform"].eq("smooth_time_varying_gain")
].copy()

dyn_new_summary = v13_target_summary(
    dyn_new_delta,
    target_q="qgain_within_segment_iqr_db",
    direction=+1.0,
)

qchan_new_delta = v13_delta.loc[
    v13_delta["transform"].eq("upper_band_restriction")
].copy()

qchan_new_summary = v13_target_summary(
    qchan_new_delta,
    target_q="qchan_rolloff95_deficit_hz",
    direction=+1.0,
)

# Old valid candidates remain eligible for dynamic QGAIN and QCHAN.
dyn_old_summary = v11_summary.loc[
    v11_summary["transform"].eq("smooth_time_varying_gain")
].copy()

qchan_old_summary = v11_summary.loc[
    v11_summary["transform"].eq("upper_band_restriction")
].copy()

# Harmonize to the subset needed by the unchanged selection rule.
_summary_cols = [
    "family",
    "transform",
    "target_q",
    "expected_direction_multiplier",
    "candidate_value",
    "candidate_unit",
    "physical_strength_order",
    "unique_sources_total",
    "finite_target_sources",
    "finite_target_fraction",
    "expected_direction_fraction",
    "oriented_target_delta_raw_median",
    "natural_target_iqr_raw",
    "target_delta_natural_iqr_median",
    "target_delta_natural_iqr_q25",
    "target_delta_natural_iqr_q75",
    "execution_error_fraction",
]

for _frame in [dyn_old_summary, qchan_old_summary]:
    for _col in _summary_cols:
        if _col not in _frame.columns:
            _frame[_col] = np.nan

dyn_combined_summary = (
    pd.concat(
        [dyn_old_summary[_summary_cols], dyn_new_summary[_summary_cols]],
        ignore_index=True,
    )
    .sort_values("physical_strength_order")
    .drop_duplicates("candidate_value", keep="last")
    .reset_index(drop=True)
)

qchan_combined_summary = (
    pd.concat(
        [qchan_old_summary[_summary_cols], qchan_new_summary[_summary_cols]],
        ignore_index=True,
    )
    .sort_values("physical_strength_order")
    .drop_duplicates("candidate_value", keep="last")
    .reset_index(drop=True)
)

am_final_summary = am_final_summary[_summary_cols].copy()

revision_candidate_summary = pd.concat(
    [
        am_final_summary,
        dyn_combined_summary,
        qchan_combined_summary,
    ],
    ignore_index=True,
)

v13_atomic_csv(
    revision_candidate_summary,
    V13_TABLES / "goal3_v13_revised_candidate_summary.csv",
)

# ---------------------------------------------------------------------
# 9. Unchanged 0.5/1/2-IQR dose selection
# ---------------------------------------------------------------------

def v13_select_three(summary):
    work = summary.loc[
        np.isfinite(
            pd.to_numeric(
                summary["target_delta_natural_iqr_median"],
                errors="coerce",
            )
        )
    ].copy()

    work = work.sort_values(
        "physical_strength_order"
    ).reset_index(drop=True)

    if len(work) < 3:
        return None, "fewer_than_three_valid_settings"

    targets = np.array([0.5, 1.0, 2.0], dtype=float)
    best = None

    for indices in itertools.combinations(range(len(work)), 3):
        chosen = work.iloc[list(indices)].copy()
        response = chosen[
            "target_delta_natural_iqr_median"
        ].to_numpy(float)

        if not np.isfinite(response).all():
            continue
        if not np.all(response > 0):
            continue
        if not np.all(np.diff(response) > 0):
            continue

        score = float(np.sum((response - targets) ** 2))

        if best is None or score < best[0]:
            best = (score, chosen)

    if best is None:
        return None, "no_strictly_ordered_positive_three_setting_solution"

    selected = best[1].copy()
    selected["dose_label"] = ["low", "medium", "high"]
    selected["dose_code"] = [1, 2, 3]
    selected["target_natural_iqr"] = [0.5, 1.0, 2.0]
    selected["selection_objective_sse"] = best[0]
    return selected, "pass"

revision_selected_parts = []
revision_selection_status = []

for transform in [
    "amplitude_modulated_colored_broadband_v3_intermittent",
    "smooth_time_varying_gain",
    "upper_band_restriction",
]:
    local = revision_candidate_summary.loc[
        revision_candidate_summary["transform"].eq(transform)
    ].copy()

    selected, status = v13_select_three(local)

    revision_selection_status.append({
        "transform": transform,
        "selection_status": status,
    })

    if selected is not None:
        revision_selected_parts.append(selected)

revision_selection_status = pd.DataFrame(revision_selection_status)

revision_selected = (
    pd.concat(revision_selected_parts, ignore_index=True)
    if revision_selected_parts
    else pd.DataFrame()
)

v13_atomic_csv(
    revision_selection_status,
    V13_TABLES / "goal3_v13_selection_status.csv",
)

v13_atomic_csv(
    revision_selected,
    V13_TABLES / "goal3_v13_selected_doses.csv",
    allow_empty=True,
)

# ---------------------------------------------------------------------
# 10. EXACT SAME acceptance gates for revised transforms
# ---------------------------------------------------------------------

gate_rows = []

for transform in [
    "amplitude_modulated_colored_broadband_v3_intermittent",
    "smooth_time_varying_gain",
    "upper_band_restriction",
]:
    selected = revision_selected.loc[
        revision_selected["transform"].eq(transform)
    ].sort_values("dose_code")

    gate_rows.append({
        "transform": transform,
        "gate": "three_selected_doses",
        "passed": len(selected) == 3,
        "observed": len(selected),
        "required": 3,
    })

    if len(selected) != 3:
        continue

    response = selected[
        "target_delta_natural_iqr_median"
    ].to_numpy(float)

    finite_sources = selected[
        "finite_target_sources"
    ].to_numpy(float)

    direction_fraction = selected[
        "expected_direction_fraction"
    ].to_numpy(float)

    high_response = float(
        selected.loc[
            selected["dose_label"].eq("high"),
            "target_delta_natural_iqr_median",
        ].iloc[0]
    )

    gate_rows.extend([
        {
            "transform": transform,
            "gate": "strict_median_monotonicity",
            "passed": bool(
                np.isfinite(response).all()
                and np.all(np.diff(response) > 0)
            ),
            "observed": response.tolist(),
            "required": "strictly increasing low < medium < high",
        },
        {
            "transform": transform,
            "gate": "minimum_finite_pilot_sources",
            "passed": bool(
                np.all(finite_sources >= V13_MIN_SELECTED_FINITE)
            ),
            "observed": finite_sources.tolist(),
            "required": V13_MIN_SELECTED_FINITE,
        },
        {
            "transform": transform,
            "gate": "expected_direction_fraction",
            "passed": bool(
                np.all(direction_fraction >= V13_MIN_DIRECTION)
            ),
            "observed": direction_fraction.tolist(),
            "required": V13_MIN_DIRECTION,
        },
        {
            "transform": transform,
            "gate": "high_dose_natural_span",
            "passed": bool(
                np.isfinite(high_response)
                and high_response >= V13_MIN_HIGH_IQR
            ),
            "observed": high_response,
            "required": f">= {V13_MIN_HIGH_IQR:.2f} natural IQR",
        },
    ])

revision_acceptance = pd.DataFrame(gate_rows)

v13_atomic_csv(
    revision_acceptance,
    V13_TABLES / "goal3_v13_acceptance_gates.csv",
)

V13_REVISION_PASS = bool(
    len(revision_acceptance)
    and revision_acceptance["passed"].astype(bool).all()
)

# ---------------------------------------------------------------------
# 11. Off-target Q characterization for revised selected doses
# ---------------------------------------------------------------------

revision_offtarget_rows = []

if len(revision_selected):
    # Build response source for selected revised physical values.
    for selected_row in revision_selected.itertuples(index=False):
        transform = str(selected_row.transform)
        candidate_value = float(selected_row.candidate_value)
        dose_label = str(selected_row.dose_label)
        dose_code = int(selected_row.dose_code)
        target_q = str(selected_row.target_q)

        if transform == "amplitude_modulated_colored_broadband_v3_intermittent":
            local = am_final_delta.loc[
                np.isclose(
                    pd.to_numeric(
                        am_final_delta["candidate_value"],
                        errors="coerce",
                    ),
                    candidate_value,
                )
            ].copy()
        elif transform == "smooth_time_varying_gain":
            # Selected dose may be old or new. Use v1.1 delta for <=8 dB,
            # v1.3 delta for extension settings.
            if candidate_value <= 8.0:
                local = v11_delta.loc[
                    v11_delta["transform"].eq(transform)
                    & np.isclose(
                        pd.to_numeric(
                            v11_delta["candidate_value"],
                            errors="coerce",
                        ),
                        candidate_value,
                    )
                ].copy()
            else:
                local = dyn_new_delta.loc[
                    np.isclose(
                        pd.to_numeric(
                            dyn_new_delta["candidate_value"],
                            errors="coerce",
                        ),
                        candidate_value,
                    )
                ].copy()
        elif transform == "upper_band_restriction":
            if candidate_value >= 3500.0:
                local = v11_delta.loc[
                    v11_delta["transform"].eq(transform)
                    & np.isclose(
                        pd.to_numeric(
                            v11_delta["candidate_value"],
                            errors="coerce",
                        ),
                        candidate_value,
                    )
                ].copy()
            else:
                local = qchan_new_delta.loc[
                    np.isclose(
                        pd.to_numeric(
                            qchan_new_delta["candidate_value"],
                            errors="coerce",
                        ),
                        candidate_value,
                    )
                ].copy()
        else:
            raise ValueError(transform)

        for feature in CORE_Q:
            delta_col = f"delta__{feature}"
            if delta_col not in local.columns:
                continue

            source_values = (
                local[
                    ["logical_recording_id", delta_col]
                ]
                .assign(
                    _delta=lambda d: pd.to_numeric(
                        d[delta_col], errors="coerce"
                    )
                )
                .groupby("logical_recording_id", as_index=False)
                .agg(mean_delta=("_delta", "mean"))
            )

            finite = pd.to_numeric(
                source_values["mean_delta"],
                errors="coerce",
            )
            finite = finite[np.isfinite(finite)]

            scale = float(iqr_lookup.get(feature, np.nan))

            scaled = (
                np.abs(finite) / scale
                if np.isfinite(scale) and scale > 0
                else pd.Series(dtype=float)
            )

            revision_offtarget_rows.append({
                "transform": transform,
                "dose_label": dose_label,
                "dose_code": dose_code,
                "target_q": target_q,
                "q_feature": feature,
                "is_primary_target": feature == target_q,
                "natural_iqr_raw": scale,
                "finite_sources": len(finite),
                "median_abs_delta_raw": (
                    float(np.abs(finite).median())
                    if len(finite)
                    else np.nan
                ),
                "median_abs_delta_natural_iqr": (
                    float(scaled.median())
                    if len(scaled)
                    else np.nan
                ),
                "q25_abs_delta_natural_iqr": (
                    float(scaled.quantile(0.25))
                    if len(scaled)
                    else np.nan
                ),
                "q75_abs_delta_natural_iqr": (
                    float(scaled.quantile(0.75))
                    if len(scaled)
                    else np.nan
                ),
            })

revision_offtarget = pd.DataFrame(revision_offtarget_rows)

v13_atomic_csv(
    revision_offtarget,
    V13_TABLES / "goal3_v13_offtarget_q_summary.csv",
    allow_empty=True,
)

# ---------------------------------------------------------------------
# 12. Stop BEFORE final perturbation manifest; report for audit.
# ---------------------------------------------------------------------

print("\n" + "=" * 78)
print("GOAL 3 STAGE B v1.3 TARGETED REVISION RESULTS")
print("=" * 78)

print("\nAM-QADD selected modulation structure:")
print(
    f"  {V13_AM_SELECTED_DEPTH_DB_PP:g} dB peak-to-peak | "
    f"state={V13_AM_STATE_SEC:g}s | "
    f"jitter=±{100*V13_AM_STATE_JITTER_FRACTION:.0f}% | "
    f"transition={V13_AM_TRANSITION_SEC:g}s"
)

print("\nRevision selection status:")
display(revision_selection_status)

print("\nRevised low / medium / high dose selection:")
display(
    revision_selected[
        [
            "family",
            "transform",
            "candidate_value",
            "candidate_unit",
            "dose_label",
            "target_q",
            "target_delta_natural_iqr_median",
            "finite_target_sources",
            "expected_direction_fraction",
        ]
    ]
    if len(revision_selected)
    else revision_selected
)

print("\nRevision acceptance gates:")
display(revision_acceptance)

print("\nHigh-dose off-target Q characterization:")
if len(revision_offtarget):
    display(
        revision_offtarget.loc[
            revision_offtarget["dose_label"].eq("high")
        ].sort_values(
            ["transform", "median_abs_delta_natural_iqr"],
            ascending=[True, False],
        )
    )
else:
    print("No off-target table available.")

if V13_REVISION_PASS:
    ready_payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "READY_FOR_FINAL_SIGNAL_ONLY_SEAL_REVIEW",
        "engine": V13_ENGINE,
        "paper1_commit": PAPER1_COMMIT,
        "clinical_outcomes_inspected": False,
        "clinical_predictions_inspected": False,
        "v11_passing_transforms_locked": LOCKED_TRANSFORMS,
        "am_qadd_selected_modulation_depth_db_pp": (
            V13_AM_SELECTED_DEPTH_DB_PP
        ),
        "revised_selected_doses_sha256": v13_sha256_file(
            V13_TABLES / "goal3_v13_selected_doses.csv"
        ),
        "revision_acceptance_sha256": v13_sha256_file(
            V13_TABLES / "goal3_v13_acceptance_gates.csv"
        ),
        "offtarget_summary_sha256": v13_sha256_file(
            V13_TABLES / "goal3_v13_offtarget_q_summary.csv"
        ),
        "final_perturbation_manifest_written": False,
        "next_step": (
            "Human/scientific audit of signal-only revision tables; "
            "then write one combined immutable perturbation manifest."
        ),
    }

    v13_atomic_json(
        ready_payload,
        V13_OUT / "READY_FOR_SEAL.json",
    )

    print("\nGOAL 3 STAGE B v1.3 TARGETED REVISION: PASS")
    print("All revised signal-only calibration gates passed.")
    print("Clinical outcomes/predictions inspected: FALSE")
    print("Final perturbation manifest written: FALSE")
    print("Status: READY FOR FINAL SIGNAL-ONLY SEAL REVIEW")

else:
    failed = revision_acceptance.loc[
        ~revision_acceptance["passed"].astype(bool)
    ].to_dict("records")

    v13_atomic_json(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "NEEDS_FURTHER_SIGNAL_ONLY_REVISION",
            "engine": V13_ENGINE,
            "clinical_outcomes_inspected": False,
            "clinical_predictions_inspected": False,
            "failed_gates": failed,
            "final_perturbation_manifest_written": False,
        },
        V13_OUT / "NEEDS_FURTHER_REVISION.json",
    )

    print("\nGOAL 3 STAGE B v1.3: FURTHER SIGNAL-ONLY REVISION REQUIRED")
    print("No final perturbation manifest was written.")
    print("Clinical outcomes/predictions inspected: FALSE")

print("\nSaved under:")
print(V13_OUT)
