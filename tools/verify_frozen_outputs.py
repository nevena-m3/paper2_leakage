#!/usr/bin/env python3
"""Verify the frozen Paper-2 output archive using only the Python standard library.

Usage:
    python tools/verify_frozen_outputs.py path/to/outputs.zip

The verifier checks the archive identity, Goal-1/2/3 final seals and recorded
artifact hashes, Goal-3 Stage-E core invariants, final figure hashes, and reports
the recorded Stage-E runtime. It does not require access to participant data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import zipfile
from pathlib import PureWindowsPath

EXPECTED_ARCHIVE_SHA256 = "adc2dd5661c182bd7acec20b17ddcf5d0a90de5f495fb1f6e9721428a16ffd4e"
EXPECTED_ARCHIVE_BYTES = 176301940

G1_DONE = "outputs/goal1/goal1_complete_v1_1/final/DONE.json"
G1_FREEZE = "outputs/goal1/goal1_complete_v1_1/final/final_freeze_v1_0/GOAL1_FINAL_FREEZE.json"
G2_DONE = "outputs/goal2/goal2_completion_v1_0/final/DONE.json"
G3_DONE = "outputs/goal3/goal3_completion_v1_0/final/DONE.json"
G3_FREEZE = "outputs/goal3/goal3_completion_v1_0/final/final_freeze_v1_0/GOAL3_FINAL_FREEZE.json"
G3_FIG_MANIFEST = "outputs/goal3/FINAL_FIGURES/GOAL3_FINAL_FIGURE_MANIFEST.json"
G3_RUNTIME = "outputs/goal3/stageE_controlled_perturbation_v1_0/final/audit/goal3_stageE_runtime_environment.json"
G3_RESPONSE = "outputs/goal3/stageE_controlled_perturbation_v1_0/final/tables/goal3_controlled_response_exemplar.csv"
G3_TARGET_ORDER = "outputs/goal3/stageE_controlled_perturbation_v1_0/final/audit/goal3_target_q_order_audit.csv"
G3_BOOTSTRAP = "outputs/goal3/stageE_controlled_perturbation_v1_0/final/tables/goal3_bootstrap_ci.csv"
G3_HGB = "outputs/goal3/goal3_completion_v1_0/final/tables/goal3_hgb_goal2_reproduction_audit.csv"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def zip_path_from_manifest(path: str) -> str:
    """Convert recorded Windows relative/absolute output paths to archive paths."""
    p = path.replace("/", "\\")
    lower = p.lower()
    marker = "\\outputs\\"
    if marker in lower:
        i = lower.index(marker) + 1
        p = p[i:]
    elif lower.startswith("outputs\\"):
        pass
    else:
        # PureWindowsPath preserves Windows semantics when running on Linux/macOS.
        parts = [x for x in PureWindowsPath(p).parts if x not in ("\\", "/")]
        try:
            i = [x.lower() for x in parts].index("outputs")
            p = "\\".join(parts[i:])
        except ValueError:
            p = "\\".join(parts)
    return p.replace("\\", "/")


def require(z: zipfile.ZipFile, name: str) -> bytes:
    try:
        return z.read(name)
    except KeyError as exc:
        raise AssertionError(f"Missing required archive member: {name}") from exc


def read_json(z: zipfile.ZipFile, name: str):
    return json.loads(require(z, name).decode("utf-8"))


def read_csv(z: zipfile.ZipFile, name: str):
    text = require(z, name).decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def verify_hash_map(z: zipfile.ZipFile, mapping: dict[str, str], label: str) -> int:
    checked = 0
    for recorded_path, expected in mapping.items():
        name = zip_path_from_manifest(recorded_path)
        observed = sha256_bytes(require(z, name))
        if observed.lower() != str(expected).lower():
            raise AssertionError(
                f"{label}: SHA mismatch for {name}: expected {expected}, observed {observed}"
            )
        checked += 1
    return checked


def verify_goal1(z: zipfile.ZipFile) -> None:
    done = read_json(z, G1_DONE)
    freeze = read_json(z, G1_FREEZE)
    assert str(done.get("status", "")).upper() == "PASS"
    assert str(freeze.get("status", "")).upper() == "PASS"
    expected_freeze_sha = done["freeze_manifest_sha256"]
    observed_freeze_sha = sha256_bytes(require(z, G1_FREEZE))
    assert observed_freeze_sha == expected_freeze_sha
    n = verify_hash_map(z, freeze["frozen_artifact_hashes"], "Goal 1")
    print(f"Goal 1: PASS ({n} frozen artifact hashes verified)")


def verify_goal2(z: zipfile.ZipFile) -> None:
    done = read_json(z, G2_DONE)
    assert str(done.get("status", "")).upper() == "PASS"
    assert int(done.get("bootstrap_replicates", -1)) == 2000
    n = verify_hash_map(z, done["artifact_hashes"], "Goal 2")
    print(f"Goal 2: PASS ({n} frozen artifact hashes verified; B=2000)")


def truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "pass"}


def verify_goal3(z: zipfile.ZipFile) -> None:
    done = read_json(z, G3_DONE)
    freeze = read_json(z, G3_FREEZE)
    assert str(done.get("status", "")).upper() == "DONE"
    assert str(freeze.get("status", "")).upper() == "PASS"
    assert freeze.get("scientific_state") == "COMPLETE_AND_FROZEN"
    assert int(freeze.get("stage_e_response_rows", -1)) == 14792
    assert int(freeze.get("stage_e_expected_rows", -1)) == 14792
    assert int(freeze.get("measurement_unavailable_rows", -1)) == 0
    assert int(freeze.get("bootstrap_n", -1)) == 2000
    assert freeze.get("target_q_order_audit") == "PASS"
    assert freeze.get("target_q_direction_audit") == "PASS"
    assert freeze.get("frozen_hgb_goal2_reproduction") == "PASS"
    assert float(freeze.get("frozen_hgb_max_abs_prediction_difference", math.inf)) <= 1e-10

    # Verify every critical file recorded by the final freeze.
    critical_map = {row["file"]: row["sha256"] for row in freeze["critical_file_hashes"]}
    n_critical = verify_hash_map(z, critical_map, "Goal 3 critical")

    # Verify final figure assets and source tables from the final figure manifest.
    fig = read_json(z, G3_FIG_MANIFEST)
    asset_map = {row["final_file"]: row["sha256"] for row in fig["final_assets"]}
    source_map = {row["file"]: row["sha256"] for row in fig["source_tables"]}
    n_assets = verify_hash_map(z, asset_map, "Goal 3 figure asset")
    n_sources = verify_hash_map(z, source_map, "Goal 3 figure source")

    # Direct row-count / measurement-status check of the Stage-E response.
    response = read_csv(z, G3_RESPONSE)
    assert len(response) == 14792
    assert all(row.get("execution_status") == "PASS" for row in response)

    # Target-order audit must explicitly contain and pass both final columns.
    order = read_csv(z, G3_TARGET_ORDER)
    assert order, "Goal 3 target-Q order audit is empty"
    assert "strict_low_medium_high_order" in order[0]
    assert "all_medians_expected_direction" in order[0]
    assert all(truthy(row["strict_low_medium_high_order"]) for row in order)
    assert all(truthy(row["all_medians_expected_direction"]) for row in order)

    # Bootstrap table: every row must be PASS and B=2000 where fields exist.
    boot = read_csv(z, G3_BOOTSTRAP)
    assert boot, "Goal 3 bootstrap table is empty"
    if "status" in boot[0]:
        assert all(str(row["status"]).upper() == "PASS" for row in boot)
    if "n_bootstraps" in boot[0]:
        assert all(int(float(row["n_bootstraps"])) == 2000 for row in boot)

    # HGB reproduction: all rows PASS and numerical discrepancy <=1e-10.
    hgb = read_csv(z, G3_HGB)
    assert len(hgb) == 30, f"Expected 30 HGB reproduction rows, found {len(hgb)}"
    status_key = "status" if "status" in hgb[0] else None
    if status_key:
        assert all(str(row[status_key]).upper() == "PASS" for row in hgb)
    diff_candidates = [
        "max_abs_prediction_difference", "max_abs_diff", "abs_prediction_difference"
    ]
    diff_key = next((k for k in diff_candidates if k in hgb[0]), None)
    if diff_key:
        assert max(float(row[diff_key]) for row in hgb if row[diff_key] not in ("", None)) <= 1e-10

    runtime = read_json(z, G3_RUNTIME)["runtime"]
    print(
        "Goal 3: PASS "
        f"({n_critical} critical hashes; {n_assets} final figure assets; "
        f"{n_sources} figure-source hashes; 14,792/14,792 measurements; B=2000)"
    )
    print(
        "Goal 3 historical runtime: "
        f"Python {runtime.get('python_version')}, silero-vad {runtime.get('silero_vad')}, "
        f"onnxruntime {runtime.get('onnxruntime')}, gammatone {runtime.get('gammatone')}, "
        f"torch {runtime.get('torch')}"
    )
    if str(runtime.get("python_version", "")).startswith("3.14"):
        print(
            "Runtime provenance note: historical Stage E used Python 3.14.2, "
            "outside the pinned Paper-1 package declaration >=3.11,<3.13. "
            "See docs/RUNTIME_PROVENANCE.md."
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", help="Path to frozen outputs.zip")
    parser.add_argument(
        "--allow-different-archive-container",
        action="store_true",
        help="Verify internal manifests even if the outer ZIP byte hash differs (e.g., repacked archive).",
    )
    args = parser.parse_args()

    observed_size = __import__("os").path.getsize(args.archive)
    observed_sha = sha256_file(args.archive)
    if not args.allow_different_archive_container:
        assert observed_size == EXPECTED_ARCHIVE_BYTES, (
            f"Archive byte size mismatch: expected {EXPECTED_ARCHIVE_BYTES}, observed {observed_size}"
        )
        assert observed_sha == EXPECTED_ARCHIVE_SHA256, (
            f"Archive SHA mismatch: expected {EXPECTED_ARCHIVE_SHA256}, observed {observed_sha}"
        )
    print(f"Archive SHA-256: {observed_sha}")

    with zipfile.ZipFile(args.archive) as z:
        bad = z.testzip()
        assert bad is None, f"ZIP CRC failure in {bad}"
        verify_goal1(z)
        verify_goal2(z)
        verify_goal3(z)

    print("PAPER 2 FROZEN OUTPUT RELEASE AUDIT: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"PAPER 2 FROZEN OUTPUT RELEASE AUDIT: FAIL\n{exc}", file=sys.stderr)
        raise SystemExit(1)
