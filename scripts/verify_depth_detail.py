#!/usr/bin/env python3
"""Independent verifier for ARROW depth-detail outputs (audit scope, schema v2).

Latest user policy (2026-10-16): MoGe v3 is NOT in the main path (optional
comparison only); units are ``arrow_provisional`` with ``metric_calibrated=false``;
required stages are anchor, geometry, prep, lingbot, ndr, sceneflow, sr,
temporal, final (each an explicit subprocess with a real recorder receipt).
Scene-flow super-resolution must preserve 3D displacement magnitudes (no
resolution-ratio scaling) and mark unsupported pixels invalid: 3D-flow valid =
low support AND high-depth support AND sceneflow support/compensation.

Layout (pipeline verifier_schema.json v2): ``<output-dir>/<dataset>/`` holds
``manifest.json``, per-frame ``frames/NNNNNN.npz`` with stage-prefixed keys,
``signatures/<stage>.json``, ``attempts/<stage>_<utc>/exit.json`` receipts and
``configs/stage_configs/<stage>.json`` resolved configs.

The verifier re-derives acceptance facts from artifacts: unique source/frame
indices for all 107+80 frames, per-clip single arrow scale (per-frame min-max
normalization is rejected), the pose_to_reconstruct invariant
(inv(P0) @ camera_to_reference == camera_to_reconstruct per clip), pixel-centre
K in pixel stages and normalized K only for LingBot, valid -> finite>0 and
invalid -> NaN/<=0 depth, positive temporal denominators (0 = failure),
explicit rejection records (silent worsening/clipping = fail), real recorder
exit codes, resumable signatures, and render outputs (ffprobe/PNG/RRD).

Documented limits: absolute metric scale needs calibration/ground truth;
producer quality numbers are structure/consistency-checked only; SR boundary
improvement is delegated to the synthetic-GT audit tests (core + docs), while
this verifier independently enforces the SR storage/validity/scale contract;
RRD content is structural unless a rerun CLI is provided; real-data artifacts
are audited in a later assignment.

Usage:
  uv run --no-sync python scripts/verify_depth_detail.py \
      --dataset all --output-dir results/depth_detail_oct16 \
      [--evidence-dir temp/herdr_depth_workflow/evidence] \
      [--check-render] [--json-report out.json]
Exit code 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeGuard

import numpy as np

REQUIRED_STAGES = ("anchor", "geometry", "lingbot", "prep", "ndr", "sceneflow", "sr", "temporal", "final")
OPTIONAL_STAGES = ("moge",)
EXPECTED_COUNTS = {"grass": 107, "railway": 80}
UNITS = "arrow_provisional"
STAGE_CHAIN = "arrow->anchor->lingbot->ndr->sceneflow->sr->temporal"
MANIFEST_KEYS = (
    "nonce", "dataset", "expected_frame_count", "original_size", "units", "metric_calibrated",
    "calibration", "stages", "optional_stages", "models", "config_hash", "frames",
    "stage_receipts", "stage_signatures", "stage_configs", "temporal", "sr", "visuals",
)
FRAME_METADATA_KEYS = (
    "format_version", "stage", "dataset", "model", "params", "input_sha256", "argv",
    "status", "created_at", "units", "metric_calibrated",
)
STAGE_FILE_SETS = {
    "anchor": ("arrow_depth", "arrow_valid", "arrow_scale", "arrow_clip_index", "arrow_support_count"),
    "geometry": ("arrow_K", "arrow_K_valid", "arrow_pose_valid", "arrow_pose_frame0_valid",
                 "arrow_pose_reference_index", "arrow_camera_to_reference", "arrow_camera_to_reconstruct",
                 "arrow_camera_to_metric"),
    "prep": ("prep_edges", "prep_normal", "prep_normal_valid", "prep_normal_count"),
    "lingbot": ("lingbot_depth_in", "lingbot_depth_raw", "lingbot_valid", "lingbot_mask",
                "lingbot_K_norm", "lingbot_K_pixel", "lingbot_support_count"),
    "lingbot_fix": ("lingbot_depth", "lingbot_scale_fix", "lingbot_scale_raw"),
    "ndr": ("ndr_depth_in", "ndr_depth", "ndr_valid", "ndr_support_count"),
    "sceneflow_post": ("sceneflow_depth", "sceneflow_valid", "sceneflow_support_count",
                       "sceneflow_next_depth", "sceneflow_next_valid", "sceneflow_next_support_count",
                       "sceneflow_compensation_valid", "sceneflow_pair_index", "sceneflow_depth_source"),
    "sr": ("sr_flow", "sr_valid", "sr_support_count", "sr_flow_low", "sr_low_valid", "sr_pair_index"),
    "temporal": ("temporal_depth", "temporal_valid", "temporal_support_count", "temporal_common_support",
                 "temporal_denom", "temporal_accepted_fwd", "temporal_accepted_bwd", "temporal_residual_stats"),
    "final": ("final_depth", "final_valid", "final_support_count"),
}
PER_FRAME_STAGE_DIRS = ("anchor", "geometry", "prep", "lingbot", "lingbot_fix", "ndr", "sceneflow_post",
                        "temporal", "final")
SIGNATURE_KEYS = ("inputs", "params", "model", "argv", "code_sha256")
K_OFFDIAG_TOLERANCE = 1e-6
REL_TOLERANCE = 1e-4
POSE_TOLERANCE = 1e-3
RESIDUAL_COLUMNS = ("fwd_before_med", "fwd_after_med", "fwd_before_n", "fwd_after_n",
                    "bwd_before_med", "bwd_after_med", "bwd_before_n", "bwd_after_n")
SCENEFLOW_CONFIG_BOOL_KEYS = ("edge_aware", "use_tgv", "enforce_bound", "centered_photo",
                              "reciprocal_check", "diverse_candidates", "fast_match", "affine_photo",
                              "rigid_prior", "fused_solver", "warm_start", "keep_identity", "geometry_huber")


class Recorder:
    """Collects every check result so partial failures never abort silently."""

    def __init__(self) -> None:
        self.checks: list[dict] = []

    def check(self, check_id: str, ok: bool, detail: str = "") -> bool:
        self.checks.append({"id": check_id, "status": "pass" if ok else "fail", "detail": detail})
        return ok

    @property
    def failures(self) -> list[dict]:
        return [c for c in self.checks if c["status"] == "fail"]


def _is_number(value) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _is_size(value) -> TypeGuard[list[int]]:
    return isinstance(value, list) and len(value) == 2 and all(isinstance(v, int) and v > 0 for v in value)


def _numeric_tree(value) -> bool:
    if _is_number(value):
        return True
    if isinstance(value, dict):
        return bool(value) and all(_numeric_tree(item) for item in value.values())
    if isinstance(value, list):
        return bool(value) and all(_numeric_tree(item) for item in value)
    return False


def _finite_positive(array: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.isfinite(array) & (array > 0)


def _invalid_ok(array: np.ndarray, valid: np.ndarray) -> bool:
    """Contract: valid -> finite and > 0; invalid -> NaN or <= 0."""
    depth = np.asarray(array, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    return bool(not np.any(mask & ~_finite_positive(depth)) and not np.any(~mask & _finite_positive(depth)))


def _open_npz(path: Path) -> dict | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    except (OSError, ValueError, EOFError):
        return None


def _metadata(arrays: dict) -> dict | None:
    raw = arrays.get("metadata")
    if raw is None:
        return None
    if isinstance(raw, np.ndarray):
        raw = raw.item() if raw.shape == () else raw.reshape(-1)[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _scalar(arrays: dict, key: str) -> float | None:
    array = arrays.get(key)
    if array is None or array.size != 1:
        return None
    value = float(array.reshape(-1)[0])
    return value if math.isfinite(value) else None


def _bool_scalar(arrays: dict, key: str) -> bool | None:
    array = arrays.get(key)
    if array is None or array.size != 1 or array.dtype != np.bool_:
        return None
    return bool(array.reshape(-1)[0])


def _require_keys(recorder: Recorder, label: str, arrays: dict | None, keys: tuple[str, ...]) -> bool:
    if arrays is None:
        recorder.check("stage.fake", False, f"{label}: stage npz missing or unreadable")
        return False
    missing = [key for key in keys if key not in arrays]
    recorder.check("stage.fake" if len(missing) == len(keys) else "stage.keys", not missing,
                   f"{label}: missing keys {missing}")
    return not missing


def _check_shape(recorder: Recorder, label: str, arrays: dict, key: str, shape: tuple) -> bool:
    array = arrays.get(key)
    if array is None:
        return False
    ok = tuple(array.shape) == tuple(shape)
    recorder.check("shape.mismatch", ok, f"{label}: {key} shape {tuple(array.shape)} != {tuple(shape)}")
    return ok


def _check_dtype(recorder: Recorder, label: str, arrays: dict, key: str, dtype) -> None:
    array = arrays.get(key)
    if array is not None:
        recorder.check("dtype.mismatch", array.dtype == dtype, f"{label}: {key} dtype {array.dtype} != {dtype}")


def _check_depth_valid(recorder: Recorder, label: str, depth: np.ndarray, valid: np.ndarray) -> None:
    recorder.check("depth.valid_finites", _invalid_ok(depth, valid),
                   f"{label}: valid pixels must be finite>0 and invalid pixels NaN or <=0")


def _check_support_count(recorder: Recorder, label: str, arrays: dict, depth_key: str, valid_key: str,
                         count_key: str, require_positive: bool = True) -> None:
    depth, valid, count = arrays.get(depth_key), arrays.get(valid_key), arrays.get(count_key)
    if depth is None or valid is None or count is None:
        return
    actual = int(np.count_nonzero(valid))
    recorder.check("support.count", int(count) == actual,
                   f"{label}: {count_key}={int(count)} but valid pixels={actual}")
    if require_positive:
        recorder.check("support.zero", actual > 0, f"{label}: {valid_key} is all False (0 supported pixels)")


def _resolve_path(entry, base: Path, evidence_dir: Path | None = None) -> Path | None:
    if not isinstance(entry, str) or not entry:
        return None
    path = Path(entry)
    if path.is_absolute():
        return path
    if evidence_dir is not None and (evidence_dir / path).exists():
        return evidence_dir / path
    return base / path


def _check_receipts(recorder: Recorder, receipts, base: Path, evidence_dir: Path | None) -> None:
    for stage in REQUIRED_STAGES:
        value = receipts.get(stage) if isinstance(receipts, dict) else None
        attempt = _resolve_path(value, base, evidence_dir)
        if attempt is None or not (attempt / "exit.json").is_file():
            recorder.check("receipts.missing", False, f"stage {stage}: attempt exit.json missing for {value!r}")
            continue
        missing_files = [name for name in ("cmd.json", "stdout.log", "stderr.log") if not (attempt / name).is_file()]
        recorder.check("receipts.files", not missing_files, f"stage {stage}: missing receipt files {missing_files}")
        try:
            receipt = json.loads((attempt / "exit.json").read_text())
        except (json.JSONDecodeError, OSError) as error:
            recorder.check("receipt.schema", False, f"stage {stage}: cannot read exit.json: {error!r}")
            continue
        code = receipt.get("actual_exit_code")
        recorder.check("receipt.exit", isinstance(code, int) and code == 0, f"stage {stage}: actual_exit_code={code!r}")
        recorder.check("receipt.timeout", receipt.get("timed_out") is False, f"stage {stage}: timed_out={receipt.get('timed_out')!r}")
        recorder.check(
            "receipt.schema",
            isinstance(receipt.get("measured_by"), str) and bool(receipt.get("measured_by"))
            and isinstance(receipt.get("ended_at"), str) and bool(receipt.get("ended_at")),
            f"stage {stage}: measured_by/ended_at missing",
        )


def _check_signatures(recorder: Recorder, signatures, base: Path) -> None:
    for stage in REQUIRED_STAGES:
        entry = signatures.get(stage) if isinstance(signatures, dict) else None
        path = _resolve_path(entry, base)
        if path is None or not path.is_file():
            recorder.check("signatures.missing", False, f"stage {stage}: signature file missing for {entry!r}")
            continue
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as error:
            recorder.check("signature.invalid", False, f"stage {stage}: unreadable signature: {error!r}")
            continue
        missing = [key for key in SIGNATURE_KEYS if key not in payload]
        inputs = payload.get("inputs")
        inputs_ok = (
            isinstance(inputs, list) and bool(inputs)
            and all(isinstance(item, dict) and isinstance(item.get("path"), str)
                    and isinstance(item.get("sha256"), str) and len(item.get("sha256", "")) == 64
                    for item in inputs)
        )
        code_sha = payload.get("code_sha256")
        recorder.check(
            "signature.invalid",
            not missing and inputs_ok and isinstance(payload.get("model"), str) and bool(payload.get("model"))
            and isinstance(payload.get("argv"), list) and bool(payload.get("argv"))
            and isinstance(code_sha, str) and len(code_sha) == 64,
            f"stage {stage}: signature invalid (missing={missing})",
        )


def _check_stage_configs(recorder: Recorder, base: Path) -> None:
    config_path = base / "configs" / "stage_configs" / "sceneflow_refine.json"
    if not config_path.is_file():
        recorder.check("sceneflow.config", False, f"resolved sceneflow config missing at {config_path}")
        return
    try:
        payload = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        recorder.check("sceneflow.config", False, f"unreadable sceneflow config: {error!r}")
        return
    config = payload.get("config")
    clipped_points = payload.get("clipped_points")
    clipped_frames = payload.get("clipped_frames")
    resolved = config if isinstance(config, dict) else {}
    min_depth = resolved.get("min_depth")
    max_depth = resolved.get("max_depth")
    max_motion = resolved.get("max_motion")
    fields_ok = bool(resolved)
    for key, value in (resolved.items() if isinstance(resolved, dict) else []):
        if key in SCENEFLOW_CONFIG_BOOL_KEYS:
            fields_ok = fields_ok and isinstance(value, bool)
        else:
            fields_ok = fields_ok and _is_number(value)
    ok = (
        _is_number(min_depth) and _is_number(max_depth) and _is_number(max_motion)
        and float(min_depth) > 0 and float(max_depth) > float(min_depth) and float(max_motion) > 0
        and isinstance(clipped_points, int) and clipped_points >= 0
        and isinstance(clipped_frames, int) and clipped_frames >= 0
        and fields_ok
    )
    recorder.check(
        "sceneflow.config", ok,
        f"sceneflow_refine.json must resolve min/max_depth, max_motion, Config values and clipping counts "
        f"explicitly (default 5m clipping must not be silent): {config_path}",
    )
    if ok and clipped_points > 0:
        recorder.check("gate.silent_clipping", clipped_frames > 0,
                       f"clipped_points={clipped_points} without recorded clipped_frames")


def _check_ndr_config(recorder: Recorder, base: Path) -> None:
    config_path = base / "configs" / "stage_configs" / "ndr.json"
    if not config_path.is_file():
        recorder.check("ndr.resolved_config", False, f"resolved ndr config missing at {config_path}")
        return
    try:
        config = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        recorder.check("ndr.resolved_config", False, f"unreadable ndr config: {error!r}")
        return
    fallback_ok = (
        config.get("fallback_forbidden") is True
        or config.get("fallback") in ("forbidden", "disabled", "none")
        or "fallback 禁止" in str(config.get("backend_policy", ""))
        or "fallback禁止" in str(config.get("backend_policy", ""))
    )
    recorder.check(
        "ndr.resolved_config",
        config.get("backend") == "rust" and config.get("iterations") == 40 and fallback_ok,
        f"NDR resolved config must be backend=rust, iterations=40, fallback forbidden "
        f"(PERFORMANCE_DECISION; numpy/auto is rejected): {config!r}",
    )


def _check_manifest(recorder: Recorder, manifest: dict, dataset: str, expected_count: int,
                    base: Path, evidence_dir: Path | None) -> tuple[int, int, int] | None:
    missing = [key for key in MANIFEST_KEYS if key not in manifest]
    recorder.check("manifest.keys", not missing, f"missing: {missing}")
    recorder.check("manifest.dataset", manifest.get("dataset") == dataset,
                   f"expected {dataset!r}, found {manifest.get('dataset')!r}")
    frames = manifest.get("frames")
    frames_ok = isinstance(frames, list)
    declared = len(frames) if frames_ok else -1
    recorder.check("frames.count",
                   frames_ok and declared == expected_count and manifest.get("expected_frame_count") == expected_count,
                   f"frames={declared} expected_frame_count={manifest.get('expected_frame_count')!r} expected={expected_count}")
    original = manifest.get("original_size")
    dimensions: tuple[int, int] | None = (int(original[0]), int(original[1])) if _is_size(original) else None
    recorder.check("manifest.original_size", dimensions is not None, f"original_size invalid: {original!r}")
    recorder.check("manifest.units",
                   manifest.get("units") == UNITS and manifest.get("metric_calibrated") is False,
                   f"units must be {UNITS!r} with metric_calibrated=false, got {manifest.get('units')!r}/"
                   f"{manifest.get('metric_calibrated')!r}")
    calibration = manifest.get("calibration")
    targets = calibration.get("targets") if isinstance(calibration, dict) else None
    targets_ok = (
        (isinstance(targets, str) and bool(targets))
        or (isinstance(targets, list) and {"depth", "camera_translation", "scene_flow_3d"}.issubset(set(targets)))
    )
    recorder.check(
        "manifest.calibration",
        isinstance(calibration, dict)
        and _is_number(calibration.get("single_scalar")) and float(calibration["single_scalar"]) > 0
        and calibration.get("applied") is False
        and targets_ok,
        f"calibration must expose one post-calibration scalar for depth/camera translation/3D flow: {calibration!r}",
    )
    stages = manifest.get("stages")
    missing_stages = [stage for stage in REQUIRED_STAGES if not isinstance(stages, list) or stage not in stages]
    ordered = False
    if isinstance(stages, list) and not missing_stages:
        positions = [stages.index(stage) for stage in REQUIRED_STAGES]
        ordered = positions == sorted(positions)
    recorder.check("manifest.stages", not missing_stages and ordered,
                   f"stages must contain the required chain in order; missing={missing_stages}")
    optional = manifest.get("optional_stages")
    recorder.check("manifest.optional", isinstance(optional, list) and "moge" in optional,
                   f"optional_stages must list moge (comparison only): {optional!r}")
    models = manifest.get("models")
    model_stages = ("lingbot", "ndr", "sceneflow", "sr", "temporal")
    missing_models = [stage for stage in model_stages
                      if not isinstance(models, dict) or not isinstance(models.get(stage), str) or not models.get(stage)]
    if not isinstance(models, dict) or not (isinstance(models.get("arrow"), str) and models.get("arrow")
                                            or isinstance(models.get("anchor"), str) and models.get("anchor")):
        missing_models.append("arrow|anchor")
    recorder.check("models.missing", not missing_models,
                   f"missing model metadata (arrow/anchor + {model_stages}): {missing_models}")
    config_hash = manifest.get("config_hash")
    recorder.check("manifest.config_hash", isinstance(config_hash, str) and bool(config_hash),
                   f"config_hash missing: {config_hash!r}")
    if frames_ok:
        frame_indices = [entry.get("frame_index") for entry in frames if isinstance(entry, dict)]
        source_indices = [entry.get("source_index") for entry in frames if isinstance(entry, dict)]
        recorder.check("frames.unique",
                       len(set(frame_indices)) == len(frame_indices) and all(isinstance(i, int) for i in frame_indices),
                       "frame_index entries must be unique integers")
        recorder.check("frames.source_unique",
                       len(set(source_indices)) == len(source_indices)
                       and all(isinstance(i, int) for i in source_indices)
                       and set(source_indices) == set(range(expected_count)),
                       "source_index entries must be unique and cover [0, expected_count)")
        bad_hash = [entry.get("frame_index") for entry in frames
                    if isinstance(entry, dict) and not (isinstance(entry.get("rgb_sha256"), str)
                                                        and len(entry["rgb_sha256"]) == 64)]
        recorder.check("frames.rgb_sha256", not bad_hash, f"frames with invalid rgb_sha256: {bad_hash[:5]}")
        if dimensions is not None:
            wrong = [entry.get("frame_index") for entry in frames
                     if isinstance(entry, dict) and (entry.get("height"), entry.get("width")) != dimensions]
            recorder.check("frames.size", not wrong, f"frame entries with wrong size: {wrong[:5]}")
    _check_receipts(recorder, manifest.get("stage_receipts"), base, evidence_dir)
    _check_signatures(recorder, manifest.get("stage_signatures"), base)
    _check_stage_configs(recorder, base)
    _check_ndr_config(recorder, base)
    if dimensions is None:
        return None
    return expected_count, dimensions[0], dimensions[1]


def _check_frame_metadata(recorder: Recorder, label: str, arrays: dict, dataset: str,
                          *, require_extras: bool = False) -> dict | None:
    metadata = _metadata(arrays)
    if metadata is None:
        recorder.check("metadata.missing", False, f"{label}: frame metadata JSON missing/unreadable")
        return None
    missing = [key for key in FRAME_METADATA_KEYS if key not in metadata]
    recorder.check("metadata.keys", not missing, f"{label}: missing metadata keys {missing}")
    recorder.check(
        "metadata.values",
        metadata.get("format_version") == 1
        and metadata.get("dataset") == dataset
        and isinstance(metadata.get("stage"), str) and bool(metadata.get("stage"))
        and isinstance(metadata.get("model"), str) and bool(metadata.get("model"))
        and isinstance(metadata.get("params"), dict)
        and isinstance(metadata.get("input_sha256"), (str, list, dict))
        and isinstance(metadata.get("argv"), list) and bool(metadata.get("argv"))
        and isinstance(metadata.get("status"), str) and bool(metadata.get("status"))
        and isinstance(metadata.get("created_at"), str) and bool(metadata.get("created_at"))
        and metadata.get("units") == UNITS
        and metadata.get("metric_calibrated") is False
        and (not require_extras or (isinstance(metadata.get("stages"), (list, dict))
                                    and isinstance(metadata.get("calibration"), dict))),
        f"{label}: stage metadata values invalid (units={metadata.get('units')!r}, "
        f"metric_calibrated={metadata.get('metric_calibrated')!r}, "
        f"stages={type(metadata.get('stages')).__name__}, calibration={type(metadata.get('calibration')).__name__})",
    )
    return metadata


def _check_arrow(recorder: Recorder, label: str, arrays: dict, H: int, W: int) -> dict:
    keys = ("arrow_depth", "arrow_valid", "arrow_support_count", "arrow_scale", "arrow_clip_index")
    if not _require_keys(recorder, label, arrays, keys):
        return {}
    _check_shape(recorder, label, arrays, "arrow_depth", (H, W))
    _check_dtype(recorder, label, arrays, "arrow_depth", np.float32)
    _check_shape(recorder, label, arrays, "arrow_valid", (H, W))
    _check_dtype(recorder, label, arrays, "arrow_valid", np.bool_)
    depth = arrays["arrow_depth"]
    valid = arrays["arrow_valid"].astype(bool)
    if depth.shape == (H, W) and valid.shape == (H, W):
        _check_depth_valid(recorder, f"{label}.arrow", depth, valid)
    _check_support_count(recorder, f"{label}.arrow", arrays, "arrow_depth", "arrow_valid", "arrow_support_count")
    scale = _scalar(arrays, "arrow_scale")
    recorder.check("arrow.scale", scale is not None and scale > 0,
                   f"{label}: arrow_scale must be a finite positive scalar, got {scale!r}")
    clip = arrays.get("arrow_clip_index")
    return {"depth": depth, "valid": valid, "scale": scale,
            "clip": int(clip.reshape(-1)[0]) if clip is not None and clip.size == 1 else None}


def _check_geometry(recorder: Recorder, label: str, arrays: dict, H: int, W: int) -> dict:
    keys = ("arrow_K", "arrow_K_valid", "arrow_pose_reference_index", "arrow_pose_valid",
            "arrow_camera_to_reference", "arrow_camera_to_reconstruct", "arrow_camera_to_metric")
    if not _require_keys(recorder, label, arrays, keys):
        return {}
    for key in ("arrow_camera_to_reference", "arrow_camera_to_reconstruct", "arrow_camera_to_metric"):
        _check_shape(recorder, label, arrays, key, (4, 4))
    _check_shape(recorder, label, arrays, "arrow_K", (3, 3))
    k_valid = _bool_scalar(arrays, "arrow_K_valid")
    if k_valid:
        k = arrays["arrow_K"].astype(np.float64)
        if k.shape == (3, 3):
            pinhole = bool(np.all(np.abs(k[2, :] - np.array([0.0, 0.0, 1.0])) <= 1e-4)
                           and abs(k[0, 1]) <= K_OFFDIAG_TOLERANCE and abs(k[1, 0]) <= K_OFFDIAG_TOLERANCE)
            pixel_scale = k[0, 0] > 1.0 and k[1, 1] > 1.0 and k[0, 2] >= 1.0 and k[1, 2] >= 1.0
            bounds = 0 <= k[0, 2] <= W - 1 and 0 <= k[1, 2] <= H - 1
            recorder.check("geometry.K_pixel", pinhole and pixel_scale,
                           f"{label}: arrow_K must be pixel-centre intrinsics (normalized K is rejected)")
            recorder.check("geometry.K_bounds", bounds, f"{label}: principal point outside the image")
    pose_valid = _bool_scalar(arrays, "arrow_pose_valid")
    if pose_valid:
        for key in ("arrow_camera_to_reference", "arrow_camera_to_reconstruct", "arrow_camera_to_metric"):
            matrix = arrays[key].astype(np.float64)
            if matrix.shape == (4, 4):
                finite = bool(np.all(np.isfinite(matrix)))
                bottom = bool(np.all(np.abs(matrix[3, :] - np.array([0.0, 0.0, 0.0, 1.0])) <= 1e-4))
                recorder.check("geometry.camera", finite and bottom, f"{label}: {key} must be finite with bottom row [0,0,0,1]")
    ref = arrays.get("arrow_pose_reference_index")
    if ref is not None and ref.size == 1:
        recorder.check("geometry.pose_reference", int(ref.reshape(-1)[0]) >= 0,
                       f"{label}: arrow_pose_reference_index must be a valid source index")
    return {
        "pose_valid": pose_valid,
        "camera_to_reference": arrays["arrow_camera_to_reference"].astype(np.float64),
        "camera_to_reconstruct": arrays["arrow_camera_to_reconstruct"].astype(np.float64),
        "camera_to_metric": arrays["arrow_camera_to_metric"].astype(np.float64),
    }


def _check_prep(recorder: Recorder, label: str, arrays: dict, H: int, W: int) -> None:
    if not _require_keys(recorder, label, arrays, ("prep_edges", "prep_normal", "prep_normal_valid")):
        return
    _check_shape(recorder, label, arrays, "prep_edges", (H, W))
    _check_dtype(recorder, label, arrays, "prep_edges", np.float32)
    _check_shape(recorder, label, arrays, "prep_normal", (H, W, 3))
    _check_shape(recorder, label, arrays, "prep_normal_valid", (H, W))
    edges = arrays["prep_edges"].astype(np.float64)
    if edges.shape == (H, W):
        recorder.check("prep.edges", bool(np.all(np.isfinite(edges)) and np.all((edges >= 0) & (edges <= 1))),
                       f"{label}: prep_edges must be finite in [0,1]")
    normals = arrays["prep_normal"].astype(np.float64)
    normal_valid = arrays["prep_normal_valid"].astype(bool)
    if normals.shape == (H, W, 3) and normal_valid.shape == (H, W):
        norms = np.linalg.norm(normals, axis=-1)
        recorder.check("prep.normal",
                       bool(np.all(np.isfinite(normals[normal_valid]))) and
                       bool(np.all(np.abs(norms[normal_valid] - 1.0) <= 5e-2)) if normal_valid.any() else True,
                       f"{label}: prep_normal must be finite unit vectors where prep_normal_valid")


def _check_lingbot(recorder: Recorder, label: str, arrays: dict, H: int, W: int,
                   arrow_valid: np.ndarray | None = None) -> dict:
    keys = ("lingbot_depth_in", "lingbot_depth_raw", "lingbot_depth", "lingbot_valid",
            "lingbot_support_count", "lingbot_mask", "lingbot_K_norm", "lingbot_K_pixel", "lingbot_scale_fix")
    if not _require_keys(recorder, label, arrays, keys):
        return {}
    for key in ("lingbot_depth_in", "lingbot_depth_raw", "lingbot_depth"):
        _check_shape(recorder, label, arrays, key, (H, W))
        _check_dtype(recorder, label, arrays, key, np.float32)
    _check_shape(recorder, label, arrays, "lingbot_mask", (H, W))
    valid = arrays["lingbot_valid"].astype(bool)
    mask = arrays["lingbot_mask"].astype(bool)
    # Mask ownership: lingbot_depth_in is the sparse ARROW anchor (input mask = arrow_valid),
    # while the dense model outputs (raw / reanchored depth) use the output mask (lingbot_valid).
    depth_in = arrays["lingbot_depth_in"]
    if depth_in.shape == (H, W):
        if arrow_valid is not None and arrow_valid.shape == (H, W):
            _check_depth_valid(recorder, f"{label}.lingbot_depth_in", depth_in, arrow_valid)
        else:
            recorder.check("lingbot.input_mask", False,
                           f"{label}: anchor valid mask unavailable for lingbot_depth_in ownership check")
    for key in ("lingbot_depth_raw", "lingbot_depth"):
        depth = arrays[key]
        if depth.shape == (H, W) and valid.shape == (H, W):
            _check_depth_valid(recorder, f"{label}.{key}", depth, valid)
    _check_support_count(recorder, f"{label}.lingbot", arrays, "lingbot_depth", "lingbot_valid", "lingbot_support_count")
    k_norm = arrays["lingbot_K_norm"].astype(np.float64)
    if k_norm.shape == (3, 3):
        normalized = (
            bool(np.all(np.isfinite(k_norm)))
            and bool(np.all(np.abs(k_norm[2, :] - np.array([0.0, 0.0, 1.0])) <= 1e-4))
            and k_norm[0, 0] > 0 and k_norm[1, 1] > 0
            and 0 < k_norm[0, 2] < 1 and 0 < k_norm[1, 2] < 1
        )
        recorder.check("lingbot.K_normalized", normalized,
                       f"{label}: LingBot API requires normalized intrinsics (0<cx,cy<1)")
    k_pixel = arrays["lingbot_K_pixel"].astype(np.float64)
    if k_pixel.shape == (3, 3) and valid.any():
        pixel = (k_pixel[0, 0] > 1.0 and k_pixel[1, 1] > 1.0
                 and 0 <= k_pixel[0, 2] <= W - 1 and 0 <= k_pixel[1, 2] <= H - 1)
        recorder.check("lingbot.K_pixel", pixel, f"{label}: lingbot_K_pixel must be pixel-centre")
    scale_fix = _scalar(arrays, "lingbot_scale_fix")
    recorder.check("lingbot.scale_fix", scale_fix is not None and scale_fix > 0,
                   f"{label}: lingbot_scale_fix must be finite positive, got {scale_fix!r}")
    return {"depth": arrays["lingbot_depth"], "valid": valid, "mask": mask}


def _check_ndr(recorder: Recorder, label: str, arrays: dict, H: int, W: int,
               lingbot: dict) -> dict:
    keys = ("ndr_depth_in", "ndr_depth", "ndr_valid", "ndr_support_count")
    if not _require_keys(recorder, label, arrays, keys):
        return {}
    for key in ("ndr_depth_in", "ndr_depth"):
        _check_shape(recorder, label, arrays, key, (H, W))
        _check_dtype(recorder, label, arrays, key, np.float32)
    valid = arrays["ndr_valid"].astype(bool)
    depth = arrays["ndr_depth"]
    if depth.shape == (H, W) and valid.shape == (H, W):
        _check_depth_valid(recorder, f"{label}.ndr", depth, valid)
    _check_support_count(recorder, f"{label}.ndr", arrays, "ndr_depth", "ndr_valid", "ndr_support_count")
    if lingbot.get("valid") is not None and lingbot["valid"].shape == valid.shape:
        recorder.check("ndr.valid_inflation", not bool(np.any(valid & ~lingbot["valid"])),
                       f"{label}: ndr_valid must not claim pixels outside lingbot_valid")
    depth_in = arrays["ndr_depth_in"]
    if lingbot.get("depth") is not None and lingbot["depth"].shape == depth_in.shape:
        _compare(recorder, "ndr.depth_in_mismatch", f"{label}: ndr_depth_in vs lingbot_depth",
                 depth_in, lingbot["depth"], valid, lingbot["valid"])
    return {"depth": depth, "valid": valid}


def _check_sceneflow(recorder: Recorder, label: str, arrays: dict, H: int, W: int,
                     ndr: dict) -> dict:
    keys = ("sceneflow_depth", "sceneflow_valid", "sceneflow_support_count",
            "sceneflow_next_depth", "sceneflow_next_valid", "sceneflow_next_support_count",
            "sceneflow_pair_index", "sceneflow_compensation_valid")
    if not _require_keys(recorder, label, arrays, keys):
        return {}
    for key in ("sceneflow_depth", "sceneflow_next_depth"):
        _check_shape(recorder, label, arrays, key, (H, W))
        _check_dtype(recorder, label, arrays, key, np.float32)
    _check_shape(recorder, label, arrays, "sceneflow_compensation_valid", (H, W))
    valid = arrays["sceneflow_valid"].astype(bool)
    next_valid = arrays["sceneflow_next_valid"].astype(bool)
    compensation = arrays["sceneflow_compensation_valid"].astype(bool)
    depth = arrays["sceneflow_depth"]
    next_depth = arrays["sceneflow_next_depth"]
    if depth.shape == (H, W) and valid.shape == (H, W):
        _check_depth_valid(recorder, f"{label}.sceneflow", depth, valid)
    if next_depth.shape == (H, W) and next_valid.shape == (H, W):
        _check_depth_valid(recorder, f"{label}.sceneflow_next", next_depth, next_valid)
    pair = arrays["sceneflow_pair_index"].astype(np.int64)
    pair_row = None
    if pair.shape == (2,):
        pair_row = (int(pair[0]), int(pair[1]))
        recorder.check("sceneflow.pairs", pair_row == (-1, -1) or (pair_row[0] >= 0 and pair_row[1] == pair_row[0] + 1),
                       f"{label}: sceneflow_pair_index must be -1 or (self, self+1), got {pair_row}")
        if pair_row != (-1, -1) and compensation.shape == (H, W):
            recorder.check("denom.zero", bool(np.any(compensation)),
                           f"{label}: paired frame needs positive compensation (overlap) pixels")
            if next_valid.shape == (H, W):
                recorder.check("sceneflow.next_beyond_compensation",
                               not bool(np.any(next_valid & ~compensation)),
                               f"{label}: sceneflow_next_valid must stay inside compensation_valid (b content in a)")
        if pair_row == (-1, -1) and next_valid.shape == (H, W):
            recorder.check("sceneflow.next_without_pair", not bool(np.any(next_valid)),
                           f"{label}: no pair but sceneflow_next_valid claims support")
    else:
        recorder.check("shape.mismatch", False, f"{label}: sceneflow_pair_index must be int64[2]")
    _check_support_count(recorder, f"{label}.sceneflow", arrays, "sceneflow_depth", "sceneflow_valid", "sceneflow_support_count")
    _check_support_count(recorder, f"{label}.sceneflow_next", arrays, "sceneflow_next_depth",
                         "sceneflow_next_valid", "sceneflow_next_support_count",
                         require_positive=pair_row is not None and pair_row != (-1, -1))
    ndr_valid = ndr.get("valid")
    if ndr_valid is not None and valid.shape == ndr_valid.shape:
        recorder.check("sceneflow.support_inflation", not bool(np.any(valid & ~ndr_valid)),
                       f"{label}: sceneflow_valid must not fabricate support beyond the NDR input (未支持画素はvalid=false)")
    return {"depth": depth, "valid": valid, "pair": pair_row, "compensation": compensation}


def _sr_from_alt_storage(label: str, arrays: dict, sr_alt: dict | None, H: int, W: int,
                         pair_row: tuple[int, int] | None = None) -> dict:
    """Join full-res SR arrays from stages/sr.npz when the frame NPZ omits them.

    Pipeline ACK_REV allows the >1GB-class full-res arrays to live in
    stages/sr.npz with per-frame low arrays only; both storages are accepted.
    """
    sr_arrays = dict(arrays)
    if sr_alt is None:
        return sr_arrays
    pair = arrays.get("sr_pair_index")
    if pair_row is None and pair is not None and pair.size >= 2:
        pair_row = (int(pair.reshape(-1)[0]), int(pair.reshape(-1)[1]))
    alt_pairs = sr_alt.get("pair_index")
    row_index = None
    if pair_row is not None and alt_pairs is not None and alt_pairs.ndim == 2:
        matches = np.flatnonzero((alt_pairs[:, 0] == pair_row[0]) & (alt_pairs[:, 1] == pair_row[1]))
        if matches.size:
            row_index = int(matches[0])
    if row_index is None:
        unpaired = pair_row == (-1, -1) or (pair_row is None and pair is not None
                                            and (int(pair.reshape(-1)[0]) < 0))
        if unpaired and "sr_flow_low" in arrays:
            sr_arrays["sr_flow"] = np.full((H, W, 3), np.nan, np.float32)
            sr_arrays["sr_valid"] = np.zeros((H, W), dtype=bool)
            sr_arrays["sr_support_count"] = np.int64(0)
        return sr_arrays
    sr_arrays["sr_flow"] = sr_alt["flow3d"][row_index]
    valid = np.asarray(sr_alt["flow3d_valid"][row_index], dtype=bool)
    sr_arrays["sr_valid"] = valid
    sr_arrays["sr_support_count"] = np.int64(np.count_nonzero(valid))
    if "sr_pair_index" not in sr_arrays and pair_row is not None:
        sr_arrays["sr_pair_index"] = np.asarray(pair_row, dtype=np.int64)
    if "low_flow" in sr_alt:
        sr_arrays["sr_flow_low"] = sr_alt["low_flow"][row_index]
        if "sr_low_valid" not in sr_arrays:
            sr_arrays["sr_low_valid"] = np.isfinite(np.asarray(sr_arrays["sr_flow_low"], dtype=np.float64)).all(axis=-1)
    return sr_arrays


def _check_sr_empty(recorder: Recorder, label: str, arrays: dict) -> None:
    """Unpaired frames keep an explicit empty sr schema: pair [-1,-1], support 0,
    valid all False, flow all NaN. Fabricated finite flow/valid is rejected."""
    problems = []
    pair = arrays.get("sr_pair_index")
    pair_values = np.asarray(pair).reshape(-1).tolist()[:2] if pair is not None else None
    if pair_values != [-1, -1]:
        problems.append(f"sr_pair_index={pair_values}")
    valid = arrays.get("sr_valid")
    if valid is None or bool(np.asarray(valid).any()):
        problems.append("sr_valid not all False")
    count = arrays.get("sr_support_count")
    if count is None or int(np.asarray(count).reshape(-1)[0]) != 0:
        problems.append("sr_support_count != 0")
    flow = arrays.get("sr_flow")
    if flow is None or bool(np.isfinite(np.asarray(flow, dtype=np.float64)).any()):
        problems.append("sr_flow contains finite values")
    recorder.check("sr.unpaired_schema", not problems,
                   f"{label}: unpaired sr frame must be the explicit empty schema; problems={problems}")


def _check_sr(recorder: Recorder, label: str, arrays: dict, H: int, W: int,
              sceneflow: dict, ndr: dict) -> dict:
    keys = ("sr_flow", "sr_valid", "sr_support_count", "sr_flow_low", "sr_pair_index")
    if not _require_keys(recorder, label, arrays, keys):
        return {}
    _check_shape(recorder, label, arrays, "sr_flow", (H, W, 3))
    _check_dtype(recorder, label, arrays, "sr_flow", np.float32)
    valid = arrays["sr_valid"].astype(bool)
    flow = arrays["sr_flow"].astype(np.float64)
    if flow.shape == (H, W, 3) and valid.shape == (H, W):
        recorder.check("sr.flow_finite", not bool(np.any(valid & ~np.isfinite(flow).all(axis=-1))),
                       f"{label}: sr_flow must be finite where sr_valid")
    low_flow = arrays["sr_flow_low"].astype(np.float64)
    low_valid = arrays.get("sr_low_valid")
    if low_valid is None:
        low_valid = np.isfinite(low_flow).all(axis=-1)
    low_valid = np.asarray(low_valid, dtype=bool)
    if low_flow.ndim != 3 or low_flow.shape[-1] != 3:
        recorder.check("shape.mismatch", False, f"{label}: sr_flow_low must be [h,w,3]")
    support_key_ok = "sr_support_count" in arrays
    if support_key_ok:
        actual = int(np.count_nonzero(valid))
        recorder.check("support.count", int(arrays["sr_support_count"]) == actual,
                       f"{label}: sr_support_count={int(arrays['sr_support_count'])} but sr_valid pixels={actual}")
    sf_valid = sceneflow.get("valid")
    if sf_valid is not None and valid.shape == sf_valid.shape:
        recorder.check("sr.valid_beyond_support", not bool(np.any(valid & ~sf_valid)),
                       f"{label}: sr_valid must stay inside sceneflow_valid (low support)")
    high = ndr.get("depth")
    if high is not None and high.shape[:2] == valid.shape and isinstance(high, np.ndarray):
        depth_support = _finite_positive(high.astype(np.float64))
        recorder.check("sr.valid_without_depth", not bool(np.any(valid & ~depth_support)),
                       f"{label}: sr_valid must not claim pixels whose high-depth is unsupported (NaN/non-positive)")
    pair = arrays["sr_pair_index"].astype(np.int64)
    has_pair = pair.size == 0 or int(pair.reshape(-1)[0]) >= 0
    if low_flow.ndim == 3 and low_valid.shape == low_flow.shape[:2]:
        samples = _rescale_samples(flow, low_flow, valid, low_valid, H, W)
        # Real-data median ratio is a diagnostic record only (supervisor AUDIT_REVIEW):
        # a bare threshold must not reject correct interpolation on real distributions.
        detail = f"{label}: median|flow3d|/|low_flow|={samples.tolist() if samples.size else 'n/a'} (diagnostic only)"
        recorder.check("sr.rescale_diagnostic", True, detail)
        if has_pair:
            low_count = int(np.count_nonzero(low_valid))
            if low_count > 0 and not samples.size:
                recorder.check("denom.zero", False,
                               f"{label}: low support exists but zero valid SR samples on a paired frame (silent support loss)")
            preserved, checked = _unit_preservation(flow, low_flow, valid, low_valid, H, W)
            if checked > 0:
                recorder.check("sr.unit_preservation", preserved,
                               f"{label}: sr_flow magnitude differs from low_flow at {checked} locally-constant samples "
                               "(resolution-ratio scaling detected; strict guarantee is the constant-vector synthetic GT test)")
    return {"valid": valid, "flow": flow, "pair": pair}


def _rescale_samples(flow: np.ndarray, low_flow: np.ndarray, valid: np.ndarray,
                     low_valid: np.ndarray, H: int, W: int) -> np.ndarray:
    h, w = low_valid.shape
    low_mag = np.linalg.norm(low_flow, axis=-1)
    ys = np.clip(np.round((np.arange(h) + 0.5) * H / h - 0.5).astype(int), 0, H - 1)
    xs = np.clip(np.round((np.arange(w) + 0.5) * W / w - 0.5).astype(int), 0, W - 1)
    high_mag = np.linalg.norm(flow[np.ix_(ys, xs)], axis=-1)
    mask = valid[np.ix_(ys, xs)] & low_valid & np.isfinite(low_mag) & (low_mag > 1e-12) & np.isfinite(high_mag)
    if not bool(np.any(mask)):
        return np.empty(0, dtype=np.float64)
    return np.array([float(np.median(high_mag[mask]) / max(float(np.median(low_mag[mask])), 1e-12))])


def _unit_preservation(flow: np.ndarray, low_flow: np.ndarray, valid: np.ndarray,
                       low_valid: np.ndarray, H: int, W: int) -> tuple[bool, int]:
    """Strict no-rescaling check at locally constant low-flow regions.

    Where the low grid displacement is constant over a 3x3 low neighbourhood, the
    upsampled 3D displacement must equal that constant (a resolution-ratio
    multiplication changes it). General non-constant data is not constrained
    here; that is covered by the constant-vector synthetic GT test in core.
    """
    h, w = low_valid.shape
    ys = np.clip(np.round((np.arange(h) + 0.5) * H / h - 0.5).astype(int), 0, H - 1)
    xs = np.clip(np.round((np.arange(w) + 0.5) * W / w - 0.5).astype(int), 0, W - 1)
    high_sampled = flow[np.ix_(ys, xs)]
    high_valid = valid[np.ix_(ys, xs)]
    checked = 0
    violated = False
    for j in range(1, h - 1):
        for i in range(1, w - 1):
            if not low_valid[j, i] or not high_valid[j, i]:
                continue
            center = low_flow[j, i]
            neighbourhood = low_flow[j - 1:j + 2, i - 1:i + 2]
            if not np.all(low_valid[j - 1:j + 2, i - 1:i + 2]):
                continue
            if not np.all(np.isfinite(neighbourhood)):
                continue
            if float(np.max(np.abs(neighbourhood - center))) > 1e-6 * max(float(np.max(np.abs(center))), 1e-6):
                continue
            if not np.all(np.isfinite(center)) or not np.all(np.isfinite(high_sampled[j, i])):
                continue
            checked += 1
            if not np.allclose(high_sampled[j, i], center, atol=1e-4, rtol=1e-3):
                violated = True
    return (not violated, checked)


def _check_temporal(recorder: Recorder, label: str, arrays: dict, H: int, W: int) -> dict:
    keys = ("temporal_depth", "temporal_valid", "temporal_support_count", "temporal_common_support",
            "temporal_denom", "temporal_accepted_fwd", "temporal_accepted_bwd", "temporal_residual_stats")
    if not _require_keys(recorder, label, arrays, keys):
        return {}
    _check_shape(recorder, label, arrays, "temporal_depth", (H, W))
    _check_dtype(recorder, label, arrays, "temporal_depth", np.float32)
    valid = arrays["temporal_valid"].astype(bool)
    depth = arrays["temporal_depth"]
    if depth.shape == (H, W) and valid.shape == (H, W):
        _check_depth_valid(recorder, f"{label}.temporal", depth, valid)
    _check_support_count(recorder, f"{label}.temporal", arrays, "temporal_depth", "temporal_valid", "temporal_support_count")
    support = arrays["temporal_common_support"].astype(bool)
    denom = arrays["temporal_denom"]
    if support.shape == (H, W) and denom.size == 1:
        actual = int(np.count_nonzero(support))
        recorder.check("denom.mismatch", int(denom.reshape(-1)[0]) == actual,
                       f"{label}: temporal_denom={int(denom.reshape(-1)[0])} but common support pixels={actual}")
        recorder.check("denom.zero", actual > 0,
                       f"{label}: temporal common support denominator is 0 (must be > 0 per contract)")
    stats = arrays["temporal_residual_stats"].astype(np.float64)
    if stats.shape != (8,):
        recorder.check("shape.mismatch", False, f"{label}: temporal_residual_stats must be float64[8]")
        return {}
    accepted_fwd = _bool_scalar(arrays, "temporal_accepted_fwd")
    accepted_bwd = _bool_scalar(arrays, "temporal_accepted_bwd")
    index_of = {name: position for position, name in enumerate(RESIDUAL_COLUMNS)}
    for direction, accepted in (("fwd", accepted_fwd), ("bwd", accepted_bwd)):
        if accepted is None:
            recorder.check("temporal.accepted", False, f"{label}: temporal_accepted_{direction} must be a bool scalar")
            continue
        if not accepted:
            # Absent/rejected directions: producer stores NaN blocks (frame0 bwd, last fwd) or
            # computed-then-rejected values; no further constraint here.
            continue
        block = stats[index_of[f"{direction}_before_med"]: index_of[f"{direction}_before_med"] + 4]
        before_med, after_med, before_n, after_n = block
        recorder.check("temporal.stats_finite",
                       bool(np.all(np.isfinite(block))),
                       f"{label}: accepted {direction} stats must be finite (producer per-direction block): {block.tolist()}")
        integers_ok = all(
            np.isfinite(value) and abs(float(value) - round(float(value))) < 1e-6 and float(value) > 0
            for value in (before_n, after_n)
        )
        recorder.check("denom.zero", integers_ok,
                       f"{label}: accepted {direction} denominators must be positive integers, got n={before_n}/{after_n}")
        if np.all(np.isfinite(block)):
            recorder.check("gate.silent_worsening", after_med <= before_med,
                           f"{label}: accepted {direction} fusion worsened the residual median "
                           f"({before_med} -> {after_med})")
    return {"denom": int(denom.reshape(-1)[0]) if denom.size == 1 else 0,
            "accepted_fwd": accepted_fwd, "accepted_bwd": accepted_bwd}


def _check_final(recorder: Recorder, label: str, arrays: dict, H: int, W: int) -> dict:
    keys = ("final_depth", "final_valid", "final_support_count")
    if not _require_keys(recorder, label, arrays, keys):
        return {}
    _check_shape(recorder, label, arrays, "final_depth", (H, W))
    _check_dtype(recorder, label, arrays, "final_depth", np.float32)
    valid = arrays["final_valid"].astype(bool)
    depth = arrays["final_depth"]
    if depth.shape == (H, W) and valid.shape == (H, W):
        _check_depth_valid(recorder, f"{label}.final", depth, valid)
    _check_support_count(recorder, f"{label}.final", arrays, "final_depth", "final_valid", "final_support_count")
    return {"depth": depth, "valid": valid}


def _compare(recorder: Recorder, check_id: str, label: str, a: np.ndarray, b: np.ndarray,
             mask_a: np.ndarray, mask_b: np.ndarray) -> None:
    if a.shape != b.shape or mask_a.shape != a.shape or mask_b.shape != a.shape:
        recorder.check(check_id, False, f"{label}: shape mismatch {a.shape} vs {b.shape}")
        return
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    overlap = np.asarray(mask_a, dtype=bool) & np.asarray(mask_b, dtype=bool) & np.isfinite(left) & np.isfinite(right)
    count = int(np.count_nonzero(overlap))
    if count == 0:
        recorder.check("denom.zero", False, f"{label}: no overlapping valid pixels for comparison")
        return
    close = np.all(np.abs(left[overlap] - right[overlap]) <= REL_TOLERANCE * np.maximum(np.abs(right[overlap]), 1e-12))
    recorder.check(check_id, bool(close), f"{label}: values disagree on {count} overlapping pixels")


def _aggregate_pose_to_reconstruct(recorder: Recorder, records: list[dict]) -> None:
    clips: dict[int, list[dict]] = {}
    for record in records:
        if record.get("clip") is not None and record.get("pose_valid") and record.get("c2r") is not None:
            clips.setdefault(record["clip"], []).append(record)
    problems = []
    for clip, rows in sorted(clips.items()):
        if len(rows) < 2:
            continue
        first = min(rows, key=lambda row: row["source_index"])
        p0 = first["c2r"]
        if p0.shape != (4, 4) or not np.all(np.isfinite(p0)):
            problems.append(f"clip {clip}: first frame camera_to_reference not finite")
            continue
        try:
            pinv = np.linalg.inv(p0)
        except np.linalg.LinAlgError:
            problems.append(f"clip {clip}: first frame camera_to_reference singular")
            continue
        for row in rows:
            expected = pinv @ row["c2r"]
            if row["c2rec"] is None or row["c2rec"].shape != (4, 4) or not np.allclose(
                row["c2rec"], expected, atol=POSE_TOLERANCE, rtol=POSE_TOLERANCE
            ):
                problems.append(f"clip {clip} frame {row['source_index']}: camera_to_reconstruct != inv(P0) @ camera_to_reference")
    recorder.check("geometry.pose_to_reconstruct", not problems,
                   f"pose reference=middle -> reconstruct(0) conversion invariant violated: {problems[:3]}")


def _aggregate_clip_scale(recorder: Recorder, records: list[dict]) -> None:
    clips: dict[int, list[float]] = {}
    for record in records:
        if record.get("clip") is not None and record.get("scale") is not None:
            clips.setdefault(record["clip"], []).append(record["scale"])
    problems = []
    for clip, scales in sorted(clips.items()):
        values = np.asarray(scales, dtype=np.float64)
        if values.size > 1 and not np.allclose(values, values[0], rtol=REL_TOLERANCE, atol=0):
            problems.append(f"clip {clip}: per-frame scales {values.tolist()[:5]}")
    recorder.check("arrow.per_frame_normalization", not problems,
                   f"arrow_scale must be one scalar per clip (per-frame min-max normalization is forbidden): {problems[:3]}")


def _aggregate_geometry_invariants(recorder: Recorder, records: list[dict]) -> None:
    """SO3 rigidity and a single post-calibration scalar (supervisor REVIEW_GEOMETRY)."""
    so3_problems = []
    for record in records:
        if not record.get("pose_valid"):
            continue
        for name in ("c2r", "c2rec", "c2m"):
            matrix = record.get(name)
            if matrix is None or matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
                continue
            rotation = matrix[:3, :3]
            orthonormal = float(np.max(np.abs(rotation @ rotation.T - np.eye(3)))) <= 1e-3
            det = float(np.linalg.det(rotation))
            if not orthonormal or abs(det - 1.0) > 1e-3:
                so3_problems.append(f"frame {record['source_index']} {name}: det={det:.4f}")
    recorder.check("geometry.SO3", not so3_problems,
                   f"rotation parts must be orthonormal with det=1 (scaled rotation / double scale detected): {so3_problems[:3]}")

    clips: dict[int, list[dict]] = {}
    for record in records:
        if (record.get("clip") is not None and record.get("pose_valid")
                and record.get("c2rec") is not None and record.get("c2m") is not None
                and record.get("scale") is not None):
            clips.setdefault(record["clip"], []).append(record)
    scale_problems, consistency_problems = [], []
    for clip, rows in sorted(clips.items()):
        rows.sort(key=lambda row: row["source_index"])
        first = rows[0]
        scale = float(first["scale"])
        if len(rows) > 1:
            d_recon = rows[-1]["c2rec"][:3, 3] - first["c2rec"][:3, 3]
            d_metric = rows[-1]["c2m"][:3, 3] - first["c2m"][:3, 3]
            if float(np.linalg.norm(d_recon)) > 1e-6:
                estimated = float(np.linalg.norm(d_metric)) / float(np.linalg.norm(d_recon))
                if abs(estimated - scale) > 2e-2 * abs(scale):
                    scale_problems.append(
                        f"clip {clip}: translation-implied scale {estimated:.4f} != arrow_scale {scale:.4f} "
                        "(scale must be applied to depth and translation exactly once)"
                    )
        r_align = first["c2m"][:3, :3] @ first["c2rec"][:3, :3].T
        first_translation = first["c2m"][:3, 3] - scale * (r_align @ first["c2rec"][:3, 3])
        for row in rows:
            r_row = row["c2m"][:3, :3] @ row["c2rec"][:3, :3].T
            if float(np.max(np.abs(r_row - r_align))) > 1e-3:
                consistency_problems.append(f"clip {clip} frame {row['source_index']}: R_align differs")
                break
            t_row = row["c2m"][:3, 3] - scale * (r_align @ row["c2rec"][:3, 3])
            if float(np.max(np.abs(t_row - first_translation))) > 1e-3:
                consistency_problems.append(
                    f"clip {clip} frame {row['source_index']}: t_align differs (per-frame scale/translation inconsistency)"
                )
                break
    recorder.check("geometry.scale_once", not scale_problems,
                   f"one scalar applied to depth+translation: {scale_problems[:3]}")
    recorder.check("geometry.clip_transform_consistency", not consistency_problems,
                   f"clip metric transform A = metric o reconstruct^-1 must be constant: {consistency_problems[:3]}")


def clip_membership(clips_manifest, frame_count: int) -> list[set[int]]:
    """Native clips manifest ([start,end), status=done) -> frame membership sets."""
    membership: list[set[int]] = [set() for _ in range(frame_count)]
    clips = clips_manifest.get("clips", []) if isinstance(clips_manifest, dict) else []
    for index, clip in enumerate(clips):
        if not isinstance(clip, dict) or clip.get("status") != "done":
            continue
        start, end = clip.get("start"), clip.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        for frame in range(max(0, start), min(frame_count, end)):
            membership[frame].add(index)
    return membership


def expected_pairs_from_clips(clips_manifest, frame_count: int) -> tuple[set[tuple[int, int]], int]:
    """Independent expected pair set from the native clips manifest.

    Adjacent frames sharing at least one done clip form a pair (F-1 under full overlap;
    F-C only when clips are disjoint); adjacent frames sharing none are cross-clip
    skipped. No producer helper is imported."""
    membership = clip_membership(clips_manifest, frame_count)
    pairs = {(i, i + 1) for i in range(frame_count - 1) if membership[i] & membership[i + 1]}
    skipped = sum(1 for i in range(frame_count - 1) if not (membership[i] & membership[i + 1]))
    return pairs, skipped


def _aggregate_sceneflow_pairs(recorder: Recorder, dataset_dir: Path, records: list[dict], F: int) -> None:
    clips_path = dataset_dir / "stages" / "arrow_clips" / "manifest.json"
    pairs_path = dataset_dir / "stages" / "sceneflow" / "pairs.json"
    if not clips_path.is_file():
        recorder.check("sceneflow.clips_manifest", False, f"native clips manifest missing: {clips_path}")
        return
    if not pairs_path.is_file():
        recorder.check("sceneflow.pairs_expected", False, f"sceneflow pairs.json missing: {pairs_path}")
        return
    try:
        clips_manifest = json.loads(clips_path.read_text())
        payload = json.loads(pairs_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        recorder.check("sceneflow.pairs_expected", False, f"unreadable clips manifest/pairs: {error!r}")
        return
    expected, skipped_expected = expected_pairs_from_clips(clips_manifest, F)
    membership = clip_membership(clips_manifest, F)
    entries = payload.get("pairs", []) if isinstance(payload, dict) else []
    actual = []
    clip_problems = []
    for entry in entries:
        pair = entry.get("pair_index") if isinstance(entry, dict) else None
        row = tuple(int(v) for v in pair) if isinstance(pair, list) and len(pair) == 2 else None
        actual.append(row)
        if row is None or row not in expected:
            continue
        common = membership[row[0]] & membership[row[1]]
        if entry.get("clip_index") not in common:
            clip_problems.append(f"{row}: clip_index={entry.get('clip_index')!r} not in {sorted(common)}")
    unique = all(row is not None for row in actual) and len(set(actual)) == len(actual)
    recorder.check("sceneflow.pairs_expected",
                   unique and set(actual) == expected,
                   f"pairs.json entries must equal the independently computed set "
                   f"(entries={len(actual)} unique={len(set(actual))} expected={len(expected)})")
    recorder.check("sceneflow.pairs_clip", not clip_problems,
                   f"pair clip_index must lie in the common membership: {clip_problems[:4]}")
    counts = payload.get("counts", {}) if isinstance(payload, dict) else {}
    recorder.check("sceneflow.cross_clip",
                   isinstance(counts.get("cross_clip_skipped"), int)
                   and counts["cross_clip_skipped"] == skipped_expected,
                   f"cross_clip_skipped={counts.get('cross_clip_skipped')!r} != computed {skipped_expected}")
    recorder.check("sceneflow.pairs_count_record",
                   isinstance(counts.get("pairs"), int) and counts["pairs"] == len(actual),
                   f"counts.pairs={counts.get('pairs')!r} != entries {len(actual)}")
    per_frame_pairs = [tuple(record["pair"]) for record in records
                       if record.get("pair") and record["pair"] != (-1, -1)]
    rows_ok = all(row in expected for row in per_frame_pairs)
    recorder.check("sceneflow.pairs_aggregate",
                   rows_ok and len(set(per_frame_pairs)) == len(per_frame_pairs),
                   f"per-frame pair rows must be unique and within the expected set ({len(per_frame_pairs)} rows)")
    recorder.check("sceneflow.pair_count", len(per_frame_pairs) == len(expected),
                   f"per-frame pairs={len(per_frame_pairs)} expected={len(expected)}")


def _aggregate_temporal_manifest(recorder: Recorder, temporal, denom_sum: int, F: int) -> None:
    required = ("pairs_compared", "denom_total", "error_before", "error_after", "worsened_frames",
                "rejected_frames")
    if not isinstance(temporal, dict):
        recorder.check("temporal.manifest_keys", False, f"manifest.temporal missing: {temporal!r}")
        return
    missing = [key for key in required if key not in temporal]
    if "edge_leakage" not in temporal and "edge_leakage_median" not in temporal:
        missing.append("edge_leakage|edge_leakage_median")
    if "anchor_deviation" not in temporal and "anchor_deviation_median" not in temporal:
        missing.append("anchor_deviation|anchor_deviation_median")
    recorder.check("temporal.manifest_keys", not missing, f"manifest.temporal missing keys {missing}")
    if missing:
        return
    denom_total = temporal.get("denom_total")
    recorder.check("denom.total_mismatch", isinstance(denom_total, int) and denom_total == denom_sum,
                   f"manifest.temporal.denom_total={denom_total!r} but per-frame sum={denom_sum}")
    recorder.check("denom.zero", isinstance(denom_total, int) and denom_total > 0,
                   f"manifest.temporal.denom_total={denom_total!r} must be > 0")
    before, after = temporal.get("error_before"), temporal.get("error_after")
    recorder.check("gate.values", _is_number(before) and _is_number(after) and float(before) >= 0 and float(after) >= 0,
                   f"temporal error values invalid: before={before!r} after={after!r}")
    worsened = temporal.get("worsened_frames")
    if _is_number(before) and _is_number(after):
        recorder.check("gate.silent_worsening", not (float(after) > float(before) and (not isinstance(worsened, int) or worsened == 0)),
                       f"error_after={after} > error_before={before} without recorded worsened_frames={worsened!r}")
    leakage = temporal.get("edge_leakage", temporal.get("edge_leakage_median"))
    recorder.check("gate.edge_leakage", _is_number(leakage) or leakage is None,
                   f"edge_leakage must be numeric (or explicit null when undefined): {leakage!r}")
    anchor = temporal.get("anchor_deviation", temporal.get("anchor_deviation_median"))
    recorder.check("gate.anchor", _is_number(anchor) or anchor is None,
                   f"anchor_deviation must be numeric (or explicit null when undefined): {anchor!r}")
    recorder.check("temporal.pairs", isinstance(temporal.get("pairs_compared"), int) and temporal["pairs_compared"] >= 1,
                   f"pairs_compared invalid: {temporal.get('pairs_compared')!r}")


def _aggregate_sr_manifest(recorder: Recorder, sr, sr_pair_count: int, temporal) -> None:
    required = ("pairs", "bilinear_vs_sr", "gt_gate")
    if not isinstance(sr, dict):
        recorder.check("sr.manifest_keys", False, f"manifest.sr missing: {sr!r}")
        return
    missing = [key for key in required if key not in sr]
    recorder.check("sr.manifest_keys", not missing, f"manifest.sr missing keys {missing}")
    if missing:
        return
    recorder.check("sr.pairs", sr.get("pairs") == sr_pair_count,
                   f"manifest.sr.pairs={sr.get('pairs')!r} but per-frame SR pairs={sr_pair_count}")
    bilinear = sr.get("bilinear_vs_sr")
    bilinear_ok = isinstance(bilinear, dict) and "abs_diff" in bilinear
    bilinear_problems = []
    if isinstance(bilinear, dict):
        for key, value in bilinear.items():
            if key == "note":
                if not isinstance(value, str):
                    bilinear_problems.append(f"{key} must be a string annotation")
                continue
            if key == "compared_pairs":
                if not (isinstance(value, int) and not isinstance(value, bool) and value >= 0):
                    bilinear_problems.append(f"{key} must be a nonnegative integer")
                continue
            if not _numeric_tree(value):
                bilinear_problems.append(f"{key} must be strictly numeric")
    recorder.check("sr.bilinear_record", bilinear_ok and not bilinear_problems,
                   f"bilinear_vs_sr numeric fields must be strict (note=string annotation only): "
                   f"problems={bilinear_problems[:4]}")
    residual = sr.get("reprojection_residual")
    temporal_before = temporal.get("error_before") if isinstance(temporal, dict) else None
    temporal_after = temporal.get("error_after") if isinstance(temporal, dict) else None
    residual_ok = _numeric_tree(residual) or (_is_number(temporal_before) and _is_number(temporal_after))
    recorder.check("sr.reprojection_record", residual_ok,
                   f"reprojection residual must be recorded on common masks (sr.reprojection_residual or "
                   f"manifest.temporal error_before/after): {residual!r}")
    consumption = sr.get("consumption")
    consumed = expected = unconsumed = None
    if isinstance(consumption, dict):
        consumed = consumption.get("pairs_consumed")
        expected = consumption.get("pairs_total")
        unconsumed = [] if consumed == expected else ["unknown"]
    elif isinstance(temporal, dict):
        consumed = temporal.get("sr_pairs_consumed")
        expected = temporal.get("sr_pairs_expected")
        unconsumed = temporal.get("sr_pairs_unconsumed")
    consumption_ok = (
        isinstance(consumed, int) and isinstance(expected, int)
        and expected == sr_pair_count and consumed == expected
        and isinstance(unconsumed, list) and not unconsumed
    )
    recorder.check("sr.consumption", consumption_ok,
                   f"every SR pair must be consumed by the temporal correspondence "
                   f"(consumed=expected={sr_pair_count}, unconsumed=[]): consumed={consumed!r} "
                   f"expected={expected!r} unconsumed={unconsumed!r}")
    recorder.check("sr.gt_gate", isinstance(sr.get("gt_gate"), str) and bool(sr.get("gt_gate")),
                   f"gt_gate must reference the synthetic-GT verification (core/audit): {sr.get('gt_gate')!r}")


def _check_visuals(recorder: Recorder, dataset_dir: Path, visuals, F: int, H: int, W: int,
                   ffprobe_bin: str, rerun_bin: str | None, rrd_min_bytes: int) -> None:
    # Panel-composite layout contract (not raw original size):
    # cell_width C (render arg/record), panel_h = max(2, round(H*C/W)) then even-floor,
    # frame width = 2 + 7*(C+2), frame height = panel_h + 4 + 16,
    # representative height = len({0, F//2, F-1}) * (frame_height + 6).
    report_path = dataset_dir / "visuals" / "render_report.json"
    render_report: dict = {}
    if report_path.is_file():
        try:
            loaded = json.loads(report_path.read_text())
            if isinstance(loaded, dict):
                render_report = loaded
        except (OSError, json.JSONDecodeError):
            render_report = {}
    recorder.check("visuals.report", bool(render_report), f"render_report.json missing/unreadable: {report_path}")
    cell = render_report.get("cell_width")
    panel_order = render_report.get("panel_order")
    expected_panels = ["rgb", "arrow", "lingbot", "ndr", "sceneflow", "temporal", "sr_flow"]
    frame_w = frame_h = png_h = None
    if (isinstance(cell, int) and not isinstance(cell, bool) and 2 <= cell <= 4096
            and panel_order == expected_panels):
        panel_h = max(2, int(round(H * cell / W)))
        panel_h -= panel_h % 2
        frame_w = 2 + 7 * (cell + 2)
        frame_h = panel_h + 4 + 16
        png_h = len({0, F // 2, F - 1}) * (frame_h + 6)
    layout_ok = frame_w is not None and frame_h is not None and png_h is not None
    recorder.check("visuals.layout_config", layout_ok,
                   f"cell_width/panel_order invalid for the 7-panel layout: {cell!r}/{panel_order!r}")
    if not isinstance(visuals, dict):
        recorder.check("visuals.mp4", False, f"manifest.visuals missing: {visuals!r}")
        recorder.check("visuals.rrd", False, f"manifest.visuals missing: {visuals!r}")
        return
    mp4_path = _resolve_path(visuals.get("mp4"), dataset_dir)
    png_path = _resolve_path(visuals.get("png"), dataset_dir)
    rrd_path = _resolve_path(visuals.get("rrd"), dataset_dir)
    sr_path = _resolve_path(render_report.get("sr_comparison_png"), dataset_dir)
    mp4_ok = mp4_path is not None and mp4_path.is_file() and mp4_path.stat().st_size > 0
    recorder.check("visuals.mp4", mp4_ok, f"comparison MP4 missing/empty: {visuals.get('mp4')!r}")
    png_problems = []
    if png_path is not None and png_path.is_file():
        try:
            from PIL import Image
            with Image.open(png_path) as image:
                if not layout_ok:
                    png_problems.append("layout unknown")
                elif (image.width, image.height) != (frame_w, png_h):
                    png_problems.append(f"png size {(image.width, image.height)} != layout {(frame_w, png_h)}")
        except ImportError:
            pass
    else:
        png_problems.append("missing")
    recorder.check("visuals.png", not png_problems, f"representative PNG problems: {png_problems}")
    recorder.check("visuals.rrd", rrd_path is not None and rrd_path.is_file() and rrd_path.stat().st_size >= rrd_min_bytes,
                   f"final point-cloud RRD missing/too small: {visuals.get('rrd')!r}")
    recorder.check("visuals.sr_comparison",
                   sr_path is not None and sr_path.is_file() and sr_path.stat().st_size > 0,
                   f"SR comparison PNG missing: {render_report.get('sr_comparison_png')!r}")
    if mp4_ok:
        command = [ffprobe_bin, "-v", "error", "-count_frames", "-select_streams", "v:0",
                   "-show_entries", "stream=nb_read_frames,width,height", "-of", "json", str(mp4_path)]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as error:
            recorder.check("visuals.ffprobe_error", False, f"ffprobe failed: {error!r}")
            return
        if result.returncode != 0:
            recorder.check("visuals.ffprobe_error", False, f"ffprobe rc={result.returncode}: {result.stderr.strip()[:200]}")
            return
        try:
            stream = json.loads(result.stdout)["streams"][0]
            frames_probed = int(stream["nb_read_frames"])
        except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as error:
            recorder.check("visuals.ffprobe_error", False, f"unparsable ffprobe output: {error!r}")
            return
        recorder.check("visuals.ffprobe", frames_probed == F, f"ffprobe frames={frames_probed} != F={F}")
        probed = (int(stream.get("width", -1)), int(stream.get("height", -1)))
        recorder.check("visuals.layout",
                       layout_ok and probed == (frame_w, frame_h),
                       f"mp4 size {probed} != independently computed layout {(frame_w, frame_h) if layout_ok else None}")
    if rerun_bin and rrd_path is not None and rrd_path.is_file():
        try:
            result = subprocess.run([rerun_bin, "rrd", "verify", str(rrd_path)], capture_output=True, text=True,
                                    check=False, timeout=60)
            recorder.check("visuals.rrd_verify", result.returncode == 0,
                           f"rerun rrd verify rc={result.returncode}: {(result.stderr or result.stdout).strip()[:200]}")
        except (OSError, subprocess.TimeoutExpired) as error:
            recorder.check("visuals.rrd_verify", False, f"rerun rrd verify failed: {error!r}")


def verify_dataset(
    dataset_dir: str | Path,
    *,
    dataset: str,
    expected_count: int,
    evidence_dir: str | Path | None = None,
    check_render: bool = False,
    ffprobe_bin: str = "ffprobe",
    rerun_bin: str | None = None,
    rrd_min_bytes: int = 128,
    allow_stage_merge: bool = False,
) -> dict:
    dataset_dir = Path(dataset_dir)
    evidence_dir = Path(evidence_dir) if evidence_dir is not None else None
    recorder = Recorder()
    report: dict = {
        "dataset": dataset,
        "dataset_dir": str(dataset_dir),
        "expected_count": expected_count,
        "units": UNITS,
        "metric_calibrated": False,
        "checks": recorder.checks,
        "failures": [],
        "ok": False,
        "limits": [
            "absolute metric scale is not verifiable without calibration/ground truth (units=arrow_provisional)",
            "producer quality numbers are structure/consistency-checked only",
            "SR boundary improvement is delegated to synthetic-GT tests (core); this verifier enforces storage/validity/scale",
            "RRD content is structural unless rerun_bin performs rrd verify",
            "real-data artifacts are audited in a later assignment",
        ],
    }
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        recorder.check("manifest.missing", False, f"no manifest at {manifest_path}")
        return _finalize(report, recorder)
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        recorder.check("manifest.invalid_json", False, repr(error))
        return _finalize(report, recorder)
    if not isinstance(manifest, dict):
        recorder.check("manifest.invalid_json", False, "manifest root must be an object")
        return _finalize(report, recorder)
    dims = _check_manifest(recorder, manifest, dataset, expected_count, dataset_dir, evidence_dir)
    if dims is None:
        return _finalize(report, recorder)
    F, H, W = dims
    sr_alt = None
    for candidate in (dataset_dir / "stages" / "sr.npz", dataset_dir.parent / "stages" / dataset / "sr.npz"):
        loaded = _open_npz(candidate)
        if loaded is not None and "flow3d" in loaded:
            sr_alt = loaded
            break

    records: list[dict] = []
    for entry in manifest.get("frames", []):
        if not isinstance(entry, dict):
            recorder.check("frame.file", False, f"malformed frame entry: {entry!r}")
            continue
        label = f"frame_index={entry.get('frame_index')}"
        index = entry.get("frame_index")
        arrays: dict = {}
        pack = None
        if isinstance(index, int):
            pack = _open_npz(dataset_dir / "frames" / f"{index:06d}.npz")
        if pack is not None:
            _check_frame_metadata(recorder, f"{label}/frames_pack", pack, dataset, require_extras=True)
            arrays = {key: value for key, value in pack.items() if key != "metadata"}
        elif allow_stage_merge:
            recorder.check("frames.pack_partial", True,
                           f"{label}: frames pack missing; explicit partial stage merge used")
            missing_stages = []
            for stage in PER_FRAME_STAGE_DIRS:
                stage_arrays = None
                if isinstance(index, int):
                    stage_arrays = _open_npz(dataset_dir / "stages" / stage / f"{index:06d}.npz")
                if stage_arrays is None:
                    missing_stages.append(stage)
                    continue
                _check_frame_metadata(recorder, f"{label}/{stage}", stage_arrays, dataset)
                for key, value in stage_arrays.items():
                    if key != "metadata":
                        arrays[key] = value
            recorder.check("frame.file", not missing_stages, f"{label}: missing stage files for {missing_stages}")
        else:
            recorder.check("frame.file", False,
                           f"{label}: final frames pack missing at {dataset_dir / 'frames'} "
                           "(full acceptance requires it; use --partial for an explicit intermediate stage merge)")
        if not arrays:
            continue
        problems = []
        source = entry.get("source_index")
        if not isinstance(source, int) or source < 0:
            problems.append(f"source_index={source!r}")
        if (entry.get("height"), entry.get("width")) != (H, W):
            problems.append(f"manifest size {(entry.get('height'), entry.get('width'))} != {(H, W)}")
        if pack is not None:
            pack_source = pack.get("frame_source_index")
            if pack_source is None or int(pack_source.reshape(-1)[0]) != entry.get("source_index"):
                problems.append("frames pack frame_source_index mismatch")
            pack_h, pack_w = pack.get("frame_rgb_height"), pack.get("frame_rgb_width")
            if pack_h is None or pack_w is None or (int(pack_h.reshape(-1)[0]), int(pack_w.reshape(-1)[0])) != (H, W):
                problems.append("frames pack rgb size mismatch")
        rgb_relative = entry.get("rgb_path")
        if isinstance(rgb_relative, str) and rgb_relative:
            rgb_path = Path(rgb_relative)
            if not rgb_path.is_absolute():
                rgb_path = dataset_dir / rgb_path
            if rgb_path.is_file():
                import hashlib
                actual = hashlib.sha256(rgb_path.read_bytes()).hexdigest()
                if isinstance(entry.get("rgb_sha256"), str) and actual != entry["rgb_sha256"]:
                    problems.append("manifest rgb_sha256 != materialized rgb file sha256")
                try:
                    from PIL import Image
                    with Image.open(rgb_path) as image:
                        if (image.width, image.height) != (W, H):
                            problems.append(f"materialized rgb size {(image.width, image.height)} != {(W, H)}")
                except ImportError:
                    pass
            else:
                problems.append(f"materialized rgb file missing: {rgb_relative!r}")
        recorder.check("frame.source_meta", not problems, f"{label}: {'; '.join(problems)}")

        arrow = _check_arrow(recorder, label, arrays, H, W)
        geometry = _check_geometry(recorder, label, arrays, H, W)
        _check_prep(recorder, label, arrays, H, W)
        lingbot = _check_lingbot(recorder, label, arrays, H, W, arrow.get("valid"))
        ndr = _check_ndr(recorder, label, arrays, H, W, lingbot)
        sceneflow = _check_sceneflow(recorder, label, arrays, H, W, ndr)
        sr = {}
        pair_row = sceneflow.get("pair")
        sr_frame = {}
        if isinstance(index, int):
            sr_stage = _open_npz(dataset_dir / "stages" / "sr" / f"{index:06d}.npz")
            if sr_stage is not None:
                _check_frame_metadata(recorder, f"{label}/sr", sr_stage, dataset)
                sr_frame = {key: value for key, value in sr_stage.items() if key != "metadata"}
        if pair_row is not None and pair_row != (-1, -1):
            merged = dict(arrays)
            merged.update(sr_frame)
            if "sr_flow" not in merged:
                merged = _sr_from_alt_storage(label, merged, sr_alt, H, W, pair_row)
            if "sr_flow" in merged:
                sr = _check_sr(recorder, label, merged, H, W, sceneflow, ndr)
            else:
                recorder.check("frame.file", False,
                               f"{label}: sr stage data missing (frame file and stages/sr.npz both unusable)")
        elif sr_frame:
            _check_sr_empty(recorder, label, sr_frame)
        temporal = _check_temporal(recorder, label, arrays, H, W)
        final = _check_final(recorder, label, arrays, H, W)

        if arrow.get("depth") is not None and lingbot.get("depth") is not None and lingbot.get("mask") is not None:
            if arrays.get("lingbot_depth_in") is not None:
                _compare(recorder, "lingbot.depth_in_mismatch", f"{label}: lingbot_depth_in vs arrow_depth",
                         arrays["lingbot_depth_in"], arrow["depth"], arrow["valid"], arrow["valid"])
        if final.get("depth") is not None and temporal.get("denom") is not None:
            if arrays.get("final_depth") is not None and arrays.get("temporal_depth") is not None:
                _compare(recorder, "final.depth_mismatch", f"{label}: final_depth vs temporal_depth",
                         arrays["final_depth"], arrays["temporal_depth"],
                         arrays["temporal_valid"].astype(bool), arrays["temporal_valid"].astype(bool))
        records.append({
            "source_index": entry.get("source_index"),
            "clip": arrow.get("clip"),
            "scale": arrow.get("scale"),
            "pose_valid": geometry.get("pose_valid"),
            "c2r": geometry.get("camera_to_reference"),
            "c2rec": geometry.get("camera_to_reconstruct"),
            "c2m": geometry.get("camera_to_metric"),
            "pair": sceneflow.get("pair"),
            "sr_pair": int(sr["pair"].reshape(-1)[0]) if isinstance(sr.get("pair"), np.ndarray) and sr["pair"].size >= 1 else None,
            "denom": temporal.get("denom", 0),
        })

    _aggregate_pose_to_reconstruct(recorder, records)
    _aggregate_clip_scale(recorder, records)
    _aggregate_geometry_invariants(recorder, records)
    _aggregate_sceneflow_pairs(recorder, dataset_dir, records, F)
    denom_sum = sum(int(record.get("denom") or 0) for record in records)
    _aggregate_temporal_manifest(recorder, manifest.get("temporal"), denom_sum, F)
    sr_pair_count = len([record for record in records if record.get("sr_pair") is not None and record["sr_pair"] >= 0])
    _aggregate_sr_manifest(recorder, manifest.get("sr"), sr_pair_count, manifest.get("temporal"))

    if check_render:
        _check_visuals(recorder, dataset_dir, manifest.get("visuals"), F, H, W, ffprobe_bin, rerun_bin, rrd_min_bytes)

    return _finalize(report, recorder)


def _finalize(report: dict, recorder: Recorder) -> dict:
    report["failures"] = recorder.failures
    report["ok"] = not recorder.failures
    report["checked_checks"] = len(recorder.checks)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("grass", "railway", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, default=None)
    parser.add_argument("--check-render", action="store_true", help="also verify MP4/PNG/RRD render outputs")
    parser.add_argument("--partial", action="store_true",
                        help="explicit intermediate audit: merge stage dirs when the final frames pack is missing")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--rerun-bin", default=None)
    parser.add_argument("--grass-count", type=int, default=EXPECTED_COUNTS["grass"])
    parser.add_argument("--railway-count", type=int, default=EXPECTED_COUNTS["railway"])
    parser.add_argument("--json-report", type=Path, default=None)
    args = parser.parse_args(argv)

    counts = {"grass": args.grass_count, "railway": args.railway_count}
    datasets = ("grass", "railway") if args.dataset == "all" else (args.dataset,)
    reports = []
    for dataset in datasets:
        dataset_dir = args.output_dir / dataset
        report = verify_dataset(
            dataset_dir,
            dataset=dataset,
            expected_count=counts[dataset],
            evidence_dir=args.evidence_dir,
            check_render=args.check_render,
            ffprobe_bin=args.ffprobe_bin,
            rerun_bin=args.rerun_bin,
            allow_stage_merge=args.partial,
        )
        reports.append(report)
        status = "PASS" if report["ok"] else "FAIL"
        print(f"[{status}] {dataset}: {report['checked_checks']} checks, {len(report['failures'])} failures ({dataset_dir})")
        for failure in report["failures"][:20]:
            print(f"  - {failure['id']}: {failure['detail']}")
        if len(report["failures"]) > 20:
            print(f"  ... {len(report['failures']) - 20} more failures")
    overall = {
        "verifier": "scripts/verify_depth_detail.py",
        "schema": "pipeline verifier_schema.json v2",
        "policy": {"stages": list(REQUIRED_STAGES), "units": UNITS, "metric_calibrated": False},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ok": all(report["ok"] for report in reports),
        "datasets": reports,
    }
    if args.json_report is not None:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(overall, indent=2) + "\n")
        print(f"report written: {args.json_report}")
    return 0 if overall["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
