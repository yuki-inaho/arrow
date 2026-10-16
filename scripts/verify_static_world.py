#!/usr/bin/env python3
"""Independent verifier for the SR static-world consumer outputs.

Read-only. This module never imports ``scripts/build_static_world.py`` nor
``arrow.static_world``; every numeric relation is re-derived from the workdoc
contract with homogeneous matrices:

    P  = depth * inv(K0) [u, v, 1]                (camera0)
    T  = inv(M1) @ M0                            (camera1 <- camera0)
    Qr = T P,  Qt = T (P + S)
    rigid_flow = project(Qr) - [u, v]
    nonrigid_flow = project(Qt) - project(Qr)
    total_flow = project(Qt) - [u, v]
    rigid_3d = Qr - P, nonrigid_3d = R_T S, total_3d = Qt - P
    world_nonrigid = R_M0 S

Canonical births follow ``points = source_birth_world - birth_correction`` where the
correction is the producer's 32-offset donor estimate (radii 1, 2, 4, 8 x 8 directions,
relative depth <= 0.05, RGB L2 <= 80, at least 4 donors, component median of
``live_world - canonical_anchor``; zero support implies zero correction). The saved
correspondence cannot mark same-frame deduplicated pixels, so donor counts are reproduced
with ``birth_frame[label] < frame`` and kept as a superset diagnostic while the correction
value itself is checked against the donor envelope and the birth equation.

The verifier also recomputes the canonical model against the native birth
frames, checks correspondence validity/ranges and observation counts, checks
virtual RGB-D frames (depth = camera-z of the winning model point, colors,
reference_support, splat-disc coverage), hashes all declared input pins and
output artifacts, and can optionally probe the media with ffprobe / rerun.

Tolerances: pixel-like values atol 3e-4 rtol 3e-6; world-like atol 1e-5 rtol 1e-5.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

PIXEL_ATOL = 3e-4
PIXEL_RTOL = 3e-6
WORLD_ATOL = 1e-5
WORLD_RTOL = 1e-5

ARTIFACT_SUFFIXES = {".mp4", ".png", ".ply", ".rrd", ".rbl", ".npz"}
PIXEL_KEYS = {"rigid_flow", "nonrigid_flow", "total_flow", "target_uv"}
EIGHT_DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
BIRTH_OFFSET_RADII = (1, 2, 4, 8)
BIRTH_MINIMUM_DONORS = 4
BIRTH_MAX_DEPTH_RATIO = 0.05
BIRTH_MAX_RGB_DISTANCE = 80.0
BIRTH_DONOR_CHUNK = 50000


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def resolve_path(directory, value):
    path = Path(str(value))
    if not path.is_absolute():
        path = Path(directory) / path
    return path


def is_regular_nonempty(path):
    path = Path(path)
    return path.is_file() and not path.is_symlink() and path.stat().st_size > 0


class Issues:
    """Bounded failure/warning collector so per-pixel problems never spam."""

    def __init__(self, failure_cap=100, warning_cap=200):
        self.failures = []
        self.warnings = []
        self._failure_cap = failure_cap
        self._warning_cap = warning_cap

    def fail(self, message):
        if len(self.failures) < self._failure_cap:
            self.failures.append(str(message))

    def warn(self, message):
        if len(self.warnings) < self._warning_cap:
            self.warnings.append(str(message))


def rigid_matrix(matrix, name, issues):
    M = np.asarray(matrix, dtype=np.float64)
    if M.shape != (4, 4) or not np.all(np.isfinite(M)):
        issues.fail(f"{name}: must be a finite 4x4 matrix")
        return None
    if not np.allclose(M[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
        issues.fail(f"{name}: last row must be [0, 0, 0, 1]")
        return None
    rotation = M[:3, :3]
    orth = float(np.max(np.abs(rotation @ rotation.T - np.eye(3))))
    det = float(np.linalg.det(rotation))
    if orth > 1e-4 or abs(det - 1.0) > 1e-4:
        issues.fail(f"{name}: not rigid (orth={orth:.3g}, det={det:.9g})")
        return None
    return M


def intrinsics(matrix, name, issues):
    K = np.asarray(matrix, dtype=np.float64)
    if K.shape != (3, 3) or not np.all(np.isfinite(K)):
        issues.fail(f"{name}: must be a finite 3x3 matrix")
        return None
    if K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
        issues.fail(f"{name}: focal lengths must be positive")
        return None
    if abs(float(np.linalg.det(K))) < 1e-12:
        issues.fail(f"{name}: must be non-singular")
        return None
    return K


def pixel_grid(h, w):
    """Pixel-centre coordinates (u=col, v=row) as (H, W, 2) float64."""
    rows, cols = np.mgrid[0:h, 0:w]
    return np.stack([cols, rows], axis=-1).astype(np.float64)


def pixel_rays(h, w, K, z):
    """Homogeneous pixel-centre rays scaled by per-pixel z: (H, W, 3)."""
    grid = pixel_grid(h, w)
    hom = np.concatenate([grid, np.ones((h, w, 1))], axis=-1)
    return (hom @ np.linalg.inv(K).T) * z.astype(np.float64)[..., None]


def project(K, points):
    z = points[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (K[0, 0] * points[..., 0] + K[0, 1] * points[..., 1] + K[0, 2] * z) / z
        v = (K[1, 0] * points[..., 0] + K[1, 1] * points[..., 1] + K[1, 2] * z) / z
    return np.stack([u, v], axis=-1), z


def compare(stored, expected, mask, atol, rtol, issues, errors, key):
    values = np.asarray(stored, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if np.any(mask):
        inside = values[mask]
        if not np.isfinite(inside).all():
            issues.fail(f"{key}: non-finite value inside the valid mask")
        diff = np.abs(inside - expected[mask])
        errors[key] = max(errors.get(key, 0.0), float(diff.max()) if diff.size else 0.0)
        if np.any(diff > atol + rtol * np.abs(expected[mask])):
            issues.fail(
                f"{key}: {int(np.count_nonzero(diff > atol + rtol * np.abs(expected[mask])))} "
                f"elements exceed tolerance (max |diff|={float(diff.max()):.3g})"
            )
    if np.any(~mask):
        outside_bad = int(np.count_nonzero(~np.isnan(values[~mask])))
        if outside_bad:
            issues.fail(f"{key}: {outside_bad} values outside the valid mask are not NaN")


def metadata_json(value):
    return json.loads(str(value))


def verify_dataset(input_root, output_root, dataset, *, check_media=False, rerun_bin=None):
    """Verify one dataset tree and return a JSON-safe report dict."""
    issues = Issues()
    report = {
        "dataset": dataset,
        "ok": False,
        "failures": issues.failures,
        "warnings": issues.warnings,
        "counters": {},
        "max_errors": {},
        "media": {},
    }
    errors = report["max_errors"]
    counters = report["counters"]
    input_dir = Path(input_root) / dataset
    out_dir = Path(output_root) / dataset / "static_world"

    records = []
    try:
        records = _verify_inputs(input_dir, issues, counters)
    except Exception as exc:  # noqa: BLE001 - report instead of crashing
        issues.fail(f"input membership check crashed: {exc!r}")

    manifest_path = out_dir / "manifest.json"
    if not is_regular_nonempty(manifest_path):
        issues.fail(f"missing manifest: {manifest_path}")
        report["ok"] = not issues.failures
        return report
    manifest = json.loads(manifest_path.read_text())

    try:
        _verify_declarations(input_dir, out_dir, dataset, manifest, records, issues)
    except Exception as exc:  # noqa: BLE001
        issues.fail(f"declaration check crashed: {exc!r}")

    frame_rows = manifest.get("frames", [])
    count = int(manifest.get("frame_count", 0))
    counters["frame_count"] = count
    counters["pair_count"] = int(manifest.get("pair_count", 0))

    model = None
    try:
        model = _verify_model(out_dir, manifest, issues, counters, errors)
    except Exception as exc:  # noqa: BLE001
        issues.fail(f"model check crashed: {exc!r}")

    try:
        _verify_output_membership(out_dir, count, issues)
    except Exception as exc:  # noqa: BLE001
        issues.fail(f"output membership check crashed: {exc!r}")

    if model is not None:
        try:
            _verify_frames(
                input_dir, out_dir, manifest, frame_rows, records, model, issues, counters, errors
            )
        except Exception as exc:  # noqa: BLE001
            issues.fail(f"frame checks crashed: {exc!r}")
        try:
            _verify_motion_pairs(
                input_dir, out_dir, manifest, frame_rows, model, issues, counters, errors
            )
        except Exception as exc:  # noqa: BLE001
            issues.fail(f"motion pair checks crashed: {exc!r}")
        try:
            if sha256_file(out_dir / "model.npz") != manifest.get("model_sha256"):
                issues.fail("model.npz changed during verification")
        except Exception as exc:  # noqa: BLE001
            issues.fail(f"model immutability check crashed: {exc!r}")

    if check_media:
        try:
            _verify_media(out_dir, dataset, manifest, issues, report, rerun_bin)
        except Exception as exc:  # noqa: BLE001
            issues.fail(f"media check crashed: {exc!r}")

    report["ok"] = not issues.failures
    return report


def _verify_inputs(input_dir, issues, counters):
    frames_path = input_dir / "frames.json"
    if not is_regular_nonempty(frames_path):
        issues.fail(f"missing input frames.json: {frames_path}")
        return []
    frames = json.loads(frames_path.read_text())
    records = frames.get("frames", [])
    count = int(frames.get("frame_count", -1))
    counters["input_frames"] = count
    if count != len(records) or count < 1:
        issues.fail(f"frames.json frame_count={count} does not match {len(records)} records")
        return records
    indices = [int(record.get("frame_index", -1)) for record in records]
    if indices != list(range(count)):
        issues.fail("input frame_index values are not contiguous from zero")
    source_indices = [int(record.get("source_index", -1)) for record in records]
    if len(set(source_indices)) != len(source_indices):
        issues.fail("input source_index values are not member-unique")
    size = tuple(int(v) for v in frames.get("original_size", ()))
    if len(size) != 2 or size[0] < 1 or size[1] < 1:
        issues.fail(f"invalid original_size: {size}")
    for i in range(count):
        for stage in ("final", "sr", "geometry"):
            path = input_dir / "stages" / stage / f"{i:06d}.npz"
            if not is_regular_nonempty(path):
                issues.fail(f"missing input stage file: {path}")
        rgb = resolve_path(input_dir, records[i].get("rgb_path", ""))
        if not is_regular_nonempty(rgb):
            issues.fail(f"missing input RGB: {rgb}")
    return records


def _verify_declarations(input_dir, out_dir, dataset, manifest, records, issues):
    if int(manifest.get("frame_count", 0)) != int(manifest.get("pair_count", -1)) + 1:
        issues.fail("manifest pair_count must be frame_count - 1")
    if manifest.get("dataset") != dataset:
        issues.fail(f"manifest dataset {manifest.get('dataset')!r} != {dataset!r}")
    if manifest.get("units") != "arrow_provisional":
        issues.fail("manifest units must be arrow_provisional")
    if manifest.get("metric_calibrated") is not False:
        issues.fail("manifest metric_calibrated must be false")
    frames_path = input_dir / "frames.json"
    if is_regular_nonempty(frames_path):
        input_size = tuple(int(v) for v in json.loads(frames_path.read_text())["original_size"])
        if tuple(int(v) for v in manifest.get("original_size", ())) != input_size:
            issues.fail("manifest original_size differs from input frames.json")

    declared = {entry["path"]: entry for entry in manifest.get("artifacts", [])}
    actual = {
        path.name for path in out_dir.iterdir() if path.is_file() and path.suffix in ARTIFACT_SUFFIXES
    }
    if set(declared) != actual:
        missing = sorted(set(actual) - set(declared))
        extra = sorted(set(declared) - set(actual))
        if missing:
            issues.fail(f"artifacts not declared in manifest: {missing}")
        if extra:
            issues.fail(f"declared artifacts missing from disk: {extra}")
    for name, entry in declared.items():
        path = out_dir / name
        if not is_regular_nonempty(path):
            issues.fail(f"artifact is not a regular nonempty file: {name}")
            continue
        if path.stat().st_size != int(entry.get("bytes", -1)):
            issues.fail(f"artifact byte size mismatch: {name}")
        if sha256_file(path) != entry.get("sha256"):
            issues.fail(f"artifact sha256 mismatch: {name}")

    rgb_pins = {}
    for record in records:
        rgb = resolve_path(input_dir, record.get("rgb_path", ""))
        if "rgb_sha256" in record:
            rgb_pins[str(rgb)] = str(record["rgb_sha256"])
    for pin in manifest.get("input_pins", []):
        path = Path(pin["path"])
        if not is_regular_nonempty(path):
            issues.fail(f"input pin is not a regular nonempty file: {path}")
            continue
        if path.stat().st_size != int(pin.get("bytes", -1)):
            issues.fail(f"input pin byte size mismatch: {path}")
        digest = sha256_file(path)
        if digest != pin.get("sha256"):
            issues.fail(f"input pin sha256 mismatch: {path}")
        declared_rgb = rgb_pins.get(str(path))
        if declared_rgb is not None and digest != declared_rgb:
            issues.fail(f"input RGB sha256 differs from frames.json: {path}")

    for pin in manifest.get("source_pins", []):
        path = Path(pin.get("path", ""))
        if not path.is_file():
            continue  # historical code path may no longer exist: not a data failure
        if sha256_file(path) != pin.get("sha256"):
            issues.warn(
                f"historical source pin differs from the current file (diagnostic only): {path}"
            )


def _verify_output_membership(out_dir, count, issues):
    for i in range(count - 1):
        path = out_dir / "motion" / f"{i:06d}.npz"
        if not is_regular_nonempty(path):
            issues.fail(f"missing motion pair file: {path}")
    for folder in ("correspondence", "rgbd"):
        expected = [f"{i:06d}.npz" for i in range(count)]
        directory = out_dir / folder
        names = sorted(p.name for p in directory.glob("*.npz")) if directory.is_dir() else []
        if names != expected:
            issues.fail(f"{folder}: frame files are not the contiguous complete set")


def _verify_model(out_dir, manifest, issues, counters, errors):
    path = out_dir / "model.npz"
    if not is_regular_nonempty(path):
        issues.fail(f"missing canonical model: {path}")
        return None
    model = load_npz(path)
    required = {
        "points",
        "colors",
        "birth_frame",
        "birth_pixel",
        "observation_count",
        "last_seen",
        "birth_correction",
        "correction_support",
        "units",
        "metric_calibrated",
        "voxel_size",
        "model_stride",
        "canonical_policy",
    }
    missing = sorted(required - set(model))
    if missing:
        issues.fail(f"model.npz is missing keys: {missing}")
        return None
    points = np.asarray(model["points"])
    colors = np.asarray(model["colors"])
    birth = np.asarray(model["birth_frame"])
    birth_pixel = np.asarray(model["birth_pixel"])
    observations = np.asarray(model["observation_count"])
    last_seen = np.asarray(model["last_seen"])
    n = int(points.shape[0])
    counters["model_points"] = n
    if points.ndim != 2 or points.shape[1] != 3 or n < 1:
        issues.fail(f"model points must be (N, 3) with N > 0, got {points.shape}")
        return None
    if not np.issubdtype(points.dtype, np.floating):
        issues.fail(f"model points must be floating, got {points.dtype}")
    if not np.all(np.isfinite(points)):
        issues.fail("model points must be finite")
    if colors.shape != points.shape or colors.dtype != np.uint8:
        issues.fail(f"model colors must be uint8 (N, 3), got {colors.dtype} {colors.shape}")
    for name, values in (
        ("birth_frame", birth),
        ("birth_pixel", birth_pixel),
        ("observation_count", observations),
        ("last_seen", last_seen),
    ):
        if values.shape != (n,):
            issues.fail(f"model {name} must have shape (N,), got {values.shape}")
            return None
        if not np.issubdtype(values.dtype, np.integer):
            issues.fail(f"model {name} must be integer, got {values.dtype}")
    count = int(manifest.get("frame_count", 0))
    size = tuple(int(v) for v in manifest.get("original_size", ()))
    if np.any((birth < 0) | (birth >= count)):
        issues.fail("model birth_frame outside [0, frame_count)")
    if np.any((birth_pixel < 0) | (birth_pixel >= size[0] * size[1])):
        issues.fail("model birth_pixel outside the native frame size")
    if np.any((observations < 1) | (last_seen < birth) | (last_seen >= count)):
        issues.fail("model observation_count/last_seen inconsistent with birth_frame")
    if str(model["units"]) != "arrow_provisional":
        issues.fail("model units must be arrow_provisional")
    if bool(model["metric_calibrated"]) is not False:
        issues.fail("model metric_calibrated must be false")
    if float(model["voxel_size"]) <= 0.0:
        issues.fail("model voxel_size must be positive")
    if int(model["model_stride"]) < 1:
        issues.fail("model model_stride must be >= 1")
    correction = np.asarray(model["birth_correction"])
    support = np.asarray(model["correction_support"])
    if correction.shape != (n, 3) or not np.issubdtype(correction.dtype, np.floating):
        issues.fail(f"model birth_correction must be floating (N, 3), got {correction.dtype} {correction.shape}")
        correction = np.zeros((n, 3), dtype=np.float32)
    if not np.all(np.isfinite(correction)):
        issues.fail("model birth_correction must be finite")
    if support.shape != (n,) or support.dtype != np.uint8:
        issues.fail(f"model correction_support must be uint8 (N,), got {support.dtype} {support.shape}")
        support = np.zeros(n, dtype=np.uint8)
    if np.any((support != 0) & ((support < 4) | (support > 32))):
        issues.fail("model correction_support must be 0 or within [4, 32]")
    unsupported = support == 0
    if unsupported.any() and not np.all(correction[unsupported] == 0.0):
        issues.fail("zero correction_support must have zero birth_correction")
    frame0 = birth == 0
    if np.any(correction[frame0] != 0.0) or np.any(support[frame0] != 0):
        issues.fail("frame 0 anchors must have zero birth_correction and support")
    model["birth_correction"] = correction
    model["correction_support"] = support
    counters["corrected_anchors"] = int((support > 0).sum())
    if int(manifest.get("motion_corrected_anchor_count", -1)) != counters["corrected_anchors"]:
        issues.fail("manifest motion_corrected_anchor_count does not match correction_support")
    counters["frame0_anchors"] = int((birth == 0).sum())
    counters["later_anchors"] = int((birth > 0).sum())
    if int(manifest.get("model_point_count", -1)) != n:
        issues.fail("manifest model_point_count does not match model.npz")
    if int(manifest.get("frame0_anchor_count", -1)) != counters["frame0_anchors"]:
        issues.fail("manifest frame0_anchor_count does not match model.npz")
    if int(manifest.get("later_anchor_count", -1)) != counters["later_anchors"]:
        issues.fail("manifest later_anchor_count does not match model.npz")
    if sha256_file(path) != manifest.get("model_sha256"):
        issues.fail("model.npz sha256 does not match manifest model_sha256")
    errors["model_points"] = float(n)
    return model


def _verify_frames(input_dir, out_dir, manifest, frame_rows, records, model, issues, counters, errors):
    points = np.asarray(model["points"], dtype=np.float64)
    colors = np.asarray(model["colors"])
    birth = np.asarray(model["birth_frame"])
    birth_pixel = np.asarray(model["birth_pixel"])
    correction = np.asarray(model["birth_correction"], dtype=np.float64)
    support = np.asarray(model["correction_support"])
    n = int(points.shape[0])
    size = tuple(int(v) for v in manifest.get("original_size", ()))
    h, w = size
    count = int(manifest.get("frame_count", 0))
    stride = int(model["model_stride"])
    splat_radius = int(manifest.get("splat_radius", 0))
    model_sha = manifest.get("model_sha256")
    coverage_rows = manifest.get("coverage", [])
    if len(frame_rows) != count:
        issues.fail("manifest frames[] length differs from frame_count")
    if len(coverage_rows) != count:
        issues.fail("manifest coverage[] length differs from frame_count")

    observations = np.zeros(n, dtype=np.int64)
    last_seen = np.full(n, -1, dtype=np.int64)
    valid_counts = []
    camera_outputs = {}
    for i in range(count):
        try:
            final = load_npz(input_dir / "stages" / "final" / f"{i:06d}.npz")
            geometry = load_npz(input_dir / "stages" / "geometry" / f"{i:06d}.npz")
            depth_source = np.asarray(final["final_depth"])
            valid_source = (
                np.asarray(final["final_valid"]).astype(bool)
                & np.isfinite(depth_source)
                & (depth_source > 0.0)
            )
            if depth_source.shape != size:
                issues.fail(f"frame {i}: source depth shape {depth_source.shape} != {size}")
                continue
            K = intrinsics(geometry["arrow_K"], f"frame {i} arrow_K", issues)
            M = rigid_matrix(geometry["arrow_camera_to_metric"], f"frame {i} c2w", issues)
            if K is None or M is None:
                continue
            if not bool(geometry["arrow_K_valid"]) or not bool(geometry["arrow_pose_valid"]):
                issues.fail(f"frame {i}: source calibration is not marked valid")
            row = frame_rows[i] if i < len(frame_rows) else {}
            if int(row.get("frame_index", -1)) != i:
                issues.fail(f"frame {i}: manifest frame_index mismatch")
            if i < len(records) and int(row.get("source_index", -1)) != int(records[i].get("source_index", -2)):
                issues.fail(f"frame {i}: manifest source_index differs from input records")

            correspondence = load_npz(out_dir / "correspondence" / f"{i:06d}.npz")
            anchor_id = np.asarray(correspondence["anchor_id"])
            corr_valid = np.asarray(correspondence["valid"]).astype(bool)
            if anchor_id.shape != size or not np.issubdtype(anchor_id.dtype, np.integer):
                issues.fail(f"frame {i}: correspondence anchor_id must be an integer (H, W)")
            elif not np.array_equal(corr_valid, anchor_id >= 0):
                issues.fail(f"frame {i}: correspondence valid must equal anchor_id >= 0")
            else:
                if np.any((anchor_id < -1) | (anchor_id >= n)):
                    issues.fail(f"frame {i}: correspondence anchor_id outside [-1, model count)")
                if np.any(anchor_id[~corr_valid] != -1):
                    issues.fail(f"frame {i}: unsupported correspondence pixels must be -1")
                if np.any(corr_valid & ~valid_source):
                    issues.fail(f"frame {i}: anchor at a pixel without valid source depth")
                ids = anchor_id[corr_valid].astype(np.int64)
                if ids.size and ids.min() >= 0 and ids.max() < n:
                    unique_ids = np.unique(ids)
                    observations[unique_ids] += 1
                    last_seen[unique_ids] = i
            if int(correspondence["frame_index"]) != i:
                issues.fail(f"frame {i}: correspondence frame_index mismatch")
            if int(correspondence["source_index"]) != int(row.get("source_index", -1)):
                issues.fail(f"frame {i}: correspondence source_index mismatch")

            # Diagnostic only: canonical anchors are intentionally different from the live
            # world for moving surfaces (no upper motion bound exists in the contract), so the
            # live-to-canonical distance is reported and never a hard failure.
            if corr_valid.any():
                rows_c, cols_c = np.nonzero(corr_valid)
                flat_c = anchor_id[corr_valid].astype(np.int64)
                if flat_c.min() >= 0 and flat_c.max() < n and np.all(valid_source[rows_c, cols_c]):
                    z_c = depth_source[rows_c, cols_c].astype(np.float64)
                    grid_c = pixel_grid(h, w)[rows_c, cols_c]
                    hom_c = np.concatenate([grid_c, np.ones((rows_c.size, 1))], axis=1)
                    world_c = ((hom_c @ np.linalg.inv(K).T) * z_c[:, None]) @ M[:3, :3].T + M[:3, 3]
                    diff_c = np.linalg.norm(world_c - points[flat_c], axis=1)
                    errors["id_world_reprojection"] = max(
                        errors.get("id_world_reprojection", 0.0), float(diff_c.max())
                    )

            _verify_counts(row, i, valid_source, corr_valid, birth, support, issues)
            frame_birth = birth == i
            need_rgb = bool(frame_birth.any()) or bool((frame_birth & (support >= 4)).any())
            rgb_image = None
            if need_rgb and i < len(records):
                rgb_path = resolve_path(input_dir, records[i].get("rgb_path", ""))
                if is_regular_nonempty(rgb_path):
                    rgb_image = np.asarray(Image.open(rgb_path).convert("RGB"))
                else:
                    issues.fail(f"frame {i}: RGB unavailable for birth verification")
            if frame_birth.any():
                _verify_birth_frame(
                    i, birth, birth_pixel, points, correction, colors, depth_source, valid_source,
                    K, M, stride, rgb_image, issues, errors,
                )
            if anchor_id.shape == size and corr_valid.shape == size and bool(
                (frame_birth & (support >= 4)).any()
            ):
                _verify_birth_donors(
                    i, birth, birth_pixel, points, correction, support, anchor_id, depth_source,
                    rgb_image, K, M, issues, counters, errors,
                )

            rgbd = load_npz(out_dir / "rgbd" / f"{i:06d}.npz")
            camera_key, frame_hashes = _verify_rgbd(
                i, row, rgbd, K, M, points, colors, birth, n, size, splat_radius, model_sha,
                issues, errors,
            )
            if camera_key in camera_outputs and camera_outputs[camera_key] != frame_hashes:
                issues.fail(f"frame {i}: identical native K/M must reproduce byte-identical RGB-D")
            camera_outputs[camera_key] = frame_hashes

            valid_counts.append(int(np.asarray(rgbd["valid"]).sum()))
            _verify_coverage_row(i, coverage_rows, rgbd, issues)
        except Exception as exc:  # noqa: BLE001
            issues.fail(f"frame {i}: check crashed {exc!r}")

    counters["rgbd_frames"] = len(valid_counts)
    if len(valid_counts) == count and len(coverage_rows) == count:
        expected_mean = float(np.mean([c / (h * w) for c in valid_counts]))
        expected_min = float(min(c / (h * w) for c in valid_counts))
        if abs(float(manifest.get("mean_coverage", -1)) - expected_mean) > 1e-9:
            issues.fail("manifest mean_coverage does not match recomputed coverage")
        if abs(float(manifest.get("minimum_coverage", -1)) - expected_min) > 1e-9:
            issues.fail("manifest minimum_coverage does not match recomputed coverage")

    stored_obs = np.asarray(model["observation_count"], dtype=np.int64)
    stored_last = np.asarray(model["last_seen"], dtype=np.int64)
    if not np.array_equal(observations, stored_obs):
        issues.fail(
            "observation_count recomputation differs for "
            f"{int(np.count_nonzero(observations != stored_obs))} anchors"
        )
    if not np.array_equal(last_seen, stored_last):
        issues.fail(
            f"last_seen recomputation differs for "
            f"{int(np.count_nonzero(last_seen != stored_last))} anchors"
        )


def _verify_counts(row, index, valid_source, corr_valid, birth, support, issues):
    if int(row.get("valid_depth_count", -1)) != int(valid_source.sum()):
        issues.fail(f"frame {index}: valid_depth_count mismatch")
    canonical = int(corr_valid.sum())
    if int(row.get("canonical_pixel_count", -1)) != canonical:
        issues.fail(f"frame {index}: canonical_pixel_count mismatch")
    new_anchors = int((np.asarray(birth) == index).sum())
    if int(row.get("new_anchor_count", -1)) != new_anchors:
        issues.fail(f"frame {index}: new_anchor_count mismatch")
    corrected = int(((np.asarray(birth) == index) & (np.asarray(support) > 0)).sum())
    if int(row.get("motion_corrected_new_anchors", -1)) != corrected:
        issues.fail(f"frame {index}: motion_corrected_new_anchors mismatch")
    if int(row.get("model_point_count", -1)) != int((np.asarray(birth) <= index).sum()):
        issues.fail(f"frame {index}: model_point_count mismatch")
    inherited = int(row.get("inherited_pixel_count", -1))
    dedup = int(row.get("deduplicated_new_pixels", -1))
    if inherited < 0 or dedup < 0 or inherited + new_anchors + dedup != canonical:
        issues.fail(
            f"frame {index}: canonical != inherited + new_anchor + deduplicated "
            f"({canonical} != {inherited} + {new_anchors} + {dedup})"
        )
    if canonical > int(valid_source.sum()):
        issues.fail(f"frame {index}: canonical pixels exceed valid source depth pixels")


def _verify_birth_frame(
    index, birth, birth_pixel, points, correction, colors, depth, valid, K, M, stride, rgb,
    issues, errors,
):
    selected = np.flatnonzero(np.asarray(birth) == index)
    if selected.size == 0:
        return
    flat = np.asarray(birth_pixel)[selected]
    rows = flat // depth.shape[1]
    cols = flat % depth.shape[1]
    if stride > 1 and (np.any(rows % stride != 0) or np.any(cols % stride != 0)):
        issues.fail(f"frame {index}: birth pixels are not on the model sampling grid")
    if np.any(~valid[rows, cols]):
        issues.fail(
            f"frame {index}: {int(np.count_nonzero(~valid[rows, cols]))} anchors born at "
            "pixels without valid source depth"
        )
    if rgb is not None:
        expected_colors = np.asarray(rgb)[rows, cols]
        if not np.array_equal(np.asarray(colors)[selected], expected_colors):
            bad = int(np.count_nonzero(np.any(np.asarray(colors)[selected] != expected_colors, axis=1)))
            issues.fail(f"frame {index}: {bad} anchors do not carry their source RGB color")
    z = np.where(valid[rows, cols], depth[rows, cols], 1.0).astype(np.float64)
    hom = np.concatenate(
        [pixel_grid(depth.shape[0], depth.shape[1])[rows, cols], np.ones((rows.size, 1))], axis=1
    )
    camera = (hom @ np.linalg.inv(K).T) * z[:, None]
    world = camera @ M[:3, :3].T + M[:3, 3]
    # Contract: points = source birth pixel world - birth_correction.
    expected = world - np.asarray(correction)[selected].astype(np.float64)
    diff = np.linalg.norm(expected - np.asarray(points)[selected].astype(np.float64), axis=1)
    errors["birth_world"] = max(errors.get("birth_world", 0.0), float(diff.max()) if diff.size else 0.0)
    bound = WORLD_ATOL + WORLD_RTOL * np.linalg.norm(expected, axis=1)
    if np.any(diff > bound):
        issues.fail(
            f"frame {index}: {int(np.count_nonzero(diff > bound))} first-observed anchors do not "
            "match the native K/M unprojection minus birth_correction "
            f"(max |diff|={float(diff.max()):.3g})"
        )


def _verify_birth_donors(
    index, birth, birth_pixel, points, correction, support, anchor_id, depth, rgb, K, M,
    issues, counters, errors,
):
    """Independently reproduce the local inverse deformation donors of new anchors.

    The saved correspondence contains newly created anchors of the same frame, so a donor
    candidate is restricted to ``birth_frame[label] < index``. Pixels that were deduplicated
    into an older anchor in the same frame are not distinguishable from transported pixels in
    the saved artifacts; those make the reproduced donor set a superset, which is kept as a
    bounded diagnostic (count and median shift) rather than failing the producer.
    """
    birth = np.asarray(birth)
    support = np.asarray(support)
    selected = np.flatnonzero((birth == index) & (support >= BIRTH_MINIMUM_DONORS))
    if selected.size == 0:
        return
    if rgb is None:
        issues.fail(f"frame {index}: RGB unavailable for donor verification")
        return
    counters["corrected_anchors_checked"] = counters.get("corrected_anchors_checked", 0) + int(selected.size)
    h, w = depth.shape
    z_source = np.where(np.isfinite(depth) & (depth > 0.0), depth, 0.0)
    grid = pixel_grid(h, w)
    hom = np.concatenate([grid, np.ones((h, w, 1))], axis=-1)
    world = ((hom @ np.linalg.inv(K).T) * z_source[..., None]) @ M[:3, :3].T + M[:3, 3]
    born_before = birth < index
    for start in range(0, selected.size, BIRTH_DONOR_CHUNK):
        block = selected[start : start + BIRTH_DONOR_CHUNK]
        rows = np.asarray(birth_pixel)[block] // w
        cols = np.asarray(birth_pixel)[block] % w
        source_z = depth[rows, cols]
        source_rgb = np.asarray(rgb)[rows, cols].astype(np.float32)
        donor_values = np.full((len(BIRTH_OFFSET_RADII) * len(EIGHT_DIRECTIONS), block.size, 3), np.nan)
        counts = np.zeros(block.size, dtype=np.int64)
        slot = 0
        for radius in BIRTH_OFFSET_RADII:
            for dy, dx in EIGHT_DIRECTIONS:
                yy = rows + dy * radius
                xx = cols + dx * radius
                inside = (yy >= 0) & (xx >= 0) & (yy < h) & (xx < w)
                yy = np.clip(yy, 0, h - 1)
                xx = np.clip(xx, 0, w - 1)
                ids = anchor_id[yy, xx].astype(np.int64)
                z = depth[yy, xx]
                with np.errstate(divide="ignore", invalid="ignore"):
                    depth_ratio = np.abs(z - source_z) / np.maximum(z, source_z)
                color_distance = np.linalg.norm(
                    np.asarray(rgb)[yy, xx].astype(np.float32) - source_rgb, axis=-1
                )
                usable = (
                    inside
                    & (ids >= 0)
                    & born_before[ids]
                    & np.isfinite(z)
                    & (depth_ratio <= BIRTH_MAX_DEPTH_RATIO)
                    & (color_distance <= BIRTH_MAX_RGB_DISTANCE)
                )
                if usable.any():
                    donor_values[slot, usable] = world[yy, xx][usable] - points[ids[usable]]
                    counts[usable] += 1
                slot += 1
        reproduced = counts >= BIRTH_MINIMUM_DONORS
        stored_support = support[block].astype(np.int64)
        stored_correction = np.asarray(correction, dtype=np.float64)[block]
        under = (stored_support >= BIRTH_MINIMUM_DONORS) & (counts < stored_support)
        if np.any(under):
            issues.fail(
                f"frame {index}: {int(under.sum())} corrected anchors reproduce fewer donors than "
                "correction_support"
            )
        superset = counts > stored_support
        if np.any(superset):
            counters["birth_donor_superset"] = counters.get("birth_donor_superset", 0) + int(superset.sum())
        checked = (stored_support >= BIRTH_MINIMUM_DONORS) & reproduced
        if np.any(checked):
            values = donor_values[:, checked, :]
            low = np.nanmin(values, axis=0)
            high = np.nanmax(values, axis=0)
            stored = stored_correction[checked]
            outside = np.any((stored < low - WORLD_ATOL) | (stored > high + WORLD_ATOL), axis=1)
            if np.any(outside):
                issues.fail(
                    f"frame {index}: {int(outside.sum())} birth corrections lie outside the "
                    "independently reproduced donor envelope"
                )
            median = np.nanmedian(values, axis=0)
            shift = np.abs(median - stored)
            errors["birth_correction_median_shift"] = max(
                errors.get("birth_correction_median_shift", 0.0),
                float(shift.max()) if shift.size else 0.0,
            )
            exact = np.all(shift <= WORLD_ATOL + WORLD_RTOL * np.abs(stored), axis=1)
            counters["birth_envelope_checked"] = counters.get("birth_envelope_checked", 0) + int(checked.sum())
            counters["birth_median_exact"] = counters.get("birth_median_exact", 0) + int(exact.sum())


def _verify_rgbd(i, row, rgbd, K, M, points, colors, birth, n, size, splat_radius, model_sha,
                 issues, errors):
    K = np.asarray(K, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    rgb = np.asarray(rgbd["rgb"])
    depth = np.asarray(rgbd["depth"])
    valid = np.asarray(rgbd["valid"]).astype(bool)
    index_map = np.asarray(rgbd["model_point_index"])
    reference = np.asarray(rgbd["reference_support"]).astype(bool)
    if rgb.shape != (size[0], size[1], 3) or rgb.dtype != np.uint8:
        issues.fail(f"rgbd {i}: rgb must be uint8 (H, W, 3)")
    if depth.shape != size or not np.issubdtype(depth.dtype, np.floating):
        issues.fail(f"rgbd {i}: depth must be floating (H, W)")
    if valid.shape != size or not np.issubdtype(index_map.dtype, np.integer):
        issues.fail(f"rgbd {i}: valid/index shapes or dtypes are wrong")
        return (("bad", i),), (("bad", i),)
    if reference.shape != size:
        issues.fail(f"rgbd {i}: reference_support must be bool (H, W)")
        return (("bad", i),), (("bad", i),)
    if not np.array_equal(np.asarray(rgbd["K"], dtype=np.float64), K):
        issues.fail(f"rgbd {i}: K differs from the source geometry")
    if not np.array_equal(np.asarray(rgbd["camera_to_world"], dtype=np.float64), M):
        issues.fail(f"rgbd {i}: camera_to_world differs from the source geometry")
    if int(rgbd["frame_index"]) != i:
        issues.fail(f"rgbd {i}: frame_index mismatch")
    if int(rgbd["source_index"]) != int(row.get("source_index", -1)):
        issues.fail(f"rgbd {i}: source_index mismatch")
    try:
        metadata = metadata_json(rgbd["metadata"])
    except Exception:  # noqa: BLE001
        issues.fail(f"rgbd {i}: metadata is not valid JSON")
        metadata = {}
    if metadata.get("units") != "arrow_provisional" or metadata.get("metric_calibrated") is not False:
        issues.fail(f"rgbd {i}: metadata must declare arrow_provisional non-metric depth")
    if metadata.get("depth_kind") != "camera_z" or metadata.get("world_is_static") is not True:
        issues.fail(f"rgbd {i}: metadata must declare camera-z static-world depth")
    if metadata.get("model_sha256") != model_sha:
        issues.fail(f"rgbd {i}: metadata model_sha256 does not match the canonical model")

    if int(np.count_nonzero(np.isnan(depth[~valid]))) != int(np.count_nonzero(~valid)):
        issues.fail(f"rgbd {i}: depth outside valid must be NaN")
    if np.any(~np.isfinite(depth[valid])) or np.any(depth[valid] <= 0.0):
        issues.fail(f"rgbd {i}: depth inside valid must be finite and positive")
    if np.any(rgb[~valid] != 0):
        issues.fail(f"rgbd {i}: rgb outside valid must be black")
    if np.any(index_map[~valid] != -1):
        issues.fail(f"rgbd {i}: point_index outside valid must be -1")
    if np.any((index_map[valid] < 0) | (index_map[valid] >= n)):
        issues.fail(f"rgbd {i}: point_index inside valid is outside the model")
        return (("bad", i),), (("bad", i),)

    rows, cols = np.nonzero(valid)
    selected = index_map[valid].astype(np.int64)
    if not np.array_equal(rgb[valid], np.asarray(colors)[selected]):
        bad = int(np.count_nonzero(np.any(rgb[valid] != np.asarray(colors)[selected], axis=1)))
        issues.fail(f"rgbd {i}: {bad} virtual pixels do not carry the model point color")
    expected_reference = np.asarray(birth)[selected] == 0
    if not np.array_equal(reference[valid], expected_reference) or bool(reference[~valid].any()):
        issues.fail(f"rgbd {i}: reference_support must equal birth_frame == 0")

    inverse = np.linalg.inv(M)
    camera = points[selected] @ inverse[:3, :3].T + inverse[:3, 3]
    stored_z = depth[valid].astype(np.float64)
    diff = np.abs(camera[:, 2] - stored_z)
    errors["rgbd_depth"] = max(errors.get("rgbd_depth", 0.0), float(diff.max()) if diff.size else 0.0)
    bound = WORLD_ATOL + WORLD_RTOL * np.abs(stored_z)
    if np.any(diff > bound):
        issues.fail(
            f"rgbd {i}: {int(np.count_nonzero(diff > bound))} depths differ from the model "
            f"point camera-z (max |diff|={float(diff.max()):.3g})"
        )
    uv, _ = project(K, camera)
    projected_rows = np.rint(uv[:, 1]).astype(np.int64)
    projected_cols = np.rint(uv[:, 0]).astype(np.int64)
    offset = (rows - projected_rows) ** 2 + (cols - projected_cols) ** 2
    outside = offset > splat_radius * splat_radius + 1e-9
    if np.any(outside):
        issues.fail(
            f"rgbd {i}: {int(outside.sum())} pixels are outside the splat disc of their "
            "projected model point"
        )

    hashes = (
        hashlib.sha256(rgb.tobytes()).hexdigest(),
        hashlib.sha256(np.ascontiguousarray(depth).tobytes()).hexdigest(),
        hashlib.sha256(valid.tobytes()).hexdigest(),
        hashlib.sha256(index_map.tobytes()).hexdigest(),
    )
    return (K.tobytes(), M.tobytes()), hashes


def _verify_coverage_row(index, coverage_rows, rgbd, issues):
    if index >= len(coverage_rows):
        return
    row = coverage_rows[index]
    valid = np.asarray(rgbd["valid"]).astype(bool)
    reference = np.asarray(rgbd["reference_support"]).astype(bool)
    h, w = valid.shape
    if int(row.get("frame_index", -1)) != index:
        issues.fail(f"coverage[{index}]: frame_index mismatch")
    if int(row.get("total_pixels", -1)) != h * w:
        issues.fail(f"coverage[{index}]: total_pixels mismatch")
    if int(row.get("valid_pixels", -1)) != int(valid.sum()):
        issues.fail(f"coverage[{index}]: valid_pixels mismatch")
    if abs(float(row.get("coverage", -1)) - float(valid.mean())) > 1e-12:
        issues.fail(f"coverage[{index}]: coverage mismatch")
    if int(row.get("reference_supported_pixels", -1)) != int(reference.sum()):
        issues.fail(f"coverage[{index}]: reference_supported_pixels mismatch")


def _verify_motion_pairs(input_dir, out_dir, manifest, frame_rows, model, issues, counters, errors):
    size = tuple(int(v) for v in manifest.get("original_size", ()))
    h, w = size
    count = int(manifest.get("frame_count", 0))
    counters["motion_pairs"] = count - 1
    for i in range(count - 1):
        try:
            motion = load_npz(out_dir / "motion" / f"{i:06d}.npz")
            for key in ("rigid_flow", "nonrigid_flow", "total_flow"):
                if np.asarray(motion[key]).shape != (h, w, 2):
                    issues.fail(f"motion {i}: {key} must have shape (H, W, 2)")
            for key in ("rigid_3d", "nonrigid_3d", "total_3d", "world_nonrigid"):
                if np.asarray(motion[key]).shape != (h, w, 3):
                    issues.fail(f"motion {i}: {key} must have shape (H, W, 3)")
            if np.asarray(motion["valid"]).shape != size:
                issues.fail(f"motion {i}: valid must have shape (H, W)")
            if np.asarray(motion["target_uv"]).shape != (h, w, 2):
                issues.fail(f"motion {i}: target_uv must have shape (H, W, 2)")
            if np.asarray(motion["target_depth"]).shape != size:
                issues.fail(f"motion {i}: target_depth must have shape (H, W)")
            if list(np.asarray(motion["frame_pair"]).ravel()) != [i, i + 1]:
                issues.fail(f"motion {i}: frame_pair must be [i, i+1]")
            row = frame_rows[i] if i < len(frame_rows) else {}

            final_i = load_npz(input_dir / "stages" / "final" / f"{i:06d}.npz")
            sr = load_npz(input_dir / "stages" / "sr" / f"{i:06d}.npz")
            geometry_i = load_npz(input_dir / "stages" / "geometry" / f"{i:06d}.npz")
            geometry_j = load_npz(input_dir / "stages" / "geometry" / f"{i + 1:06d}.npz")
            if "sr_pair_index" in sr and list(np.asarray(sr["sr_pair_index"]).ravel()) != [i, i + 1]:
                issues.fail(f"motion {i}: source sr_pair_index is not [i, i+1]")
            depth_i = np.asarray(final_i["final_depth"])
            valid_i = (
                np.asarray(final_i["final_valid"]).astype(bool)
                & np.isfinite(depth_i)
                & (depth_i > 0.0)
            )
            if depth_i.shape != size:
                issues.fail(f"motion {i}: source depth shape differs from manifest")
                continue
            S = np.asarray(sr["sr_flow"]).astype(np.float64)
            sr_valid = np.asarray(sr["sr_valid"]).astype(bool)
            if S.shape != (h, w, 3):
                issues.fail(f"motion {i}: source sr_flow must have shape (H, W, 3)")
                continue
            K0 = intrinsics(geometry_i["arrow_K"], f"motion {i} K0", issues)
            M0 = rigid_matrix(geometry_i["arrow_camera_to_metric"], f"motion {i} M0", issues)
            K1 = intrinsics(geometry_j["arrow_K"], f"motion {i+1} K1", issues)
            M1 = rigid_matrix(geometry_j["arrow_camera_to_metric"], f"motion {i+1} M1", issues)
            if K0 is None or K1 is None or M0 is None or M1 is None:
                continue

            transform = np.linalg.inv(M1) @ M0
            rotation = transform[:3, :3]
            translation = transform[:3, 3]
            source_ok = sr_valid & valid_i & np.all(np.isfinite(S), axis=-1)
            z0 = np.where(source_ok, depth_i, 0.0)
            P = pixel_rays(h, w, K0, z0)
            Sf = np.where(source_ok[..., None], S, 0.0)
            Qr = P @ rotation.T + translation
            Qt = (P + Sf) @ rotation.T + translation
            uvr, zr = project(K1, Qr)
            uvt, zt = project(K1, Qt)
            ok = (
                source_ok
                & (zr > 0.0)
                & (zt > 0.0)
                & np.isfinite(zr)
                & np.isfinite(zt)
                & np.isfinite(uvr).all(axis=-1)
                & np.isfinite(uvt).all(axis=-1)
            )
            stored_valid = np.asarray(motion["valid"]).astype(bool)
            if not np.array_equal(stored_valid, ok):
                issues.fail(
                    f"motion {i}: stored valid differs from source final/sr valid & finite & "
                    f"positive projected depth ({int((stored_valid != ok).sum())} pixels)"
                )
            mask = ok & stored_valid

            grid = pixel_grid(h, w)
            expected = {
                "rigid_flow": uvr - grid,
                "nonrigid_flow": uvt - uvr,
                "total_flow": uvt - grid,
                "rigid_3d": Qr - P,
                "nonrigid_3d": Sf @ rotation.T,
                "total_3d": Qt - P,
                "world_nonrigid": Sf @ M0[:3, :3].T,
                "target_uv": uvt,
                "target_depth": zt,
            }
            for key, values in expected.items():
                atol, rtol = (
                    (PIXEL_ATOL, PIXEL_RTOL)
                    if key in {"rigid_flow", "nonrigid_flow", "total_flow", "target_uv"}
                    else (WORLD_ATOL, WORLD_RTOL)
                )
                compare(
                    motion[key], values, mask, atol, rtol, issues, errors, f"motion{i}.{key}"
                )

            if np.any(mask):
                add2 = float(
                    np.max(
                        np.abs(
                            motion["total_flow"].astype(np.float64)[mask]
                            - motion["rigid_flow"].astype(np.float64)[mask]
                            - motion["nonrigid_flow"].astype(np.float64)[mask]
                        )
                    )
                )
                add3 = float(
                    np.max(
                        np.abs(
                            motion["total_3d"].astype(np.float64)[mask]
                            - motion["rigid_3d"].astype(np.float64)[mask]
                            - motion["nonrigid_3d"].astype(np.float64)[mask]
                        )
                    )
                )
            else:
                add2 = add3 = 0.0
            errors[f"motion{i}.additive_2d"] = add2
            errors[f"motion{i}.additive_3d"] = add3
            if add2 > PIXEL_ATOL:
                issues.fail(f"motion {i}: stored total_flow != rigid_flow + nonrigid_flow ({add2:.3g})")
            if add3 > WORLD_ATOL:
                issues.fail(f"motion {i}: stored total_3d != rigid_3d + nonrigid_3d ({add3:.3g})")
            if np.any(mask):
                median_residual = float(
                    np.median(np.linalg.norm(motion["world_nonrigid"].astype(np.float64)[mask], axis=-1))
                )
            else:
                median_residual = 0.0
            target_depth = np.asarray(motion["target_depth"], dtype=np.float64)
            if np.any(target_depth[mask] <= 0.0):
                issues.fail(f"motion {i}: target_depth must be positive inside the valid mask")
            if int(row.get("pair_valid_count", -1)) != int(stored_valid.sum()):
                issues.fail(f"motion {i}: manifest pair_valid_count mismatch")
            if abs(float(row.get("additive_2d_max_error", -1)) - add2) > PIXEL_ATOL:
                issues.fail(f"motion {i}: manifest additive_2d_max_error mismatch")
            if abs(float(row.get("additive_3d_max_error", -1)) - add3) > WORLD_ATOL:
                issues.fail(f"motion {i}: manifest additive_3d_max_error mismatch")
            if abs(float(row.get("median_world_residual", -1.0)) - median_residual) > WORLD_ATOL:
                issues.fail(f"motion {i}: manifest median_world_residual mismatch")
            if "metadata" in motion:
                try:
                    meta = metadata_json(motion["metadata"])
                    if meta.get("units_3d") != "arrow_provisional":
                        issues.fail(f"motion {i}: metadata units_3d must be arrow_provisional")
                except Exception:  # noqa: BLE001
                    issues.fail(f"motion {i}: metadata is not valid JSON")
        except Exception as exc:  # noqa: BLE001
            issues.fail(f"motion {i}: check crashed {exc!r}")

    terminal = frame_rows[count - 1] if count - 1 < len(frame_rows) else {}
    if int(terminal.get("pair_valid_count", -1)) != 0 or terminal.get("pair_status") != "terminal_no_next_frame":
        issues.fail("terminal frame must declare pair_valid_count=0 and terminal_no_next_frame")


def _verify_media(out_dir, dataset, manifest, issues, report, rerun_bin):
    count = int(manifest.get("frame_count", 0))
    h, w = (int(v) for v in manifest.get("original_size", (0, 0)))
    cell_height = max(2, round(h * 420 / w))
    panel = (420 * 3, (cell_height + 70) // 2 * 2)
    ffprobe = shutil.which("ffprobe")
    media = report["media"]
    if ffprobe is None:
        issues.fail("--check-media requires ffprobe for the MP4 frame/size checks")
    else:
        for suffix, expected in (("rgb", (w, h)), ("depth", (w, h)), ("rgbd", panel)):
            path = out_dir / f"{dataset}_static_{suffix}.mp4"
            command = [
                ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
                "-show_entries", "stream=nb_read_frames,width,height", "-of", "json", str(path),
            ]
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                issues.fail(f"ffprobe failed for {path.name}: {result.stderr.strip()[:200]}")
                continue
            stream = json.loads(result.stdout)["streams"][0]
            frames = int(stream.get("nb_read_frames", -1))
            width = int(stream.get("width", -1))
            height = int(stream.get("height", -1))
            media[path.name] = {"frames": frames, "width": width, "height": height}
            if frames != count:
                issues.fail(f"{path.name}: {frames} frames != {count}")
            if (width, height) != expected:
                issues.fail(f"{path.name}: size {(width, height)} != {expected}")

    rrd = out_dir / manifest.get("presentation_rrd", f"{dataset}_static_world.rrd")
    runner = Path(rerun_bin) if rerun_bin else Path(__file__).resolve().parents[1] / ".venv" / "bin" / "rerun"
    if not runner.exists():
        issues.fail(f"--check-media requires the rerun binary for RRD verification: {runner}")
        return
    result = subprocess.run(
        [str(runner), "rrd", "verify", str(rrd)], capture_output=True, text=True, timeout=600
    )
    media[rrd.name] = {"returncode": int(result.returncode)}
    if result.returncode != 0:
        issues.fail(f"rerun rrd verify failed: {result.stderr.strip()[:200]}")


def verify_reference_dataset(input_root, output_root, dataset, *, check_media=False, rerun_bin=None):
    """Check the strict reference subset and every rendered RGB-D independently."""
    issues = Issues()
    report = {"dataset": dataset, "kind": "reference_only", "ok": False,
              "failures": issues.failures, "warnings": issues.warnings,
              "counters": {}, "max_errors": {}, "media": {}}
    out = Path(output_root) / dataset / "reference_world"
    source = Path(input_root) / dataset
    counters, errors = report["counters"], report["max_errors"]
    try:
        manifest = json.loads((out / "manifest.json").read_text())
        records = _verify_inputs(source, issues, counters)
        count = manifest["frame_count"]
        if count <= 0 or count != len(records) or manifest["render_status"] != "complete":
            issues.fail("reference frame count/status mismatch")
        if manifest["export_kind"] != "reference_only" or manifest["later_anchor_count"] != 0:
            issues.fail("reference export policy mismatch")
        for pin in manifest["input_pins"]:
            if Path(pin["path"]).stat().st_size != pin["bytes"] or sha256_file(pin["path"]) != pin["sha256"]:
                issues.fail(f"reference input pin changed: {pin['path']}")
        parent = load_npz(manifest["parent_model"])
        if sha256_file(manifest["parent_model"]) != manifest["parent_model_sha256"]:
            issues.fail("reference parent model hash mismatch")
        model = load_npz(out / "model.npz")
        selected = np.flatnonzero(parent["birth_frame"] == 0)
        if not selected.size or not np.array_equal(model["source_anchor_id"], selected):
            issues.fail("reference model must contain every frame0 anchor exactly once")
        for key in ("points", "colors", "birth_frame", "birth_pixel", "birth_correction", "correction_support"):
            if not np.array_equal(model[key], parent[key][selected]):
                issues.fail(f"reference model subset mismatch: {key}")
        if np.any(model["birth_frame"]) or np.any(model["birth_correction"]):
            issues.fail("reference model contains later or corrected geometry")
        if sha256_file(out / "model.npz") != manifest["model_sha256"]:
            issues.fail("reference model SHA mismatch")
        n = len(model["points"])
        counters.update(frame_count=count, model_point_count=n, rgbd_frames=0)
        if manifest["model_point_count"] != n:
            issues.fail("reference model count mismatch")
        expected_files = {f"{i:06d}.npz" for i in range(count)}
        if {p.name for p in (out / "rgbd").glob("*.npz")} != expected_files:
            issues.fail("reference RGB-D membership mismatch")
        coverage = []
        for i, record in enumerate(records):
            geometry = load_npz(source / "stages/geometry" / f"{i:06d}.npz")
            rgbd = load_npz(out / "rgbd" / f"{i:06d}.npz")
            _verify_rgbd(i, record, rgbd, geometry["arrow_K"], geometry["arrow_camera_to_metric"],
                         model["points"].astype(np.float64), model["colors"], model["birth_frame"],
                         n, tuple(manifest["original_size"]), manifest["splat_radius"],
                         manifest["model_sha256"], issues, errors)
            _verify_coverage_row(i, manifest["coverage"], rgbd, issues)
            coverage.append(float(rgbd["valid"].mean()))
            counters["rgbd_frames"] += 1
        if min(coverage) <= 0 or not np.isclose(min(coverage), manifest["minimum_coverage"]):
            issues.fail("reference minimum coverage mismatch or zero coverage")
        if not np.isclose(np.mean(coverage), manifest["mean_coverage"]):
            issues.fail("reference mean coverage mismatch")
        for artifact in manifest["artifacts"]:
            path = out / artifact["path"]
            if path.stat().st_size != artifact["bytes"] or sha256_file(path) != artifact["sha256"]:
                issues.fail(f"reference artifact hash mismatch: {path}")
        if check_media:
            _verify_media(out, dataset, manifest, issues, report, rerun_bin)
    except Exception as exc:  # noqa: BLE001 - preserve a failed audit as structured evidence
        issues.fail(f"reference verification failed: {exc!r}")
    report["ok"] = not issues.failures
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results/depth_detail_oct16"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/sr_static_oct16"))
    parser.add_argument("--dataset", choices=("grass", "railway", "all"), default="all")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--check-media", action="store_true", help="probe MP4 (ffprobe) and RRD (rerun)")
    parser.add_argument("--reference-only", action="store_true", help="verify reference_world instead of the fused static_world")
    parser.add_argument("--rerun-bin", type=Path, default=None)
    args = parser.parse_args(argv)
    datasets = ("grass", "railway") if args.dataset == "all" else (args.dataset,)
    reports = {}
    ok = True
    for dataset in datasets:
        verify = verify_reference_dataset if args.reference_only else verify_dataset
        report = verify(
            args.input_dir, args.output_dir, dataset,
            check_media=args.check_media, rerun_bin=args.rerun_bin,
        )
        reports[dataset] = report
        ok = ok and bool(report["ok"])
        print(
            f"{dataset}: {'OK' if report['ok'] else 'FAIL'} "
            f"failures={len(report['failures'])} warnings={len(report['warnings'])}"
        )
        for message in report["failures"][:10]:
            print(f"  FAIL: {message}")
        for message in report["warnings"][:5]:
            print(f"  WARN: {message}")
    output = {
        "tool": "verify_static_world",
        "ok": ok,
        "verified_at": now_iso(),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "datasets": reports,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"report: {args.report}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
