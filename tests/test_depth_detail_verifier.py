"""Rejection tests for scripts/verify_depth_detail.py (audit scope, schema v2).

Fixtures build a *minimal but contract-complete* dataset tree under the
pipeline verifier_schema.json v2 layout (per-frame NPZ with stage-prefixed
keys, manifest, signatures, attempt receipts, resolved stage configs) and each
test mutates exactly one property, asserting the verifier rejects it with a
specific failure id. Acceptance tests assert the untouched fixture passes and
that real-data-only diagnostics (SR median ratio) never become fatal.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = ROOT / "scripts" / "verify_depth_detail.py"

F = 4
C = 2  # clip 0: frames 0-1, clip 1: frames 2-3
H = 8
W = 10
LOW_H = 4
LOW_W = 5
CLIP_OF = (0, 0, 1, 1)
PAIR_OF = ((0, 1), (-1, -1), (2, 3), (-1, -1))
ACCEPTED_FWD = (True, False, True, False)
ACCEPTED_BWD = (False, True, False, True)
LOW_VECTOR = (0.01, 0.02, 0.0)


def _temporal_stats(index: int) -> list[float]:
    """Producer layout: [fwd_before_med, fwd_after_med, fwd_before_n, fwd_after_n,
    bwd_before_med, bwd_after_med, bwd_before_n, bwd_after_n]; absent direction = NaN block."""
    fwd = [0.05, 0.04, float(H * W), float(H * W)] if ACCEPTED_FWD[index] else [float("nan")] * 4
    bwd = [0.05, 0.04, float(H * W), float(H * W)] if ACCEPTED_BWD[index] else [float("nan")] * 4
    return fwd + bwd


def _load_verifier():
    if not VERIFIER_PATH.is_file():
        raise FileNotFoundError(f"verifier not implemented yet: {VERIFIER_PATH}")
    spec = importlib.util.spec_from_file_location("verify_depth_detail", VERIFIER_PATH)
    assert spec is not None, "spec_from_file_location returned None"
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None, "module spec has no loader"
    spec.loader.exec_module(module)
    return module


def _read_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _write_npz(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _rotation_y(degrees: float) -> np.ndarray:
    radians = np.radians(degrees)
    c, s = np.cos(radians), np.sin(radians)
    matrix = np.eye(4)
    matrix[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return matrix


def _camera_to_reference(index: int) -> np.ndarray:
    if index == 0:
        return np.eye(4)
    if index == 1:
        matrix = _rotation_y(8.0)
        matrix[:3, 3] = [0.05, 0.0, 0.1]
        return matrix
    if index == 2:
        matrix = _rotation_y(10.0)
        matrix[:3, 3] = [0.1, 0.2, 0.3]
        return matrix
    delta = _rotation_y(5.0)
    delta[:3, 3] = [0.02, 0.03, 0.04]
    return _camera_to_reference(2) @ delta


def _camera_to_reconstruct(index: int) -> np.ndarray:
    first = 0 if CLIP_OF[index] == 0 else 2
    return np.linalg.inv(_camera_to_reference(first)) @ _camera_to_reference(index)


def _camera_to_metric(index: int) -> np.ndarray:
    """metric world = rigid rotation, translation scaled once by arrow_scale."""
    scale = 2.0 if CLIP_OF[index] == 0 else 3.0
    matrix = _camera_to_reconstruct(index).copy()
    matrix[:3, 3] = matrix[:3, 3] * scale
    return matrix


def _metadata(stages: list[str]) -> str:
    return json.dumps(
        {
            "format_version": 1,
            "stage": "final",
            "dataset": "grass",
            "model": "test-model",
            "params": {"residual_stats_columns": list(_load_verifier().RESIDUAL_COLUMNS)},
            "input_sha256": "abc",
            "argv": ["run_depth_detail.py"],
            "status": "ok",
            "created_at": "2026-10-16T03:00:00Z",
            "units": "arrow_provisional",
            "metric_calibrated": False,
            "stages": stages,
            "calibration": {"single_scalar": 1.0, "applied": False, "targets": "depth, camera translation, 3D scene flow"},
        }
    )


def _frame_arrays(index: int, stages: list[str]) -> dict:
    depth = np.full((H, W), 2.0 + index * 0.1, dtype=np.float32)
    ones = np.ones((H, W), dtype=bool)
    flow_vector = np.array(LOW_VECTOR, dtype=np.float64)
    sr_flow = np.broadcast_to(flow_vector, (H, W, 3)).astype(np.float32).copy()
    low_flow = np.broadcast_to(flow_vector, (LOW_H, LOW_W, 3)).astype(np.float32).copy()
    low_valid = np.ones((LOW_H, LOW_W), dtype=bool)
    low_flow[0, 0] = np.float32(123.0)  # non-constant corner keeps the fixture general
    arrays = {
        "metadata": np.array(_metadata(stages)),
        # arrow
        "arrow_depth": depth.copy(),
        "arrow_valid": ones.copy(),
        "arrow_support_count": np.int64(H * W),
        "arrow_scale": np.float64(2.0 if CLIP_OF[index] == 0 else 3.0),
        "arrow_clip_index": np.int32(CLIP_OF[index]),
        # geometry
        "arrow_K": np.array([[0.8 * W, 0.0, W / 2.0], [0.0, 0.8 * H, H / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        "arrow_K_valid": np.bool_(True),
        "arrow_pose_reference_index": np.int64(0 if CLIP_OF[index] == 0 else 2),
        "arrow_pose_valid": np.bool_(True),
        "arrow_camera_to_reference": _camera_to_reference(index).astype(np.float32),
        "arrow_camera_to_reconstruct": _camera_to_reconstruct(index).astype(np.float32),
        "arrow_camera_to_metric": _camera_to_metric(index).astype(np.float32),
        # prep
        "prep_edges": np.zeros((H, W), dtype=np.float32),
        "prep_normal": np.broadcast_to(np.array([0.0, 0.0, -1.0], dtype=np.float32), (H, W, 3)).copy(),
        "prep_normal_valid": ones.copy(),
        # lingbot
        "lingbot_depth_in": depth.copy(),
        "lingbot_depth_raw": depth.copy(),
        "lingbot_depth": depth.copy(),
        "lingbot_valid": ones.copy(),
        "lingbot_support_count": np.int64(H * W),
        "lingbot_mask": ones.copy(),
        "lingbot_K_norm": np.array([[0.8, 0.0, 0.5], [0.0, 0.8, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32),
        "lingbot_K_pixel": np.array([[0.8 * W, 0.0, W / 2.0], [0.0, 0.8 * H, H / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        "lingbot_scale_fix": np.float64(1.0),
        # ndr
        "ndr_depth_in": depth.copy(),
        "ndr_depth": depth.copy(),
        "ndr_valid": ones.copy(),
        "ndr_support_count": np.int64(H * W),
        # sceneflow
        "sceneflow_depth": depth.copy(),
        "sceneflow_valid": ones.copy(),
        "sceneflow_support_count": np.int64(H * W),
        "sceneflow_next_depth": depth.copy() if PAIR_OF[index] != (-1, -1) else np.full((H, W), np.nan, np.float32),
        "sceneflow_next_valid": ones.copy() if PAIR_OF[index] != (-1, -1) else np.zeros((H, W), dtype=bool),
        "sceneflow_next_support_count": np.int64(H * W if PAIR_OF[index] != (-1, -1) else 0),
        "sceneflow_pair_index": np.array(PAIR_OF[index], dtype=np.int64),
        "sceneflow_compensation_valid": ones.copy(),
        # temporal
        "temporal_depth": depth.copy(),
        "temporal_valid": ones.copy(),
        "temporal_support_count": np.int64(H * W),
        "temporal_common_support": ones.copy(),
        "temporal_denom": np.int64(H * W),
        "temporal_accepted_fwd": np.bool_(ACCEPTED_FWD[index]),
        "temporal_accepted_bwd": np.bool_(ACCEPTED_BWD[index]),
        "temporal_residual_stats": np.array(_temporal_stats(index), dtype=np.float64),
        # final
        "final_depth": depth.copy(),
        "final_valid": ones.copy(),
        "final_support_count": np.int64(H * W),
    }
    if PAIR_OF[index] != (-1, -1):
        # sr exists only on pair-first frames (producer layout)
        arrays.update({
            "sr_flow": sr_flow,
            "sr_valid": ones.copy(),
            "sr_support_count": np.int64(H * W),
            "sr_flow_low": low_flow,
            "sr_low_valid": low_valid,
            "sr_pair_index": np.array(PAIR_OF[index], dtype=np.int64),
        })
    return arrays


def _write_receipts_and_signatures(dataset_dir: Path, stages: list[str]) -> tuple[dict, dict]:
    receipts, signatures = {}, {}
    for stage in stages:
        attempt = dataset_dir / "attempts" / f"{stage}_20261016T030000"
        attempt.mkdir(parents=True)
        (attempt / "exit.json").write_text(json.dumps({
            "actual_exit_code": 0, "ended_at": "2026-10-16T03:00:03+00:00", "wall_seconds": 1.5,
            "measured_by": "run_depth_detail.py", "timed_out": False, "signals_sent": [],
        }))
        (attempt / "cmd.json").write_text(json.dumps({"argv": ["run_depth_detail.py"]}))
        (attempt / "stdout.log").write_text("ok\n")
        (attempt / "stderr.log").write_text("")
        receipts[stage] = str(attempt)
        signature = dataset_dir / "signatures" / f"{stage}.json"
        signature.parent.mkdir(parents=True, exist_ok=True)
        signature.write_text(json.dumps({
            "inputs": [{"path": "results/arrow/grass/manifest.json", "sha256": "a" * 64}],
            "params": {"stage": stage}, "model": "test-model", "argv": ["run_depth_detail.py"],
            "code_sha256": "b" * 64,
        }))
        signatures[stage] = str(signature)
    return receipts, signatures


def _write_stage_configs(dataset_dir: Path) -> None:
    config_dir = dataset_dir / "configs" / "stage_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "sceneflow_refine.json").write_text(json.dumps({
        "config": {"scale": 0.01, "sigma_depth": 0.003, "iterations": 400,
                   "min_depth": 0.1, "max_depth": 50.0, "max_motion": 0.05, "trust_sigma": 4.0,
                   "edge_aware": True, "use_tgv": True, "fast_match": True},
        "clipped_points": 0, "clipped_frames": 0,
        "pairs": [{"pair_index": [0, 1], "file": "stages/sceneflow/pairs/outputs/pair_000000_000001.npz"}],
    }))
    (config_dir / "ndr.json").write_text(json.dumps({
        "backend": "rust", "iterations": 40,
        "backend_policy": "rust 明示 (benchmark)。失敗時 fallback 禁止",
        "config": {"iterations": 40, "trust_abs": 0.05, "trust_rel": 0.05},
        "benchmark_evidence": "temp/herdr_depth_workflow/ndr_backend_benchmark.json",
    }))


def _manifest(frames: list[dict], receipts: dict, signatures: dict) -> dict:
    stages = ["ingest", "arrow", "align", "anchor", "geometry", "lingbot", "prep", "ndr",
              "sceneflow", "sr", "temporal", "final", "render"]
    return {
        "nonce": "82e63c25084c",
        "dataset": "grass",
        "expected_frame_count": F,
        "original_size": [H, W],
        "units": "arrow_provisional",
        "metric_calibrated": False,
        "calibration": {"single_scalar": 1.0, "applied": False,
                        "targets": ["depth", "camera_translation", "scene_flow_3d"]},
        "stages": stages,
        "optional_stages": ["moge"],
        "models": {stage: "test-model" for stage in ("arrow", "lingbot", "ndr", "sceneflow", "sr", "temporal")},
        "config_hash": "cfg-hash-0001",
        "frames": frames,
        "stage_receipts": receipts,
        "stage_signatures": signatures,
        "stage_configs": {"sceneflow": "configs/stage_configs/sceneflow_refine.json"},
        "temporal": {
            "pairs_compared": F, "denom_total": F * H * W, "error_before": 0.05, "error_after": 0.04,
            "worsened_frames": 0, "rejected_frames": 0,
            "edge_leakage_median": 0.01, "anchor_deviation_median": 0.01,
            "sr_pairs_expected": 2, "sr_pairs_consumed": 2, "sr_pairs_unconsumed": [],
        },
        "sr": {
            "pairs": 2,
            "bilinear_vs_sr": {"abs_diff": {"mean": 0.01, "median": 0.01, "count": 2},
                               "note": "bilinearは比較基準のみ (SRの暗黙fallbackではない)"},
            "low_consistency_abs_diff": {"mean": 0.001, "count": 2},
            "valid_fraction": {"mean": 0.5},
            "gt_gate": "core synthetic step-motion GT test (constant-vector unit preservation)",
        },
        "visuals": {
            "mp4": "visuals/grass_compare.mp4",
            "png": "visuals/grass_representative.png",
            "rrd": "visuals/grass_final_pointcloud.rrd",
        },
    }


def stage_of(key: str) -> str:
    """Map a merged frame key to its producer stage directory (real layout)."""
    if key.startswith("arrow_"):
        geometry_keys = ("arrow_K", "arrow_K_valid", "arrow_pose_valid", "arrow_pose_frame0_valid",
                         "arrow_pose_reference_index", "arrow_camera_to_reference",
                         "arrow_camera_to_reconstruct", "arrow_camera_to_metric")
        return "geometry" if key in geometry_keys else "anchor"
    if key.startswith("prep_"):
        return "prep"
    if key in ("lingbot_depth", "lingbot_scale_fix", "lingbot_scale_raw"):
        return "lingbot_fix"
    if key.startswith("lingbot_"):
        return "lingbot"
    if key.startswith("ndr_"):
        return "ndr"
    if key.startswith("sceneflow_"):
        return "sceneflow_post"
    if key.startswith("sr_"):
        return "sr"
    if key.startswith("temporal_"):
        return "temporal"
    if key.startswith("final_"):
        return "final"
    raise AssertionError(f"unknown merged frame key: {key}")


def read_frame_merged(dataset_dir: Path, index: int) -> dict:
    merged = {}
    for path in sorted((dataset_dir / "stages").glob(f"*/{index:06d}.npz")):
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                if key != "metadata":
                    merged[key] = data[key]
    return merged


def make_tree(tmp_path: Path, *, visuals: bool = False) -> tuple[Path, Path]:
    verifier = _load_verifier()
    stages = list(verifier.REQUIRED_STAGES)
    dataset_dir = tmp_path / "grass"
    frames_dir = dataset_dir / "frames_rgb"
    frames_dir.mkdir(parents=True)
    from PIL import Image

    frames = []
    for index in range(F):
        arrays = _frame_arrays(index, stages)
        arrays.pop("metadata", None)
        stage_payloads: dict[str, dict] = {}
        for key, value in arrays.items():
            stage_payloads.setdefault(stage_of(key), {})[key] = value
        for stage, payload in stage_payloads.items():
            _write_npz(dataset_dir / "stages" / stage / f"{index:06d}.npz",
                       {"metadata": np.array(_metadata(stages)), **payload})
        merged = {"frame_source_index": np.int64(index), "frame_rgb_height": np.int32(H),
                  "frame_rgb_width": np.int32(W), **arrays}
        _write_npz(dataset_dir / "frames" / f"{index:06d}.npz",
                   {"metadata": np.array(_metadata(stages)), **merged})
        png_path = frames_dir / f"{index:06d}.png"
        Image.new("RGB", (W, H), (index, index, index)).save(png_path)
        frames.append({
            "frame_index": index, "source_index": index,
            "relative_path": f"frames/{index:06d}.npz",
            "rgb_path": f"frames_rgb/{index:06d}.png",
            "rgb_sha256": hashlib.sha256(png_path.read_bytes()).hexdigest(),
            "height": H, "width": W,
        })
    receipts, signatures = _write_receipts_and_signatures(dataset_dir, stages)
    _write_stage_configs(dataset_dir)
    clips_dir = dataset_dir / "stages" / "arrow_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    (clips_dir / "manifest.json").write_text(json.dumps({
        "clips": [
            {"name": "clip_0000-0002", "start": 0, "end": 2, "status": "done"},
            {"name": "clip_0002-0004", "start": 2, "end": 4, "status": "done"},
        ]
    }))
    sceneflow_dir = dataset_dir / "stages" / "sceneflow"
    sceneflow_dir.mkdir(parents=True, exist_ok=True)
    (sceneflow_dir / "pairs.json").write_text(json.dumps({
        "counts": {"pairs": 2, "written": 2, "cross_clip_skipped": 1, "frame_count": F},
        "pairs": [
            {"pair_index": [0, 1], "clip_index": 0, "file": "stages/sceneflow/pairs/inputs/pair_000000_000001.npz"},
            {"pair_index": [2, 3], "clip_index": 1, "file": "stages/sceneflow/pairs/inputs/pair_000002_000003.npz"},
        ],
    }))
    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(frames, receipts, signatures), indent=2))
    if visuals:
        cell = 256
        panel_h = max(2, int(round(H * cell / W)))
        panel_h -= panel_h % 2
        frame_w = 2 + 7 * (cell + 2)
        frame_h = panel_h + 4 + 16
        rep_frames = sorted({0, F // 2, F - 1})
        png_h = len(rep_frames) * (frame_h + 6)
        visuals_dir = dataset_dir / "visuals"
        visuals_dir.mkdir()
        (visuals_dir / "grass_compare.mp4").write_bytes(b"\x00" * 256)
        Image.new("RGB", (frame_w, png_h)).save(visuals_dir / "grass_representative.png")
        (visuals_dir / "grass_final_pointcloud.rrd").write_bytes(b"RRF2" + b"\x00" * 256)
        (visuals_dir / "grass_sr_comparison.png").write_bytes(b"PNG" + b"\x00" * 64)
        (visuals_dir / "render_report.json").write_text(json.dumps({
            "cell_width": cell,
            "panel_order": ["rgb", "arrow", "lingbot", "ndr", "sceneflow", "temporal", "sr_flow"],
            "representative_frames": rep_frames,
            "mp4": {"path": str(visuals_dir / "grass_compare.mp4"), "width": frame_w,
                    "height": frame_h, "frames": F},
            "png": {"path": str(visuals_dir / "grass_representative.png")},
            "rrd": {"path": str(visuals_dir / "grass_final_pointcloud.rrd")},
            "sr_comparison_png": str(visuals_dir / "grass_sr_comparison.png"),
        }))
    return dataset_dir, manifest_path


def edit_manifest(manifest_path: Path, mutate) -> None:
    manifest = json.loads(manifest_path.read_text())
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2))


def edit_frame(dataset_dir: Path, index: int, mutate) -> None:
    """Edit the merged view of one frame, then write each key back to its stage dir."""
    metas: dict[str, np.ndarray] = {}
    merged: dict = {}
    for path in sorted((dataset_dir / "stages").glob(f"*/{index:06d}.npz")):
        with np.load(path, allow_pickle=False) as data:
            metas[path.parent.name] = data["metadata"]
            for key in data.files:
                if key != "metadata":
                    merged[key] = data[key]
    mutate(merged)
    payloads: dict[str, dict] = {}
    for key, value in merged.items():
        payloads.setdefault(stage_of(key), {})[key] = value
    for stage in set(metas) | set(payloads):
        path = dataset_dir / "stages" / stage / f"{index:06d}.npz"
        meta = metas.get(stage, np.array(_metadata([])))
        _write_npz(path, {"metadata": meta, **payloads.get(stage, {})})
    merged["frame_source_index"] = np.int64(index)
    merged["frame_rgb_height"] = np.int32(H)
    merged["frame_rgb_width"] = np.int32(W)
    merged.pop("metadata", None)
    _write_npz(dataset_dir / "frames" / f"{index:06d}.npz",
               {"metadata": np.array(_metadata([])), **merged})


def verify(dataset_dir: Path, *, check_render: bool = False, expected_count: int = F,
           partial: bool = False, **kwargs):
    verifier = _load_verifier()
    return verifier.verify_dataset(
        dataset_dir, dataset="grass", expected_count=expected_count, check_render=check_render,
        allow_stage_merge=partial, **kwargs
    )


def failure_ids(report: dict) -> set:
    return {failure["id"] for failure in report["failures"]}


def test_valid_fixture_is_accepted(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    report = verify(dataset_dir)
    assert report["ok"] is True, report["failures"]
    assert report["failures"] == []


def test_units_and_policy_are_recorded(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    report = verify(dataset_dir)
    assert report["units"] == "arrow_provisional"
    assert report["metric_calibrated"] is False
    assert "sr.rescale_diagnostic" in {check["id"] for check in report["checks"]}


def test_valid_fixture_with_render_and_matching_ffprobe_is_accepted(tmp_path):
    dataset_dir, _ = make_tree(tmp_path, visuals=True)
    ffprobe = tmp_path / "ffprobe-stub"
    ffprobe.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        'print(json.dumps({"streams": [{"nb_read_frames": "4", "width": 1808, "height": 224}]}))\n'
    )
    ffprobe.chmod(0o755)
    report = verify(dataset_dir, check_render=True, ffprobe_bin=str(ffprobe))
    assert report["ok"] is True, report["failures"]


def test_missing_manifest_rejected(tmp_path):
    dataset_dir, manifest_path = make_tree(tmp_path)
    manifest_path.unlink()
    report = verify(dataset_dir)
    assert "manifest.missing" in failure_ids(report)


def test_missing_manifest_key_rejected(tmp_path):
    dataset_dir, manifest_path = make_tree(tmp_path)
    edit_manifest(manifest_path, lambda m: m.pop("config_hash"))
    report = verify(dataset_dir)
    assert "manifest.keys" in failure_ids(report)


def test_dataset_mismatch_rejected(tmp_path):
    dataset_dir, manifest_path = make_tree(tmp_path)
    edit_manifest(manifest_path, lambda m: m.update(dataset="railway"))
    report = verify(dataset_dir)
    assert "manifest.dataset" in failure_ids(report)


def test_frame_count_mismatch_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    report = verify(dataset_dir, expected_count=F + 1)
    assert "frames.count" in failure_ids(report)


def test_duplicate_frame_index_rejected(tmp_path):
    dataset_dir, manifest_path = make_tree(tmp_path)
    edit_manifest(manifest_path, lambda m: m["frames"][1].update(frame_index=0))
    report = verify(dataset_dir)
    assert "frames.unique" in failure_ids(report)


def test_duplicate_source_index_rejected(tmp_path):
    dataset_dir, manifest_path = make_tree(tmp_path)
    edit_manifest(manifest_path, lambda m: m["frames"][1].update(source_index=0))
    report = verify(dataset_dir)
    assert "frames.source_unique" in failure_ids(report)


def test_bad_rgb_sha256_rejected(tmp_path):
    dataset_dir, manifest_path = make_tree(tmp_path)
    edit_manifest(manifest_path, lambda m: m["frames"][0].update(rgb_sha256="short"))
    report = verify(dataset_dir)
    assert "frames.rgb_sha256" in failure_ids(report)


def test_units_violation_rejected(tmp_path):
    dataset_dir, manifest_path = make_tree(tmp_path)
    edit_manifest(manifest_path, lambda m: m.update(units="metric_calibrated_meters"))
    report = verify(dataset_dir)
    assert "manifest.units" in failure_ids(report)


def test_metric_calibrated_true_rejected(tmp_path):
    dataset_dir, manifest_path = make_tree(tmp_path)
    edit_manifest(manifest_path, lambda m: m.update(metric_calibrated=True))
    report = verify(dataset_dir)
    assert "manifest.units" in failure_ids(report)


def test_calibration_applied_rejected(tmp_path):
    dataset_dir, manifest_path = make_tree(tmp_path)
    edit_manifest(manifest_path, lambda m: m["calibration"].update(applied=True))
    report = verify(dataset_dir)
    assert "manifest.calibration" in failure_ids(report)


def test_frame_size_entry_rejected(tmp_path):
    dataset_dir, manifest_path = make_tree(tmp_path)
    edit_manifest(manifest_path, lambda m: m["frames"][0].update(height=H + 2))
    report = verify(dataset_dir)
    assert "frames.size" in failure_ids(report)


def test_fake_stage_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    def strip(arrays):
        for key in ("sr_flow", "sr_valid", "sr_support_count", "sr_flow_low", "sr_low_valid", "sr_pair_index"):
            arrays.pop(key)
    edit_frame(dataset_dir, 0, strip)
    report = verify(dataset_dir)
    ids = failure_ids(report)
    assert "stage.fake" in ids or "frame.file" in ids


def test_fake_anchor_stage_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    def strip(arrays):
        for key in ("arrow_depth", "arrow_valid", "arrow_support_count", "arrow_scale", "arrow_clip_index"):
            arrays.pop(key)
    edit_frame(dataset_dir, 0, strip)
    report = verify(dataset_dir)
    assert "stage.fake" in failure_ids(report)


def test_missing_stage_key_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    edit_frame(dataset_dir, 0, lambda arrays: arrays.pop("arrow_valid"))
    report = verify(dataset_dir)
    assert "stage.keys" in failure_ids(report)


def test_shape_mismatch_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    edit_frame(dataset_dir, 0, lambda arrays: arrays.update(arrow_depth=np.zeros((H - 2, W), np.float32)))
    report = verify(dataset_dir)
    assert "shape.mismatch" in failure_ids(report)


def test_dtype_mismatch_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    edit_frame(dataset_dir, 0, lambda arrays: arrays.update(
        ndr_depth=arrays["ndr_depth"].astype(np.int16),
        lingbot_valid=arrays["lingbot_valid"].astype(np.uint8),
    ))
    report = verify(dataset_dir)
    assert "dtype.mismatch" in failure_ids(report)


def test_valid_nonfinite_depth_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    def corrupt(arrays):
        depth = arrays["arrow_depth"].copy()
        depth[0, 0] = np.nan
        arrays["arrow_depth"] = depth
    edit_frame(dataset_dir, 0, corrupt)
    report = verify(dataset_dir)
    assert "depth.valid_finites" in failure_ids(report)


def test_support_count_mismatch_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    edit_frame(dataset_dir, 0, lambda arrays: arrays.update(arrow_support_count=np.int64(1)))
    report = verify(dataset_dir)
    assert "support.count" in failure_ids(report)


def test_zero_support_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    def wipe(arrays):
        arrays["final_valid"] = np.zeros((H, W), dtype=bool)
        arrays["final_depth"] = np.full((H, W), np.nan, np.float32)
        arrays["final_support_count"] = np.int64(0)
    edit_frame(dataset_dir, 0, wipe)
    report = verify(dataset_dir)
    assert "support.zero" in failure_ids(report)


def test_per_frame_scale_variation_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    edit_frame(dataset_dir, 1, lambda arrays: arrays.update(arrow_scale=np.float64(2.5)))
    report = verify(dataset_dir)
    assert "arrow.per_frame_normalization" in failure_ids(report)


def test_normalized_K_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    normalized = np.array([[0.8, 0.0, 0.5], [0.0, 0.8, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    edit_frame(dataset_dir, 0, lambda arrays: arrays.update(arrow_K=normalized))
    report = verify(dataset_dir)
    assert "geometry.K_pixel" in failure_ids(report)


def test_K_bounds_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    def shift(arrays):
        k = arrays["arrow_K"].copy()
        k[0, 2] = float(W)
        arrays["arrow_K"] = k
    edit_frame(dataset_dir, 0, shift)
    report = verify(dataset_dir)
    assert "geometry.K_bounds" in failure_ids(report)


def test_pose_to_reconstruct_violation_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    edit_frame(dataset_dir, 1, lambda arrays: arrays.update(
        arrow_camera_to_reconstruct=_rotation_y(20.0).astype(np.float32)))
    report = verify(dataset_dir)
    assert "geometry.pose_to_reconstruct" in failure_ids(report)


def test_geometry_so3_scaled_rotation_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    def scale_rotation(arrays):
        matrix = arrays["arrow_camera_to_metric"].astype(np.float64).copy()
        matrix[:3, :3] = matrix[:3, :3] * 2.0  # double scale in the linear part
        arrays["arrow_camera_to_metric"] = matrix.astype(np.float32)
    edit_frame(dataset_dir, 0, scale_rotation)
    report = verify(dataset_dir)
    assert "geometry.SO3" in failure_ids(report)


def test_geometry_scale_once_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    def unscaled_translation(arrays):
        matrix = arrays["arrow_camera_to_metric"].astype(np.float64).copy()
        matrix[:3, 3] = arrays["arrow_camera_to_reconstruct"].astype(np.float64)[:3, 3]
        arrays["arrow_camera_to_metric"] = matrix.astype(np.float32)
    edit_frame(dataset_dir, 1, unscaled_translation)
    report = verify(dataset_dir)
    assert "geometry.scale_once" in failure_ids(report)


def test_geometry_clip_transform_consistency_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    def inconsistent(arrays):
        matrix = arrays["arrow_camera_to_metric"].astype(np.float64).copy()
        matrix[:3, 3] = arrays["arrow_camera_to_reconstruct"].astype(np.float64)[:3, 3] * 3.0
        arrays["arrow_camera_to_metric"] = matrix.astype(np.float32)
    edit_frame(dataset_dir, 1, inconsistent)
    report = verify(dataset_dir)
    assert "geometry.clip_transform_consistency" in failure_ids(report)


def test_ndr_numpy_backend_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    config_path = dataset_dir / "configs" / "stage_configs" / "ndr.json"
    config = json.loads(config_path.read_text())
    config.update(backend="numpy")
    config_path.write_text(json.dumps(config))
    report = verify(dataset_dir)
    assert "ndr.resolved_config" in failure_ids(report)


def test_ndr_wrong_iterations_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    config_path = dataset_dir / "configs" / "stage_configs" / "ndr.json"
    config = json.loads(config_path.read_text())
    config.update(iterations=20)
    config_path.write_text(json.dumps(config))
    report = verify(dataset_dir)
    assert "ndr.resolved_config" in failure_ids(report)


def test_lingbot_normalized_K_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    pixel = np.array([[8.0, 0.0, 5.0], [0.0, 6.4, 4.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    edit_frame(dataset_dir, 0, lambda arrays: arrays.update(lingbot_K_norm=pixel))
    report = verify(dataset_dir)
    assert "lingbot.K_normalized" in failure_ids(report)


def test_lingbot_depth_in_mismatch_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    def shift(arrays):
        depth = arrays["lingbot_depth_in"].copy()
        depth += 1.0
        arrays["lingbot_depth_in"] = depth
    edit_frame(dataset_dir, 0, shift)
    report = verify(dataset_dir)
    assert "lingbot.depth_in_mismatch" in failure_ids(report)


def _sparsify_anchor_and_input(arrays, fabricated_finite: bool = False):
    """Real semantics: lingbot_depth_in is the sparse ARROW anchor (mask=arrow_valid);
    the LingBot output stays dense (mask=lingbot_valid)."""
    av = arrays["arrow_valid"].copy()
    av[0, 0] = False
    ad = arrays["arrow_depth"].copy()
    ad[0, 0] = np.nan
    di = arrays["lingbot_depth_in"].copy()
    di[0, 0] = 3.0 if fabricated_finite else np.nan
    arrays["arrow_valid"] = av
    arrays["arrow_depth"] = ad
    arrays["arrow_support_count"] = np.int64(int(av.sum()))
    arrays["lingbot_depth_in"] = di


def test_lingbot_sparse_input_dense_output_accepted(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    edit_frame(dataset_dir, 0, lambda arrays: _sparsify_anchor_and_input(arrays))
    report = verify(dataset_dir)
    assert report["ok"] is True, report["failures"]


def test_lingbot_input_fabricated_finite_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    edit_frame(dataset_dir, 0, lambda arrays: _sparsify_anchor_and_input(arrays, fabricated_finite=True))
    report = verify(dataset_dir)
    assert "depth.valid_finites" in failure_ids(report)


def test_lingbot_raw_nonfinite_on_valid_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    def corrupt(arrays):
        raw = arrays["lingbot_depth_raw"].copy()
        raw[0, 0] = np.nan
        arrays["lingbot_depth_raw"] = raw
    edit_frame(dataset_dir, 0, corrupt)
    report = verify(dataset_dir)
    assert "depth.valid_finites" in failure_ids(report)


def test_ndr_valid_inflation_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    def inflate(arrays):
        mask = arrays["lingbot_valid"].copy()
        mask[0, 0] = False
        valid = arrays["ndr_valid"].copy()
        valid[0, 0] = True
        arrays["lingbot_valid"] = mask
        arrays["ndr_valid"] = valid
    edit_frame(dataset_dir, 0, inflate)
    report = verify(dataset_dir)
    assert "ndr.valid_inflation" in failure_ids(report)


def test_ndr_depth_in_mismatch_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    def shift(arrays):
        depth = arrays["ndr_depth_in"].copy()
        depth += 2.0
        arrays["ndr_depth_in"] = depth
    edit_frame(dataset_dir, 0, shift)
    report = verify(dataset_dir)
    assert "ndr.depth_in_mismatch" in failure_ids(report)


def test_sceneflow_next_beyond_compensation_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    def shrink(arrays):
        compensation = arrays["sceneflow_compensation_valid"].copy()
        compensation[0, 0] = False
        arrays["sceneflow_compensation_valid"] = compensation
    edit_frame(dataset_dir, 0, shrink)
    report = verify(dataset_dir)
    assert "sceneflow.next_beyond_compensation" in failure_ids(report)


def test_sceneflow_support_inflation_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    def inflate(arrays):
        ndr_valid = arrays["ndr_valid"].copy()
        ndr_valid[0, 0] = False
        ndr_depth = arrays["ndr_depth"].copy()
        ndr_depth[0, 0] = np.nan
        arrays["ndr_valid"] = ndr_valid
        arrays["ndr_depth"] = ndr_depth
    edit_frame(dataset_dir, 0, inflate)
    report = verify(dataset_dir)
    assert "sceneflow.support_inflation" in failure_ids(report)


def test_sceneflow_zero_compensation_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    def wipe(arrays):
        arrays["sceneflow_compensation_valid"] = np.zeros((H, W), dtype=bool)
        arrays["sceneflow_valid"] = np.zeros((H, W), dtype=bool)
        arrays["sceneflow_support_count"] = np.int64(0)
        arrays["sceneflow_depth"] = np.full((H, W), np.nan, np.float32)
        arrays["sr_valid"] = np.zeros((H, W), dtype=bool)
        arrays["sr_support_count"] = np.int64(0)
    edit_frame(dataset_dir, 0, wipe)
    report = verify(dataset_dir)
    assert "denom.zero" in failure_ids(report)


def test_sceneflow_next_without_pair_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    def claim(arrays):
        arrays["sceneflow_next_valid"] = np.ones((H, W), dtype=bool)
        arrays["sceneflow_next_depth"] = np.full((H, W), 2.0, np.float32)
        arrays["sceneflow_next_support_count"] = np.int64(H * W)
    edit_frame(dataset_dir, 1, claim)
    report = verify(dataset_dir)
    assert "sceneflow.next_without_pair" in failure_ids(report)


def test_sceneflow_pair_count_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    edit_frame(dataset_dir, 2, lambda arrays: arrays.update(sceneflow_pair_index=np.array([-1, -1], np.int64)))
    report = verify(dataset_dir)
    assert "sceneflow.pair_count" in failure_ids(report)


def test_sr_valid_beyond_support_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    def extend(arrays):
        valid = arrays["sceneflow_valid"].copy()
        valid[0, 0] = False
        arrays["sceneflow_valid"] = valid
    edit_frame(dataset_dir, 0, extend)
    report = verify(dataset_dir)
    assert "sr.valid_beyond_support" in failure_ids(report)


def test_sr_valid_without_depth_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    def break_depth(arrays):
        depth = arrays["ndr_depth"].copy()
        depth[0, 0] = np.nan
        ndr_valid = arrays["ndr_valid"].copy()
        ndr_valid[0, 0] = False
        sceneflow = arrays["sceneflow_valid"].copy()
        sceneflow[0, 0] = False
        arrays["ndr_depth"] = depth
        arrays["ndr_valid"] = ndr_valid
        arrays["sceneflow_valid"] = sceneflow
    edit_frame(dataset_dir, 0, break_depth)
    report = verify(dataset_dir)
    assert "sr.valid_without_depth" in failure_ids(report)


def test_sr_constant_vector_rescale_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    def rescale(arrays):
        arrays["sr_flow"] = (arrays["sr_flow"].astype(np.float64) * (H / LOW_H)).astype(np.float32)
    edit_frame(dataset_dir, 0, rescale)
    report = verify(dataset_dir)
    assert "sr.unit_preservation" in failure_ids(report)


def test_sr_variable_flow_ratio_is_diagnostic_only(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    def variable(arrays):
        low = np.zeros((LOW_H, LOW_W, 3), dtype=np.float64)
        for j in range(LOW_H):
            for i in range(LOW_W):
                low[j, i] = (1.0 + 0.01 * i, 0.0, 0.0)
        high = np.zeros((H, W, 3), dtype=np.float64)
        for y in range(H):
            for x in range(W):
                high[y, x] = ((1.0 + 0.01 * x) * (H / LOW_H), 0.0, 0.0)
        arrays["sr_flow_low"] = low.astype(np.float32)
        arrays["sr_flow"] = high.astype(np.float32)
    edit_frame(dataset_dir, 0, variable)
    report = verify(dataset_dir)
    ids = failure_ids(report)
    assert "sr.unit_preservation" not in ids
    assert "sr.rescale_diagnostic" in {check["id"] for check in report["checks"]}


def test_temporal_zero_denominator_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    def zero(arrays):
        arrays["temporal_common_support"] = np.zeros((H, W), dtype=bool)
        arrays["temporal_denom"] = np.int64(0)
    edit_frame(dataset_dir, 1, zero)
    report = verify(dataset_dir)
    assert "denom.zero" in failure_ids(report)


def test_temporal_accepted_zero_denominator_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    def zero_n(arrays):
        stats = arrays["temporal_residual_stats"].copy()
        stats[2] = 0.0  # fwd_before_n in the producer per-direction layout
        arrays["temporal_residual_stats"] = stats
    edit_frame(dataset_dir, 0, zero_n)
    report = verify(dataset_dir)
    assert "denom.zero" in failure_ids(report)


def test_temporal_silent_worsening_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    def worsen(arrays):
        stats = arrays["temporal_residual_stats"].copy()
        stats[1] = 0.09
        arrays["temporal_residual_stats"] = stats
    edit_frame(dataset_dir, 0, worsen)
    report = verify(dataset_dir)
    assert "gate.silent_worsening" in failure_ids(report)


def test_final_depth_mismatch_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    def shift(arrays):
        depth = arrays["final_depth"].copy()
        depth += 1.0
        arrays["final_depth"] = depth
    edit_frame(dataset_dir, 0, shift)
    report = verify(dataset_dir)
    assert "final.depth_mismatch" in failure_ids(report)


def test_receipt_missing_rejected(tmp_path):
    dataset_dir, manifest_path = make_tree(tmp_path)
    edit_manifest(manifest_path, lambda m: m["stage_receipts"].pop("ndr"))
    report = verify(dataset_dir)
    assert "receipts.missing" in failure_ids(report)


def test_receipt_nonzero_exit_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    exit_path = dataset_dir / "attempts" / "lingbot_20261016T030000" / "exit.json"
    payload = json.loads(exit_path.read_text())
    payload["actual_exit_code"] = 1
    exit_path.write_text(json.dumps(payload))
    report = verify(dataset_dir)
    assert "receipt.exit" in failure_ids(report)


def test_receipt_timeout_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    exit_path = dataset_dir / "attempts" / "sr_20261016T030000" / "exit.json"
    payload = json.loads(exit_path.read_text())
    payload["timed_out"] = True
    exit_path.write_text(json.dumps(payload))
    report = verify(dataset_dir)
    assert "receipt.timeout" in failure_ids(report)


def test_receipt_files_missing_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    (dataset_dir / "attempts" / "anchor_20261016T030000" / "cmd.json").unlink()
    report = verify(dataset_dir)
    assert "receipts.files" in failure_ids(report)


def test_signature_missing_rejected(tmp_path):
    dataset_dir, manifest_path = make_tree(tmp_path)
    edit_manifest(manifest_path, lambda m: m["stage_signatures"].pop("temporal"))
    report = verify(dataset_dir)
    assert "signatures.missing" in failure_ids(report)


def test_sceneflow_config_missing_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    config_path = dataset_dir / "configs" / "stage_configs" / "sceneflow_refine.json"
    config = json.loads(config_path.read_text())
    config["config"].pop("min_depth")
    config_path.write_text(json.dumps(config))
    report = verify(dataset_dir)
    assert "sceneflow.config" in failure_ids(report)


def test_sceneflow_silent_clipping_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    config_path = dataset_dir / "configs" / "stage_configs" / "sceneflow_refine.json"
    config = json.loads(config_path.read_text())
    config.update(clipped_points=5, clipped_frames=0)
    config_path.write_text(json.dumps(config))
    report = verify(dataset_dir)
    assert "gate.silent_clipping" in failure_ids(report)


def test_manifest_temporal_missing_key_rejected(tmp_path):
    dataset_dir, manifest_path = make_tree(tmp_path)
    edit_manifest(manifest_path, lambda m: m["temporal"].pop("error_before"))
    report = verify(dataset_dir)
    assert "temporal.manifest_keys" in failure_ids(report)


def test_manifest_temporal_silent_worsening_rejected(tmp_path):
    dataset_dir, manifest_path = make_tree(tmp_path)
    edit_manifest(manifest_path, lambda m: m["temporal"].update(error_after=0.09, worsened_frames=0))
    report = verify(dataset_dir)
    assert "gate.silent_worsening" in failure_ids(report)


def test_denominator_total_mismatch_rejected(tmp_path):
    dataset_dir, manifest_path = make_tree(tmp_path)
    edit_manifest(manifest_path, lambda m: m["temporal"].update(denom_total=1))
    report = verify(dataset_dir)
    assert "denom.total_mismatch" in failure_ids(report)


def test_sr_manifest_missing_key_rejected(tmp_path):
    dataset_dir, manifest_path = make_tree(tmp_path)
    edit_manifest(manifest_path, lambda m: m["sr"].pop("bilinear_vs_sr"))
    report = verify(dataset_dir)
    assert "sr.manifest_keys" in failure_ids(report)


def test_sr_unconsumed_pair_rejected(tmp_path):
    dataset_dir, manifest_path = make_tree(tmp_path)
    edit_manifest(manifest_path, lambda m: m["temporal"].update(sr_pairs_consumed=1))
    report = verify(dataset_dir)
    assert "sr.consumption" in failure_ids(report)


def test_sr_consumption_missing_rejected(tmp_path):
    dataset_dir, manifest_path = make_tree(tmp_path)
    def strip(m):
        m["temporal"].pop("sr_pairs_consumed", None)
        m["temporal"].pop("sr_pairs_expected", None)
        m["temporal"].pop("sr_pairs_unconsumed", None)
        m["sr"].pop("consumption", None)
    edit_manifest(manifest_path, strip)
    report = verify(dataset_dir)
    assert "sr.consumption" in failure_ids(report)


def test_visuals_missing_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    report = verify(dataset_dir, check_render=True)
    ids = failure_ids(report)
    assert "visuals.mp4" in ids
    assert "visuals.png" in ids
    assert "visuals.rrd" in ids


def test_ffprobe_frame_count_mismatch_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path, visuals=True)
    ffprobe = tmp_path / "ffprobe-stub"
    ffprobe.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        'print(json.dumps({"streams": [{"nb_read_frames": "3", "width": 1808, "height": 224}]}))\n'
    )
    ffprobe.chmod(0o755)
    report = verify(dataset_dir, check_render=True, ffprobe_bin=str(ffprobe))
    assert "visuals.ffprobe" in failure_ids(report)


def test_sr_stage_alt_storage_supported(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    stages_dir = dataset_dir / "stages"
    stages_dir.mkdir(exist_ok=True)
    flow_rows, valid_rows, low_rows = [], [], []
    for index in (0, 2):
        arrays = read_frame_merged(dataset_dir, index)
        flow_rows.append(arrays["sr_flow"])
        valid_rows.append(arrays["sr_valid"])
        low_rows.append(arrays["sr_flow_low"])
        (stages_dir / "sr" / f"{index:06d}.npz").unlink()
    np.savez(
        stages_dir / "sr.npz",
        pair_index=np.array([[0, 1], [2, 3]], dtype=np.int64),
        flow3d=np.stack(flow_rows),
        flow3d_valid=np.stack(valid_rows),
        low_flow=np.stack(low_rows),
        low_shape=np.tile(np.array([[LOW_H, LOW_W]], dtype=np.int32), (2, 1)),
    )
    report = verify(dataset_dir)
    assert report["ok"] is True, report["failures"]


def test_missing_frames_pack_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    (dataset_dir / "frames" / "000001.npz").unlink()
    report = verify(dataset_dir)
    assert "frame.file" in failure_ids(report)


def test_partial_merge_requires_explicit_flag(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    for path in (dataset_dir / "frames").glob("*.npz"):
        path.unlink()
    full = verify(dataset_dir)
    assert "frame.file" in failure_ids(full)
    partial = verify(dataset_dir, partial=True)
    assert partial["ok"] is True, partial["failures"]
    assert "frames.pack_partial" in {check["id"] for check in partial["checks"]}


def test_rgb_sha256_content_mismatch_rejected(tmp_path):
    dataset_dir, manifest_path = make_tree(tmp_path)
    edit_manifest(manifest_path, lambda m: m["frames"][0].update(rgb_sha256="0" * 64))
    report = verify(dataset_dir)
    assert "frame.source_meta" in failure_ids(report)


def test_rgb_size_mismatch_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    from PIL import Image
    Image.new("RGB", (W + 4, H)).save(dataset_dir / "frames_rgb" / "000000.png")
    report = verify(dataset_dir)
    assert "frame.source_meta" in failure_ids(report)


def _write_empty_sr(dataset_dir: Path, index: int, *, flow_value: float | None = None,
                    valid_value: bool | None = None) -> None:
    arrays = {
        "sr_flow": np.full((H, W, 3), np.nan, np.float32),
        "sr_valid": np.zeros((H, W), dtype=bool),
        "sr_support_count": np.int64(0),
        "sr_flow_low": np.full((LOW_H, LOW_W, 3), np.nan, np.float32),
        "sr_low_valid": np.zeros((LOW_H, LOW_W), dtype=bool),
        "sr_pair_index": np.array([-1, -1], dtype=np.int64),
    }
    if flow_value is not None:
        arrays["sr_flow"][0, 0] = flow_value
    if valid_value is not None:
        arrays["sr_valid"][0, 0] = valid_value
        arrays["sr_support_count"] = np.int64(int(arrays["sr_valid"].sum()))
    _write_npz(dataset_dir / "stages" / "sr" / f"{index:06d}.npz",
               {"metadata": np.array(_metadata([])), **arrays})


def test_manifest_stage_dependency_inversion_rejected(tmp_path):
    dataset_dir, manifest_path = make_tree(tmp_path)
    def invert(manifest):
        stages = manifest["stages"]
        i, j = stages.index("lingbot"), stages.index("prep")
        stages[i], stages[j] = stages[j], stages[i]
    edit_manifest(manifest_path, invert)
    report = verify(dataset_dir)
    assert "manifest.stages" in failure_ids(report)


def test_sceneflow_config_bool_string_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    config_path = dataset_dir / "configs" / "stage_configs" / "sceneflow_refine.json"
    config = json.loads(config_path.read_text())
    config["config"]["edge_aware"] = "true"
    config_path.write_text(json.dumps(config))
    report = verify(dataset_dir)
    assert "sceneflow.config" in failure_ids(report)


def test_sceneflow_config_numeric_string_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    config_path = dataset_dir / "configs" / "stage_configs" / "sceneflow_refine.json"
    config = json.loads(config_path.read_text())
    config["config"]["max_depth"] = "50.0"
    config_path.write_text(json.dumps(config))
    report = verify(dataset_dir)
    assert "sceneflow.config" in failure_ids(report)


def test_temporal_interleaved_stats_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    def old_layout(arrays):
        arrays["temporal_residual_stats"] = np.array(
            [0.05, 0.04, 0.05, 0.04, H * W, H * W, H * W, H * W], dtype=np.float64)
    edit_frame(dataset_dir, 0, old_layout)
    report = verify(dataset_dir)
    assert "denom.zero" in failure_ids(report)


def test_temporal_accepted_direction_nan_stats_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    def nan_fwd(arrays):
        stats = arrays["temporal_residual_stats"].copy()
        stats[0:4] = np.nan
        arrays["temporal_residual_stats"] = stats
    edit_frame(dataset_dir, 0, nan_fwd)
    report = verify(dataset_dir)
    ids = failure_ids(report)
    assert "denom.zero" in ids or "temporal.stats_finite" in ids


def test_sr_unpaired_empty_file_accepted(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    _write_empty_sr(dataset_dir, 3)
    report = verify(dataset_dir)
    assert report["ok"] is True, report["failures"]


def test_sr_unpaired_finite_flow_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    _write_empty_sr(dataset_dir, 3, flow_value=1.0)
    report = verify(dataset_dir)
    assert "sr.unpaired_schema" in failure_ids(report)


def test_sr_unpaired_fabricated_valid_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    _write_empty_sr(dataset_dir, 3, valid_value=True)
    report = verify(dataset_dir)
    assert "sr.unpaired_schema" in failure_ids(report)


def test_expected_pairs_overlap_and_nonoverlap():
    verifier = _load_verifier()
    fn = getattr(verifier, "expected_pairs_from_clips")
    overlap = {"clips": [{"name": "c0", "start": 0, "end": 24, "status": "done"},
                         {"name": "c1", "start": 20, "end": 44, "status": "done"}]}
    pairs, skipped = fn(overlap, 44)
    assert len(pairs) == 43 and skipped == 0
    disjoint = {"clips": [{"name": "c0", "start": 0, "end": 24, "status": "done"},
                          {"name": "c1", "start": 24, "end": 44, "status": "done"}]}
    pairs2, skipped2 = fn(disjoint, 44)
    assert len(pairs2) == 42 and skipped2 == 1
    assert (23, 24) not in pairs2


def test_pairs_missing_entry_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    pairs_path = dataset_dir / "stages" / "sceneflow" / "pairs.json"
    payload = json.loads(pairs_path.read_text())
    payload["pairs"].pop()
    payload["counts"]["pairs"] -= 1
    pairs_path.write_text(json.dumps(payload))
    report = verify(dataset_dir)
    ids = failure_ids(report)
    assert "sceneflow.pairs_expected" in ids or "sceneflow.pair_count" in ids


def test_pairs_duplicate_entry_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    pairs_path = dataset_dir / "stages" / "sceneflow" / "pairs.json"
    payload = json.loads(pairs_path.read_text())
    payload["pairs"].append(dict(payload["pairs"][0]))
    payload["counts"]["pairs"] += 1
    pairs_path.write_text(json.dumps(payload))
    report = verify(dataset_dir)
    assert "sceneflow.pairs_expected" in failure_ids(report)


def test_pairs_invalid_clip_index_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    pairs_path = dataset_dir / "stages" / "sceneflow" / "pairs.json"
    payload = json.loads(pairs_path.read_text())
    payload["pairs"][0]["clip_index"] = 9
    pairs_path.write_text(json.dumps(payload))
    report = verify(dataset_dir)
    assert "sceneflow.pairs_clip" in failure_ids(report)


def test_pairs_cross_clip_count_mismatch_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path)
    pairs_path = dataset_dir / "stages" / "sceneflow" / "pairs.json"
    payload = json.loads(pairs_path.read_text())
    payload["counts"]["cross_clip_skipped"] = 0
    pairs_path.write_text(json.dumps(payload))
    report = verify(dataset_dir)
    assert "sceneflow.cross_clip" in failure_ids(report)


def test_bilinear_unknown_string_rejected(tmp_path):
    dataset_dir, manifest_path = make_tree(tmp_path)
    edit_manifest(manifest_path,
                  lambda m: m["sr"]["bilinear_vs_sr"].update({"weird": "text"}))
    report = verify(dataset_dir)
    assert "sr.bilinear_record" in failure_ids(report)


def test_bilinear_nan_numeric_rejected(tmp_path):
    dataset_dir, manifest_path = make_tree(tmp_path)
    edit_manifest(manifest_path,
                  lambda m: m["sr"]["bilinear_vs_sr"]["abs_diff"].update({"mean": None}))
    report = verify(dataset_dir)
    assert "sr.bilinear_record" in failure_ids(report)


def test_visuals_wrong_layout_rejected(tmp_path):
    dataset_dir, _ = make_tree(tmp_path, visuals=True)
    report_path = dataset_dir / "visuals" / "render_report.json"
    payload = json.loads(report_path.read_text())
    payload["cell_width"] = 128
    report_path.write_text(json.dumps(payload))
    report = verify(dataset_dir, check_render=True)
    report_with_ffprobe = verify(dataset_dir, check_render=True,
                                 ffprobe_bin=str(_write_layout_ffprobe(tmp_path)))
    ids = failure_ids(report) | failure_ids(report_with_ffprobe)
    assert "visuals.layout" in ids


def _write_layout_ffprobe(tmp_path: Path) -> Path:
    ffprobe = tmp_path / "ffprobe-layout-stub"
    ffprobe.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        'print(json.dumps({"streams": [{"nb_read_frames": "4", "width": 1808, "height": 224}]}))\n'
    )
    ffprobe.chmod(0o755)
    return ffprobe


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
