#!/usr/bin/env python3
"""Prepare an appearance guide using one reference-camera plane.

This is a 2D vegetation-freeze approximation, not a reconstructed static 3D world.
Only the reference frame has canonical depth; later canonical geometry is invalid.
Original camera poses, intrinsics, reference observations and known RGB are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arrow.static_world import transform_points, unproject_depth  # noqa: E402
from scripts.prepare_bullettime_controls import (  # noqa: E402
    bilinear_rgb, source_map, validate_intrinsics, write_video,
)
from scripts.prepare_flow_freeze import (  # noqa: E402
    compose_draft, foreground_candidate, foreground_seed, map_mask_native,
    sample_depth_safe,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_pose(M):
    M = np.asarray(M, np.float64)
    if (M.shape != (4, 4) or not np.isfinite(M).all()
            or not np.allclose(M[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(M[:3, :3].T @ M[:3, :3], np.eye(3), atol=2e-4, rtol=0)
            or not np.isclose(np.linalg.det(M[:3, :3]), 1, atol=2e-4)):
        raise ValueError("Expected a finite rigid camera-to-world pose")
    return M


def plane_homography(K0, Kout, M0, Mt, distance):
    """Source z=d plane: H=Kout*(R_t0+t_t0*n.T/d)*inv(K0), n=(0,0,1)."""
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("Plane distance must be finite and positive")
    K0, Kout = validate_intrinsics(K0), validate_intrinsics(Kout)
    relative = np.linalg.inv(validate_pose(Mt)) @ validate_pose(M0)
    H = Kout @ (relative[:3, :3] + np.outer(relative[:3, 3], [0, 0, 1]) / distance) @ np.linalg.inv(K0)
    if not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
        raise ValueError("Invalid plane homography")
    return H, relative


def render_plane(rgb, seed, candidate, K0, Kout, M0, Mt, distance, shape):
    """Inverse-map original RGB; every positive-weight color donor must be green.

    Projected plane depth gates behind-camera samples only. It is not exported
    as measured/canonical geometry. Pixel centers use integer coordinates.
    """
    if (rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8
            or seed.shape != rgb.shape[:2] or candidate.shape != rgb.shape[:2]
            or seed.dtype != bool or candidate.dtype != bool or len(shape) != 2 or min(shape) <= 0):
        raise ValueError("Invalid RGB, mask or output dimensions")
    H, relative = plane_homography(K0, Kout, M0, Mt, distance)
    yy, xx = np.indices(shape)
    target = np.stack([xx, yy, np.ones(shape)], -1)
    native_h = target @ np.linalg.inv(H).T
    with np.errstate(divide="ignore", invalid="ignore"):
        native = native_h[..., :2] / native_h[..., 2, None]
    h, w = rgb.shape[:2]
    inside = np.isfinite(native).all(-1)
    inside &= ((native[..., 0] >= -1e-7) & (native[..., 0] <= w - 1 + 1e-7)
               & (native[..., 1] >= -1e-7) & (native[..., 1] <= h - 1 + 1e-7))
    xy = np.clip(np.nan_to_num(native), [0, 0], [w - 1, h - 1])
    source_ray = np.concatenate([xy, np.ones(shape + (1,))], -1) @ np.linalg.inv(K0).T
    target_depth = ((source_ray * distance) @ relative[:3, :3].T + relative[:3, 3])[..., 2]
    inside &= np.isfinite(target_depth) & (target_depth > 0)
    x0, y0 = np.moveaxis(np.floor(xy).astype(np.int64), -1, 0)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = np.moveaxis(xy - np.floor(xy), -1, 0)
    weights = np.stack([(1-fx)*(1-fy), fx*(1-fy), (1-fx)*fy, fx*fy], -1)
    donors = np.stack([seed[y0, x0], seed[y0, x1], seed[y1, x0], seed[y1, x1]], -1)
    trusted = inside & ((weights <= 1e-8) | donors).all(-1)
    footprint = inside & (candidate[y0, x0] | candidate[y0, x1] | candidate[y1, x0] | candidate[y1, x1])
    colors = np.stack([rgb[y0, x0], rgb[y0, x1], rgb[y1, x0], rgb[y1, x1]], -2)
    out = np.rint((colors * weights[..., None]).sum(-2)).clip(0, 255).astype(np.uint8)
    source_uv = np.where(trusted[..., None], native, np.nan).astype(np.float32)
    return out, trusted, footprint, source_uv, H


def validate_reference_model(path, rgb, depth, K, M):
    """Accept only frame-zero observed points/colors, kept separate from the plane."""
    with np.load(path, allow_pickle=False) as data:
        if np.asarray(data["reference_frame"]).shape != () or int(data["reference_frame"]) != 0:
            raise ValueError("Reference model must describe frame zero")
        points, colors, uv = data["points"], data["colors"], data["native_uv"]
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) == 0
            or colors.shape != points.shape or colors.dtype != np.uint8
            or uv.shape != (len(points), 2) or not np.issubdtype(uv.dtype, np.integer)
            or not np.isfinite(points).all()):
        raise ValueError("Invalid observed reference model arrays")
    x, y = uv[:, 0], uv[:, 1]
    if (x.min() < 0 or x.max() >= rgb.shape[1] or y.min() < 0 or y.max() >= rgb.shape[0]
            or not np.array_equal(colors, rgb[y, x])):
        raise ValueError("Reference model colors/UV differ from original frame zero")
    expected = transform_points(unproject_depth(depth, K), M)[y, x]
    if not np.isfinite(expected).all() or not np.allclose(points, expected, atol=1e-6, rtol=2e-6):
        raise ValueError("Reference model points differ from observed reference depth/pose")


def build(input_dir, rectified_controls, reference_model, output_dir):
    started = time.monotonic()
    input_dir, rectified_controls, reference_model, output_dir = (
        Path(p).resolve() for p in (input_dir, rectified_controls, reference_model, output_dir))
    if (input_dir == output_dir or input_dir in output_dir.parents
            or output_dir in input_dir.parents or output_dir in rectified_controls.parents
            or output_dir in reference_model.parents):
        raise ValueError("Output must be separate from all original inputs")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output must be empty")
    pins = {str(p): digest(p) for p in (rectified_controls, reference_model, input_dir / "frames.json")}
    with np.load(rectified_controls, allow_pickle=False) as data:
        source = data["rgb"].copy()
        Ks, common, Ms = data["original_K"].copy(), data["K"].copy(), data["camera_to_world"].copy()
        indices = data["native_indices"].copy()
        fps_array = data["fps"]
        actual_time = data["actual_timestamps"].copy()
    if source.ndim != 4 or source.shape[-1] != 3 or source.dtype != np.uint8:
        raise ValueError("Rectified RGB must be uint8[T,H,W,3]")
    count, height, width, _ = source.shape
    if (count < 5 or (count-1) % 4 or min(height, width) <= 0 or height % 16 or width % 16
            or indices.shape != (count,) or not np.array_equal(indices, np.arange(count))):
        raise ValueError("Require T=4k+1 contiguous frames from zero and positive 16-aligned dimensions")
    if fps_array.shape != () or not np.isfinite(fps_array) or float(fps_array) <= 0:
        raise ValueError("Invalid playback fps")
    fps = float(fps_array)
    if (actual_time.shape != (count,) or not np.isfinite(actual_time).all()
            or not np.allclose(actual_time, np.arange(count)/fps, atol=1e-9, rtol=0)):
        raise ValueError("Actual timestamps must match contiguous source playback times")
    if Ks.shape != (count, 3, 3) or common.shape != (count, 3, 3) or Ms.shape != (count, 4, 4):
        raise ValueError("Invalid per-frame calibration array shapes")
    if not np.array_equal(common, np.repeat(common[:1], count, axis=0)):
        raise ValueError("Rectified intrinsics must be constant")
    for K, common_K, M in zip(Ks, common, Ms, strict=True):
        validate_intrinsics(K)
        validate_intrinsics(common_K)
        validate_pose(M)
    info = json.loads((input_dir / "frames.json").read_text())
    records = info["frames"][:count]
    if len(records) != count or [r["frame_index"] for r in records] != list(range(count)):
        raise ValueError("Original frame list is not contiguous from zero")
    shape = (height, width)

    def load(index):
        rgb_path = Path(records[index]["rgb_path"])
        if not rgb_path.is_absolute():
            rgb_path = input_dir / rgb_path
        rgb_path = rgb_path.resolve()
        depth_path = input_dir / "stages/final" / f"{index:06d}.npz"
        geometry_path = input_dir / "stages/geometry" / f"{index:06d}.npz"
        for path in (rgb_path, depth_path, geometry_path):
            pins[str(path)] = digest(path)
        if records[index].get("rgb_sha256") != pins[str(rgb_path)]:
            raise ValueError(f"Original RGB hash mismatch at frame {index}")
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        if tuple(rgb.shape[:2]) != tuple(info["original_size"]):
            raise ValueError("Native RGB dimensions disagree with frames.json")
        with np.load(geometry_path, allow_pickle=False) as data:
            if (not bool(data["arrow_K_valid"]) or not bool(data["arrow_pose_valid"])
                    or not np.array_equal(data["arrow_K"], Ks[index])
                    or not np.array_equal(data["arrow_camera_to_metric"], Ms[index])):
                raise ValueError(f"Original K/pose differs from rectified controls at frame {index}")
        with np.load(depth_path, allow_pickle=False) as data:
            depth, valid = data["final_depth"], data["final_valid"]
            if depth.shape != rgb.shape[:2] or valid.shape != depth.shape or valid.dtype != bool:
                raise ValueError("Invalid depth/mask shape or type")
            depth = np.where(valid & np.isfinite(depth) & (depth > 0), depth, np.nan).astype(np.float32)
        x, y = source_map(Ks[index], common[index], shape)
        if not np.array_equal(bilinear_rgb(rgb, x, y), source[index]):
            raise ValueError(f"Rectified RGB differs from original rays at frame {index}")
        return rgb, depth

    initial, initial_depth = load(0)
    seed, candidate = foreground_seed(initial), foreground_candidate(initial)
    depth_support = np.isfinite(initial_depth) & seed
    if not depth_support.any():
        raise ValueError("No valid observed green reference depth for the guide plane")
    distance = float(np.median(initial_depth[depth_support]))
    quantiles = np.quantile(initial_depth[depth_support], [0, .1, .25, .5, .75, .9, 1]).tolist()
    validate_reference_model(reference_model, initial, initial_depth, Ks[0], Ms[0])
    original_frame0_depth = sample_depth_safe(initial_depth, Ks[0], common[0], shape)
    buffers = {k: [] for k in ("draft_rgb", "edit_mask", "source_preserved_mask", "reference_projected_mask",
                              "foreground_source_mask", "foreground_destination_mask", "foreground_union_mask",
                              "source_uv", "source_uv_frame_index", "observed_background_depth", "plane_homography")}
    rows = []
    previews = set(np.linspace(0, count-1, min(5, count), dtype=int).tolist())
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        current, depth = (initial, initial_depth) if index == 0 else load(index)
        source_mask = map_mask_native(foreground_candidate(current), Ks[index], common[index], shape)
        rendered, trusted, destination, uv, H = render_plane(
            initial, seed, candidate, Ks[0], common[index], Ms[0], Ms[index], distance, shape)
        draft, edit, preserved, known, union = compose_draft(
            source[index], rendered, trusted, source_mask, destination, reference=index == 0)
        uv[~known] = np.nan
        uv_frame = np.where(known, 0, -1).astype(np.int16)
        observed = sample_depth_safe(depth, Ks[index], common[index], shape)
        observed = np.where(preserved & ~union, observed, np.nan).astype(np.float32)
        values = (draft, edit, preserved, known, source_mask, destination, union, uv, uv_frame, observed, H)
        for key, value in zip(buffers, values, strict=True):
            buffers[key].append(value)
        row = {"frame": index, "source_preserved_fraction": float(preserved.mean()),
               "billboard_projected_fraction": float(known.mean()), "edit_fraction": float(edit.mean())}
        rows.append(row)
        print(json.dumps(row), flush=True)
        if index in previews:
            Image.fromarray(np.concatenate([source[index], draft], 1)).save(
                output_dir / f"frame_{index:02d}_source_draft.png")
    report = {
        "status": "2D_billboard_experiment_guide_not_3D_reconstruction", "frames": count,
        "shape_hw": [height, width], "fps": fps, "actual_timestamps": actual_time.tolist(),
        "input_dir": str(input_dir), "rectified_controls": str(rectified_controls), "reference_model": str(reference_model),
        "reference_frame": 0, "units": "arrow_provisional", "metric_calibrated": False,
        "method": "H=Kcommon*(R_t0+t_t0*n^T/d)*inv(K0); n=(0,0,1), d=median(frame0 valid actual-green depth). "
                  "Inverse-map integer pixel centers to original frame0 RGB; all positive-weight bilinear donors must be green. "
                  "No current-warp mixing. Outside source/destination union current observed RGB is preserved exactly.",
        "plane_depth_arrow_provisional": distance, "plane_depth_quantiles": quantiles,
        "plane_depth_samples": int(depth_support.sum()), "original_camera_poses": True, "pose_quality": "provisional",
        "geometry_contract": "Frame0 depth is observed reference depth with edge-safe sampling. All later canonical_depth values "
                             "are NaN and geometry_valid false: the plane is an appearance guide, not measured geometry. "
                             "Original observed reference model is separately preserved unchanged. Current observed_background_depth "
                             "outside the editing union is separate from canonical geometry.",
        "limitations": ["A single plane approximates parallax, thickness and occlusion.", "Original pose errors remain.",
                        "Green heuristic may miss pale stems and include nearby vegetation.",
                        "Reference appearance fixes illumination; hidden surfaces are not recovered.",
                        "Generated unknown RGB never implies measured depth or a validated static world."],
        "statistics": rows, "input_sha256": pins, "script_sha256": digest(__file__),
        "helper_sha256": {name: digest(Path(__file__).with_name(name))
                          for name in ("prepare_flow_freeze.py", "prepare_bullettime_controls.py")},
    }
    arrays: dict[str, Any] = {key: np.stack(value) for key, value in buffers.items()}
    canonical = np.full((count,) + shape, np.nan, np.float32)
    canonical[0] = original_frame0_depth
    geometry_valid = np.isfinite(canonical)
    if (not np.array_equal(arrays["draft_rgb"][0], source[0]) or arrays["edit_mask"][0].any()
            or not np.array_equal(arrays["draft_rgb"][arrays["source_preserved_mask"]], source[arrays["source_preserved_mask"]])
            or not np.all(arrays["draft_rgb"][arrays["edit_mask"]] == 127)):
        raise RuntimeError("Known-RGB preservation contract failed")
    np.savez_compressed(output_dir / "controls.npz", **arrays, source_rgb=source, reference_rgb=source[0],
        fps=np.array(fps), actual_timestamps=actual_time, K=common, camera_to_world=Ms, original_K=Ks,
        native_indices=indices, known_mask=~arrays["edit_mask"], warped_known_mask=np.zeros_like(arrays["edit_mask"]),
        billboard_projected_mask=arrays["reference_projected_mask"], canonical_depth=canonical,
        geometry_valid=geometry_valid, metadata=np.array(json.dumps(report, allow_nan=False)))
    shutil.copyfile(reference_model, output_dir / "reference_model.npz")
    if digest(output_dir / "reference_model.npz") != pins[str(reference_model)]:
        raise RuntimeError("Copied observed reference model differs")
    write_video(output_dir / "draft.mp4", arrays["draft_rgb"], fps)
    write_video(output_dir / "comparison.mp4", np.concatenate([source, arrays["draft_rgb"]], 2), fps)
    write_video(output_dir / "edit_mask.mp4", np.repeat(arrays["edit_mask"][..., None], 3, -1).astype(np.uint8)*255, fps)
    for path, expected in pins.items():
        if digest(path) != expected:
            raise RuntimeError("An original input changed: " + path)
    report["source_verified_after_build"] = True
    report["wall_seconds"] = time.monotonic()-started
    report["artifacts"] = {p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
                           for p in output_dir.iterdir() if p.is_file()}
    (output_dir / "manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--rectified-controls", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, required=True,
                        help="Observed frame0 NPZ with points/colors/native_uv/reference_frame; preserved unchanged")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.input_dir, args.rectified_controls, args.reference_model, args.output_dir)
    print(json.dumps({"status": result["status"], "frames": result["frames"], "wall_seconds": result["wall_seconds"]}))


if __name__ == "__main__":
    main()
