#!/usr/bin/env python3
"""Export assisted RGB beside observed reference geometry, without invented depth.

All 3D values use the native RDF world and provisional ARROW scale. Unknown
generated pixels retain NaN canonical depth. Current observed background depth
is logged separately and is never fused into the fixed reference point cloud.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
from PIL import Image


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(data)
    return value.hexdigest()


def load_pngs(directory, count, shape):
    paths = [directory / f"frame_{index:04d}.png" for index in range(count)]
    if sorted(directory.glob("*.png")) != paths:
        raise ValueError(f"Missing or extra PNG frames: {directory}")
    frames, receipts = [], []
    for path in paths:
        with Image.open(path) as image:
            if image.mode != "RGB":
                raise ValueError(f"Expected RGB PNG: {path}")
            frame = np.asarray(image).copy()
        if frame.shape != shape or frame.dtype != np.uint8:
            raise ValueError(f"Wrong frame shape or dtype: {path}")
        frames.append(frame)
        receipts.append({"path": str(path.resolve()), "sha256": digest(path)})
    return np.stack(frames), receipts


def load_inputs(controls_path, reference_path, generated_dir):
    required = ("source_rgb", "draft_rgb", "reference_rgb", "edit_mask", "fps", "K",
                "camera_to_world", "canonical_depth", "geometry_valid")
    optional = ("source_preserved_mask", "warped_known_mask", "reference_projected_mask",
                "known_mask", "observed_background_depth")
    with np.load(controls_path, allow_pickle=False) as data:
        for name in required:
            if name not in data:
                raise ValueError(f"Missing controls key: {name}")
        controls = {name: data[name].copy() for name in (*required, *optional) if name in data}
    rgb = controls["source_rgb"]
    if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8 or min(rgb.shape[:3]) < 1:
        raise ValueError("source_rgb must be uint8 nonempty [T,H,W,3]")
    count, height, width, _ = rgb.shape
    fps = controls["fps"]
    if fps.shape != () or not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a positive finite scalar")
    for name in ("edit_mask", "geometry_valid"):
        if controls[name].shape != rgb.shape[:3] or controls[name].dtype != np.bool_:
            raise ValueError(f"{name} must be bool [T,H,W]")
    depth, valid, edit = controls["canonical_depth"], controls["geometry_valid"], controls["edit_mask"]
    if depth.shape != valid.shape or depth.dtype.kind != "f":
        raise ValueError("canonical_depth must be floating [T,H,W]")
    if (valid & edit).any():
        raise ValueError("Generated pixels cannot claim canonical geometry")
    if not np.isfinite(depth[valid]).all() or np.any(depth[valid] <= 0):
        raise ValueError("geometry_valid pixels require positive finite canonical depth")
    if not np.isnan(depth[~valid]).all():
        raise ValueError("Unsupported canonical depth must remain NaN")
    if "observed_background_depth" in controls:
        background = controls["observed_background_depth"]
        if background.shape != valid.shape or background.dtype.kind != "f":
            raise ValueError("observed_background_depth must be floating [T,H,W]")
        finite = np.isfinite(background)
        if np.any(background[finite] <= 0) or not np.isnan(background[~finite]).all() or (finite & edit).any():
            raise ValueError("Observed background depth must be positive or NaN, outside generated pixels")
    cameras, intrinsics = controls["camera_to_world"], controls["K"]
    if cameras.shape != (count, 4, 4) or not np.isfinite(cameras).all():
        raise ValueError("camera_to_world must be finite [T,4,4]")
    if not np.allclose(cameras[:, 3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("Invalid camera homogeneous row")
    rotations = cameras[:, :3, :3]
    if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-4) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-4):
        raise ValueError("camera_to_world must be a rigid transform")
    if intrinsics.shape != (count, 3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError("K must be finite [T,3,3]")
    if np.any(intrinsics[:, (0, 1), (0, 1)] <= 0) or not np.allclose(intrinsics[:, 2], [0, 0, 1]):
        raise ValueError("Invalid pinhole K")
    with np.load(reference_path, allow_pickle=False) as data:
        points, colors = data["points"].copy(), data["colors"].copy()
        units = str(data["units"]) if "units" in data else "arrow_provisional"
        if "metric_calibrated" in data and bool(data["metric_calibrated"]):
            raise ValueError("This export contract expects uncalibrated ARROW reference geometry")
        if "reference_frame" in data and int(data["reference_frame"]) != 0:
            raise ValueError("Reference model must represent frame zero")
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Reference points must be nonempty finite [N,3]")
    if colors.shape != points.shape or colors.dtype != np.uint8:
        raise ValueError("Reference colors must be uint8 [N,3]")
    raw, raw_receipts = load_pngs(generated_dir / "raw_frames", count, (height, width, 3))
    composite, composite_receipts = load_pngs(generated_dir / "composite_frames", count, (height, width, 3))
    evaluator_path = Path(__file__).with_name("evaluate_vace_freeze.py")
    spec = importlib.util.spec_from_file_location("guided_pixel_audit", evaluator_path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot import independent pixel evaluator")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    pixel_report, _ = evaluator.analyze_pixels(
        rgb, controls["draft_rgb"], raw, composite, edit, controls, controls["reference_rgb"])
    if pixel_report["integrity_failures"]:
        raise ValueError(f"Composite integrity failed: {pixel_report['integrity_failures']}")
    model = {"points": points.astype(np.float32), "colors": colors, "units": units,
             "source_points_dtype": str(points.dtype)}
    return controls, model, composite, raw_receipts + composite_receipts, pixel_report


def write_ply(path, points, colors):
    records = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                          ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for axis, name in enumerate(("x", "y", "z")):
        records[name] = points[:, axis]
    for axis, name in enumerate(("red", "green", "blue")):
        records[name] = colors[:, axis]
    header = ("ply\nformat binary_little_endian 1.0\n"
              "comment Observed reference geometry; provisional ARROW units; native RDF\n"
              f"element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with Path(path).open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(records.tobytes())


def verify_recording(path):
    sibling = Path(sys.executable).with_name("rerun")
    executable = str(sibling) if sibling.is_file() else shutil.which("rerun")
    if executable is None:
        raise RuntimeError("rerun binary is required to verify mandatory recording artifacts")
    result = subprocess.run([executable, "rrd", "verify", str(path)], capture_output=True,
                            text=True, timeout=120)
    receipt = {"argv": result.args, "exit_code": result.returncode,
               "stdout": result.stdout, "stderr": result.stderr}
    if result.returncode:
        raise RuntimeError(f"Rerun verification failed: {receipt}")
    return receipt


def write_recording(output, controls, model, composite):
    import rerun as rr
    import rerun.blueprint as rrb

    source = controls["source_rgb"]
    count, height, width, _ = source.shape
    application = "guided_freeze_observed_reference"
    rr.init(application, spawn=False)
    rr.save(str(output / "guided_freeze.rrd"))
    notes = ("# VACE assisted RGB / observed reference geometry\n\n"
             "The fixed cloud contains observed frame-0 points only. Units are provisional ARROW, not calibrated meters.\n\n"
             "Camera poses and intrinsics use the original native RDF convention. Unknown generated pixels have no canonical depth.\n\n"
             "Current observed background depth is a separate 2D diagnostic and is never fused into the canonical cloud.\n\n"
             "Raw PNG compositing checks do not establish visual quality, freeze correctness, or metric RGB-D reconstruction.")
    try:
        rr.log("world", rr.ViewCoordinates.RDF, static=True)
        rr.log("world/canonical", rr.Points3D(model["points"], colors=model["colors"]), static=True)
        rr.log("world/camera_path", rr.LineStrips3D([controls["camera_to_world"][:, :3, 3]],
                                                   colors=[110, 180, 255]), static=True)
        rr.log("notes", rr.TextDocument(notes, media_type="text/markdown"), static=True)
        views: list[Any] = [rrb.Horizontal(
            rrb.Spatial2DView(name="Observed source RGB", origin="/view/source"),
            rrb.Spatial2DView(name="VACE assisted RGB / known pixels preserved", origin="/view/composite")),
            rrb.Horizontal(
            rrb.Spatial2DView(name="Generation mask / white = unknown", origin="/view/edit_mask"),
            rrb.Spatial2DView(name="Supported canonical camera-z / provisional", origin="/world/camera/canonical_depth"))]
        if "observed_background_depth" in controls:
            views.append(rrb.Spatial2DView(name="Current observed background depth / separate diagnostic",
                                           origin="/diagnostic/observed_background_depth"))
        views.append(rrb.TextDocumentView(name="Meaning and limitations", origin="/notes"))
        blueprint = rrb.Blueprint(
            rrb.Horizontal(rrb.Spatial3DView(name="Observed reference geometry / provisional ARROW", origin="/world"),
                           rrb.Vertical(*views), column_shares=[2, 3]),
            auto_layout=False, auto_views=False, collapse_panels=True)
        rr.send_blueprint(blueprint)
        blueprint.save(application, str(output / "guided_freeze.rbl"))
        for index in range(count):
            rr.set_time("frame", sequence=index)
            rr.set_time("elapsed", duration=index / float(controls["fps"]))
            pose = controls["camera_to_world"][index]
            rr.log("world/camera", rr.Transform3D(mat3x3=pose[:3, :3], translation=pose[:3, 3]))
            rr.log("world/camera", rr.Pinhole(image_from_camera=controls["K"][index], resolution=[width, height]))
            rr.log("view/source", rr.Image(source[index]))
            rr.log("view/composite", rr.Image(composite[index]))
            rr.log("view/edit_mask", rr.Image(controls["edit_mask"][index].astype(np.uint8) * 255))
            rr.log("world/camera/canonical_depth", rr.DepthImage(controls["canonical_depth"][index].astype(np.float32)))
            if "observed_background_depth" in controls:
                rr.log("diagnostic/observed_background_depth",
                       rr.DepthImage(controls["observed_background_depth"][index].astype(np.float32)))
    finally:
        rr.disconnect()
    return rr.__version__


def export(controls_path, reference_path, generated_dir, output_dir):
    controls_path, reference_path, generated_dir, output_dir = map(
        Path, (controls_path, reference_path, generated_dir, output_dir))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory must be unoccupied")
    pins = [{"path": str(path.resolve()), "sha256": digest(path)} for path in (controls_path, reference_path)]
    controls, model, composite, image_pins, pixel_report = load_inputs(controls_path, reference_path, generated_dir)
    pins.extend(image_pins)
    generation_manifest = generated_dir / "manifest.json"
    if generation_manifest.is_file():
        metadata = json.loads(generation_manifest.read_text())
        if metadata.get("controls", {}).get("sha256") != pins[0]["sha256"]:
            raise ValueError("Generated controls hash differs from export controls")
        pins.append({"path": str(generation_manifest.resolve()), "sha256": digest(generation_manifest)})
    output_dir.mkdir(parents=True)
    write_ply(output_dir / "canonicalpointcloud.ply", model["points"], model["colors"])
    version = write_recording(output_dir, controls, model, composite)
    verification = {name: verify_recording(output_dir / name) for name in ("guided_freeze.rrd", "guided_freeze.rbl")}
    for pin in pins:
        if digest(pin["path"]) != pin["sha256"]:
            raise RuntimeError(f"Input changed during export: {pin['path']}")
    shape = controls["source_rgb"].shape
    frame_rows = []
    for index in range(shape[0]):
        row = {"frame": index, "canonical_geometry_pixels": int(controls["geometry_valid"][index].sum()),
               "unknown_generated_pixels": int(controls["edit_mask"][index].sum()),
               "unknown_fraction": float(controls["edit_mask"][index].mean())}
        if "observed_background_depth" in controls:
            row["separate_observed_background_depth_pixels"] = int(np.isfinite(controls["observed_background_depth"][index]).sum())
        frame_rows.append(row)
    manifest = {"status": "exported_not_quality_accepted", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": digest(__file__), "rerun_version": version,
        "coordinate_convention": "native RDF everywhere; no display rotation", "units": model["units"],
        "metric_calibrated": False, "canonical_points": len(model["points"]),
        "geometry_storage_dtype": "float32", "source_points_dtype": model["source_points_dtype"],
        "canonical_reference_only": True, "canonical_static_log_count": 1,
        "generated_3d_points": 0, "unknown_depth": "NaN, retained in RRD float32 buffer",
        "observed_background": "separate diagnostic entity outside world; never fused into canonical cloud",
        "frame_count": shape[0], "shape_hw": list(shape[1:3]), "fps": float(controls["fps"]),
        "frames": frame_rows, "input_pins": pins, "pixel_integrity": pixel_report["integrity"],
        "rrd_verify": verification,
        "artifacts": {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in output_dir.iterdir() if p.is_file()}}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": manifest["status"], "output_dir": str(output_dir.resolve()),
                      "canonical_points": manifest["canonical_points"], "frames": shape[0]}))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        export(args.controls, args.reference_model, args.generated_dir, args.output_dir)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
