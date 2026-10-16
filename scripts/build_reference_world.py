#!/usr/bin/env python3
"""Export the strictly frame-zero canonical surface without later-frame fusion.

This companion to build_static_world protects the observed reference shape from
tracking drift. Previously unseen surfaces remain holes in the virtual RGB-D.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from build_static_world import (
    Budget, Sources, digest, now, render_dataset, save_npz, source_code_pins,
    write_json, write_ply,
)
import json


def configure_view(output, dataset, sources, budget):
    """Keep the fixed RGB model visible without the depth backprojection overlay."""
    import rerun.blueprint as rrb
    import rerun as rr

    manifest = json.loads((output / "manifest.json").read_text())
    if digest(output / "model.npz") != manifest["model_sha256"]:
        raise ValueError("Reference model changed before Rerun export")
    for pin in manifest["input_pins"]:
        if digest(pin["path"]) != pin["sha256"]:
            raise ValueError(f"Reference input changed: {pin['path']}")
    model = {}
    with np.load(output / "model.npz") as data:
        for key in ("points", "colors"):
            model[key] = data[key]
    with np.load(output / "cameras.npz") as data:
        matrices, intrinsics = data["camera_to_world"], data["K"]
    h, w = manifest["original_size"]
    budget.check(h * w * 5 * manifest["frame_count"])
    app_id = f"sr_reference_{dataset}"
    rrd_path = output / f"{dataset}_reference_world.rrd"
    rbl_path = output / f"{dataset}_reference_view.rbl"
    rr.init(app_id, spawn=False)
    rr.save(str(rrd_path))
    rr.log("world", rr.ViewCoordinates.RDF, static=True)
    step = manifest["rerun_display_stride"]
    rr.log("world/canonical", rr.Points3D(model["points"][::step], colors=model["colors"][::step],
        radii=rr.Radius.ui_points(1.0)), static=True)
    rr.log("world/camera_path", rr.LineStrips3D([matrices[:, :3, 3]], colors=[100, 165, 255]), static=True)
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="Fixed reference RGB geometry (frame 0)", origin="/world",
                contents=["/world/canonical", "/world/camera_path", "/world/camera"],
                background=[15, 17, 22, 255]),
            rrb.Vertical(
                rrb.Spatial2DView(name="Observed RGB", origin="/observed"),
                rrb.Spatial2DView(name="Virtual frozen RGB", origin="/world/camera/rgb"),
                rrb.Spatial2DView(name="Camera-z (arrow_provisional)", origin="/world/camera/depth")),
            column_shares=[2, 1]),
        auto_layout=False, auto_views=False, collapse_panels=True)
    rr.send_blueprint(blueprint)
    blueprint.save(app_id, rbl_path)
    try:
        for i, record in enumerate(sources.records):
            rr.set_time("frame", sequence=i)
            rr.set_time("time", duration=i / manifest["fps"])
            rr.log("world/camera", rr.Transform3D(translation=matrices[i, :3, 3], mat3x3=matrices[i, :3, :3]))
            rr.log("world/camera", rr.Pinhole(image_from_camera=intrinsics[i], resolution=[w, h]))
            rgb_path = Path(record["rgb_path"])
            if not rgb_path.is_absolute():
                rgb_path = sources.directory / rgb_path
            rr.log("observed", rr.Image(np.asarray(Image.open(rgb_path).convert("RGB"))).compress(jpeg_quality=88))
            with np.load(output / "rgbd" / f"{i:06d}.npz") as data:
                rr.log("world/camera/rgb", rr.Image(data["rgb"]).compress(jpeg_quality=90))
                rr.log("world/camera/depth", rr.DepthImage(np.nan_to_num(data["depth"], nan=0)))
    finally:
        rr.disconnect()
    manifest["presentation_rrd"] = rrd_path.name
    manifest["presentation_blueprint"] = rbl_path.name
    manifest["presentation_point_radius_ui"] = 1.0
    manifest["presentation_source_pin"] = {"path": str(Path(__file__).absolute()), "sha256": digest(__file__)}
    names = {rrd_path.name, rbl_path.name}
    manifest["artifacts"] = [a for a in manifest["artifacts"] if a["path"] not in names]
    manifest["artifacts"].extend({"path": p.name, "bytes": p.stat().st_size, "sha256": digest(p)}
                                 for p in (rrd_path, rbl_path))
    write_json(output / "manifest.json", manifest)
    budget.check()


def export_reference(args, dataset, budget):
    parent = args.source_dir / dataset / "static_world"
    output = args.source_dir / dataset / "reference_world"
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Reference output already exists: {output}")
    original = json.loads((parent / "manifest.json").read_text())
    if original["render_status"] != "complete":
        raise ValueError("The source static export must be complete")
    if digest(parent / "model.npz") != original["model_sha256"]:
        raise ValueError("Source canonical model hash mismatch")
    sources = Sources(args.input_dir / dataset)
    for pin in original["input_pins"]:
        if digest(pin["path"]) != pin["sha256"]:
            raise ValueError(f"Original input changed: {pin['path']}")
        sources.pin(Path(pin["path"]))
    for name in ("model.npz", "cameras.npz", "manifest.json"):
        sources.pin(parent / name)
    with np.load(parent / "model.npz", allow_pickle=False) as data:
        born = data["birth_frame"]
        selected = np.flatnonzero(born == 0)
        if not selected.size:
            raise ValueError("No reference-frame anchors")
        model = {}
        for key in data.files:
            value = data[key]
            model[key] = value[selected] if value.ndim and value.shape[0] == len(born) else value
    model["source_anchor_id"] = selected.astype(np.int64)
    model["canonical_policy"] = np.asarray("frame0 observed geometry and color; later surfaces omitted")
    if np.any(model["birth_correction"]) or np.any(model["correction_support"]):
        raise ValueError("Reference anchors must have no inverse birth correction")
    output.mkdir(parents=True, exist_ok=True)
    (output / "rgbd").mkdir()
    save_npz(output / "model.npz", budget, **model)
    budget.check(model["points"].nbytes + model["colors"].nbytes + 1024)
    write_ply(output / "model.ply", model["points"], model["colors"])
    budget.saved(output / "model.ply")
    shutil.copyfile(parent / "cameras.npz", output / "cameras.npz")
    manifest = {
        "schema_version": 1, "export_kind": "reference_only", "created_at": now(),
        "dataset": dataset, "frame_count": original["frame_count"],
        "pair_count": original["pair_count"], "original_size": original["original_size"],
        "units": "arrow_provisional", "metric_calibrated": False, "reference_frame": 0,
        "canonical_policy": str(model["canonical_policy"]),
        "late_surface_policy": "omitted; unknown reference geometry stays missing",
        "parent_model": str((parent / "model.npz").absolute()),
        "parent_model_sha256": original["model_sha256"],
        "parent_model_point_count": original["model_point_count"],
        "model_point_count": int(len(selected)), "frame0_anchor_count": int(len(selected)),
        "later_anchor_count": 0, "motion_corrected_anchor_count": 0,
        "voxel_size": original["voxel_size"], "model_stride": original["model_stride"],
        "depth_range": original["depth_range"],
        "input_pins": list(sources.pins.values()),
        "source_pins": source_code_pins() + [{"path": str(Path(__file__).absolute()),
                                             "sha256": digest(__file__)}],
        "model_sha256": digest(output / "model.npz"), "render_status": "pending",
    }
    write_json(output / "manifest.json", manifest)
    render_dataset(dataset, sources, output, args, budget, manifest)
    configure_view(output, dataset, sources, budget)
    print(f"{dataset} reference: {len(selected)} fixed points; {output}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results/depth_detail_oct16"))
    parser.add_argument("--source-dir", type=Path, default=Path("results/sr_static_oct16"))
    parser.add_argument("--dataset", choices=("all", "grass", "railway"), default="all")
    parser.add_argument("--view-only", action="store_true", help="write the RGB-model Rerun blueprint for a completed export")
    parser.add_argument("--splat-radius", type=int, choices=range(4), default=2)
    parser.add_argument("--rerun-max-points", type=int, default=250000)
    parser.add_argument("--fps", type=int, default=0)
    parser.add_argument("--max-output-gib", type=float, default=30)
    parser.add_argument("--min-free-gib", type=float, default=35)
    args = parser.parse_args(argv)
    args.input_dir = args.input_dir.resolve()
    args.source_dir = args.source_dir.resolve()
    if args.rerun_max_points < 1 or args.fps < 0:
        parser.error("Point budget must be positive and fps nonnegative")
    if args.input_dir == args.source_dir or args.input_dir in args.source_dir.parents or args.source_dir in args.input_dir.parents:
        parser.error("Input and source output roots must be disjoint")
    budget = Budget(args.source_dir, int(args.max_output_gib * 2**30), int(args.min_free_gib * 2**30))
    budget.check()
    for dataset in (("grass", "railway") if args.dataset == "all" else (args.dataset,)):
        if args.view_only:
            output = args.source_dir / dataset / "reference_world"
            if json.loads((output / "manifest.json").read_text())["render_status"] != "complete":
                raise ValueError("View-only requires a completed reference export")
            configure_view(output, dataset, Sources(args.input_dir / dataset), budget)
        else:
            export_reference(args, dataset, budget)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
