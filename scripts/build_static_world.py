#!/usr/bin/env python3
"""Decompose saved SR motion and freeze observed surfaces into a canonical world.

No inference is run. Moving surfaces retain their first-observed geometry;
successful flow correspondences inherit their original immutable anchor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone
from typing import Any

import av
import numpy as np
from PIL import Image, ImageDraw

from arrow.static_world import (
    advect_labels,
    decompose_flow,
    render_points,
    transform_points,
    unproject_depth,
)


UNITS = "arrow_provisional"


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


class Budget:
    def __init__(self, root, max_bytes, reserve):
        self.root = Path(root)
        self.maximum = max_bytes
        self.reserve = reserve
        self.used = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())

    def check(self, additional=0):
        # Include concurrently written videos/visual exports, and count rewrites
        # once. A cached byte counter misses files written by another consumer.
        self.used = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        if self.used + additional > self.maximum:
            raise RuntimeError(f"Output budget exceeded: {self.used}+{additional}>{self.maximum}")
        if shutil.disk_usage(self.root).free < self.reserve + additional:
            raise RuntimeError("Insufficient free space for output plus reserve")

    def saved(self, path):
        self.check()


def save_npz(path, budget, **arrays):
    budget.check(sum(np.asarray(v).nbytes for v in arrays.values()))
    np.savez_compressed(path, **arrays)
    budget.saved(path)


class Sources:
    """Cache each small/array input once and pin the real file read."""

    def __init__(self, directory):
        self.directory = directory
        self.pins = {}
        self.pin(directory / "frames.json")
        self.frames = json.loads((directory / "frames.json").read_text())
        self.records = self.frames["frames"]
        indices = [x["frame_index"] for x in self.records]
        if indices != list(range(len(indices))) or not indices or len(indices) != self.frames["frame_count"]:
            raise ValueError("Expected contiguous frame indices starting at zero")

    def pin(self, path):
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Input must be a regular file: {path}")
        key = str(path.absolute())
        if key not in self.pins:
            self.pins[key] = {"path": key, "bytes": path.stat().st_size, "sha256": digest(path)}

    def stage(self, name, index):
        path = self.directory / "stages" / name / f"{index:06d}.npz"
        self.pin(path)
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files if key != "metadata"}

    def frame(self, index):
        final = self.stage("final", index)
        geo = self.stage("geometry", index)
        rgb_path = Path(self.records[index]["rgb_path"])
        if not rgb_path.is_absolute():
            rgb_path = self.directory / rgb_path
        self.pin(rgb_path)
        declared_hash = self.records[index].get("rgb_sha256")
        if declared_hash is not None and declared_hash != self.pins[str(rgb_path.absolute())]["sha256"]:
            raise ValueError(f"RGB does not match frames.json hash: {rgb_path}")
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        depth = np.asarray(final["final_depth"], np.float32)
        valid = np.asarray(final["final_valid"], bool) & np.isfinite(depth) & (depth > 0)
        if depth.shape != rgb.shape[:2] or depth.shape != tuple(self.frames["original_size"]):
            raise ValueError(f"Frame {index} source dimensions do not agree")
        if not valid.any() or not bool(geo["arrow_K_valid"]) or not bool(geo["arrow_pose_valid"]):
            raise ValueError(f"Frame {index} has no valid depth or calibration")
        return {"depth": np.where(valid, depth, np.nan), "valid": valid, "rgb": rgb,
                "K": geo["arrow_K"], "M": geo["arrow_camera_to_metric"]}

    def verify(self):
        for value in self.pins.values():
            path = Path(value["path"])
            if path.stat().st_size != value["bytes"] or digest(path) != value["sha256"]:
                raise RuntimeError(f"Source changed during export: {path}")


def source_code_pins():
    import arrow.static_world as core

    return [{"path": str(p), "sha256": digest(p)} for p in
            (Path(__file__).absolute(), Path(core.__file__).absolute())]


def write_ply(path, points, colors):
    """Binary little-endian PLY, positions in the same world as the numeric model."""
    vertices = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                          ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for i, field in enumerate(("x", "y", "z")):
        vertices[field] = points[:, i]
    for i, field in enumerate(("red", "green", "blue")):
        vertices[field] = colors[:, i]
    header = ("ply\nformat binary_little_endian 1.0\ncomment units arrow_provisional\n"
              f"element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertices.tobytes())


def birth_corrections(world, labels, anchor_positions, depth, rgb, locations):
    """Estimate local inverse deformation at unmatched, newly sampled pixels.

    Matched neighbors carry D=live_world-canonical_anchor. A robust component
    median of at least four depth/color-compatible neighbors transports this
    displacement to a newly visible sample. Unsupported points keep zero
    correction and are explicitly first-observed, not frame0 reconstructions.
    """
    h, w = depth.shape
    rows, cols = np.unravel_index(locations, (h, w))
    corrections = np.full((32, len(locations), 3), np.nan, np.float32)
    source_rgb = rgb.reshape(-1, 3)[locations].astype(np.float32)
    source_z = depth.ravel()[locations]
    slot = 0
    for radius in (1, 2, 4, 8):
        for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            yy, xx = rows + dy * radius, cols + dx * radius
            inside = (yy >= 0) & (xx >= 0) & (yy < h) & (xx < w)
            yy, xx = np.clip(yy, 0, h - 1), np.clip(xx, 0, w - 1)
            ids = labels[yy, xx]
            z = depth[yy, xx]
            dz = np.abs(z - source_z) / np.maximum(z, source_z)
            dc = np.linalg.norm(rgb[yy, xx].astype(np.float32) - source_rgb, axis=-1)
            supported = inside & (ids >= 0) & np.isfinite(z) & (dz <= 0.05) & (dc <= 80)
            if supported.any():
                corrections[slot, supported] = world[yy[supported], xx[supported]] - anchor_positions[ids[supported]]
            slot += 1
    counts = np.isfinite(corrections[..., 0]).sum(axis=0)
    result = np.zeros((len(locations), 3), np.float32)
    supported = counts >= 4
    if supported.any():
        result[supported] = np.nanmedian(corrections[:, supported], axis=0)
    return result, np.where(supported, counts, 0).astype(np.uint8)


def build_dataset(name, sources, output, args, budget):
    """Streaming label transport. Anchor locations/colors never follow live deformation."""
    for folder in ("motion", "correspondence", "rgbd"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    count = len(sources.records)
    current = sources.frame(0)
    h, w = current["depth"].shape
    labels = np.full((h, w), -1, np.int64)
    stride = args.model_stride
    seed_grid = np.zeros((h, w), bool)
    seed_grid[::stride, ::stride] = True
    points_list, colors_list, birth_list, pixel_list = [], [], [], []
    correction_list, donor_count_list = [], []
    observations: list[int] = []
    last_seen: list[int] = []
    # Spatial deduplication applies to newly observed anchors only; tracking always
    # preserves exact identity and does not quantize the existing anchor position.
    voxel = args.voxel_size
    if voxel == 0:
        voxel = float(np.median(current["depth"][current["valid"]]) /
                      max(current["K"][0, 0], current["K"][1, 1]) * stride * 0.5)
    lookup: dict[tuple[int, int, int], int] = {}
    rows, cameras, intrinsics = [], [], []
    depth_samples = []
    total_points = 0

    for index in range(count):
        cameras.append(current["M"])
        intrinsics.append(current["K"])
        depth_samples.append(current["depth"][::16, ::16].ravel())
        world = transform_points(unproject_depth(current["depth"], current["K"]), current["M"])
        new_mask = current["valid"] & (labels < 0) & seed_grid
        locations = np.flatnonzero(new_mask)
        new_positions = world.reshape(-1, 3)[locations]
        new_colors = current["rgb"].reshape(-1, 3)[locations]
        if points_list:
            correction, donor_count = birth_corrections(world, labels, np.concatenate(points_list),
                current["depth"], current["rgb"], locations)
        else:
            correction = np.zeros((len(locations), 3), np.float32)
            donor_count = np.zeros(len(locations), np.uint8)
        new_positions = new_positions - correction
        keys = np.floor(new_positions / voxel).astype(np.int64)
        added_p, added_c, added_pixel, added_correction, added_donors = [], [], [], [], []
        new_labels = np.empty(len(locations), np.int64)
        for j, key_array in enumerate(keys):
            key = (int(key_array[0]), int(key_array[1]), int(key_array[2]))
            label = lookup.get(key)
            if label is None:
                label = total_points + len(added_p)
                lookup[key] = label
                added_p.append(new_positions[j])
                added_c.append(new_colors[j])
                added_pixel.append(locations[j])
                added_correction.append(correction[j])
                added_donors.append(donor_count[j])
                observations.append(0)
                last_seen.append(index)
            new_labels[j] = label
        if total_points + len(added_p) > args.max_points:
            raise RuntimeError("Canonical point count exceeds --max-points; no silent truncation")
        if added_p:
            points_list.append(np.asarray(added_p, np.float32))
            colors_list.append(np.asarray(added_c, np.uint8))
            birth_list.append(np.full(len(added_p), index, np.int32))
            pixel_list.append(np.asarray(added_pixel, np.int32))
            correction_list.append(np.asarray(added_correction, np.float32))
            donor_count_list.append(np.asarray(added_donors, np.uint8))
            total_points += len(added_p)
        labels.ravel()[locations] = new_labels
        seen = np.unique(labels[labels >= 0])
        for label in seen:
            observations[label] += 1
            last_seen[label] = index
        inherited = int(np.count_nonzero((labels >= 0) & ~new_mask))
        row: dict[str, Any] = {"frame_index": index, "source_index": int(sources.records[index]["source_index"]),
               "valid_depth_count": int(current["valid"].sum()), "canonical_pixel_count": int((labels >= 0).sum()),
               "inherited_pixel_count": inherited, "new_anchor_count": len(added_p),
               "motion_corrected_new_anchors": int(np.count_nonzero(added_donors)),
               "deduplicated_new_pixels": len(locations) - len(added_p), "model_point_count": total_points}
        save_npz(output / "correspondence" / f"{index:06d}.npz", budget,
                 anchor_id=labels.astype(np.int32), valid=labels >= 0,
                 frame_index=np.int32(index), source_index=np.int32(row["source_index"]))
        if index + 1 < count:
            following = sources.frame(index + 1)
            sr = sources.stage("sr", index)
            if "sr_pair_index" in sr and not np.array_equal(sr["sr_pair_index"], [index, index + 1]):
                raise ValueError(f"SR frame membership mismatch at {index}")
            decomposition = decompose_flow(current["depth"], current["K"], following["K"],
                    current["M"], following["M"], sr["sr_flow"], sr["sr_valid"] & current["valid"])
            support = decomposition["valid"]
            if not support.any():
                raise RuntimeError(f"No valid scene flow at pair {index}")
            err2 = np.max(np.abs(decomposition["total_flow"][support] -
                    decomposition["rigid_flow"][support] - decomposition["nonrigid_flow"][support]))
            err3 = np.max(np.abs(decomposition["total_3d"][support] -
                    decomposition["rigid_3d"][support] - decomposition["nonrigid_3d"][support]))
            row.update(pair_valid_count=int(support.sum()), additive_2d_max_error=float(err2),
                       additive_3d_max_error=float(err3),
                       median_world_residual=float(np.median(np.linalg.norm(decomposition["world_nonrigid"][support], axis=-1))))
            save_npz(output / "motion" / f"{index:06d}.npz", budget,
                     **{k: np.asarray(v, bool if k == "valid" else np.float32) for k, v in decomposition.items()},
                     frame_pair=np.array([index, index + 1], np.int32),
                     metadata=np.asarray(json.dumps({"units_3d": UNITS, "units_2d": "pixel",
                         "input_scene_flow": "camera0 residual after ego compensation",
                         "nonrigid_semantics": "all non-camera residual, including independent rigid objects",
                         "camera_delta_semantics": "coordinates in camera1 minus coordinates in camera0",
                         "world_nonrigid_semantics": "R_camera0_to_world @ sr_flow"})))
            transported = advect_labels(labels, decomposition["target_uv"], decomposition["target_depth"],
                following["depth"], support, relative_depth_tolerance=args.depth_tolerance,
                max_pixel_distance=args.pixel_tolerance)
            labels = transported["labels"]
            row["tracking"] = transported["stats"]
            current = following
        else:
            row["pair_valid_count"] = 0
            row["pair_status"] = "terminal_no_next_frame"
        rows.append(row)
        write_json(output / "build_progress.json", {"at": now(), "completed_frames": index + 1,
                                                    "frame_count": count, "last": row})
        print(f"{name} build {index + 1}/{count}: anchors={total_points} inherited_pixels={inherited}", flush=True)

    points = np.concatenate(points_list)
    colors = np.concatenate(colors_list)
    birth = np.concatenate(birth_list)
    save_npz(output / "model.npz", budget, points=points, colors=colors, birth_frame=birth,
             birth_pixel=np.concatenate(pixel_list), observation_count=np.asarray(observations, np.int32),
             birth_correction=np.concatenate(correction_list), correction_support=np.concatenate(donor_count_list),
             last_seen=np.asarray(last_seen, np.int32), units=np.asarray(UNITS),
             metric_calibrated=np.bool_(False), voxel_size=np.float64(voxel),
             model_stride=np.int32(stride), canonical_policy=np.asarray("immutable anchors with supported local inverse deformation at birth"))
    budget.check(points.nbytes + colors.nbytes + 1024)
    write_ply(output / "model.ply", points, colors)
    budget.saved(output / "model.ply")
    samples = np.concatenate(depth_samples)
    samples = samples[np.isfinite(samples) & (samples > 0)]
    depth_range = np.percentile(samples, [2, 98]).tolist()
    save_npz(output / "cameras.npz", budget, camera_to_world=np.asarray(cameras), K=np.asarray(intrinsics),
             frame_index=np.arange(count), source_index=np.asarray([r["source_index"] for r in sources.records]))
    sources.verify()
    manifest = {"schema_version": 1, "created_at": now(), "dataset": name,
        "frame_count": count, "pair_count": count - 1, "original_size": [h, w],
        "units": UNITS, "metric_calibrated": False, "reference_frame": 0,
        "canonical_policy": "immutable anchors; SR label transport and supported local inverse deformation at birth",
        "birth_correction_policy": {"offset_radii_pixels": [1, 2, 4, 8], "directions_per_radius": 8,
            "minimum_donors": 4, "max_relative_depth_difference": 0.05, "max_rgb_l2_difference": 80,
            "displacement": "component median of live_world - canonical_anchor at supported neighbors",
            "unsupported": "zero correction; explicitly first-observed shape"},
        "nonrigid_semantics": "non-camera residual (also includes independently moving rigid objects and estimation errors)",
        "late_surface_policy": "fixed at first observed shape; not claimed to recover unobserved frame0 geometry",
        "unmatched_policy": "new first-observed anchor on model sampling grid; may include tracking failures",
        "color_policy": "fixed first-observed RGB; no temporal appearance changes in virtual world",
        "voxel_size": voxel, "model_stride": stride, "model_point_count": len(points),
        "frame0_anchor_count": int((birth == 0).sum()), "later_anchor_count": int((birth > 0).sum()),
        "motion_corrected_anchor_count": int(np.count_nonzero(np.concatenate(donor_count_list))),
        "depth_range": depth_range, "depth_tolerance": args.depth_tolerance,
        "pixel_tolerance": args.pixel_tolerance, "frames": rows,
        "input_pins": list(sources.pins.values()), "source_pins": source_code_pins(),
        "source_verified_after_build": True, "argv": sys.argv,
        "model_sha256": digest(output / "model.npz"), "render_status": "pending"}
    write_json(output / "manifest.json", manifest)
    return manifest


class Video:
    def __init__(self, path, width, height, fps):
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "19", "preset": "fast"}

    def add(self, rgb):
        for packet in self.stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")):
            self.container.mux(packet)

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def color_depth(depth, valid, limits):
    from matplotlib import colormaps

    value = np.clip((np.nan_to_num(depth, nan=limits[0]) - limits[0]) / max(limits[1] - limits[0], 1e-9), 0, 1)
    rgb = (colormaps["turbo"](1 - value)[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def panel_image(source, virtual, depth_rgb, index, count, coverage):
    """A readable triptych, with no annotations in the numeric RGB-D arrays."""
    h, w = source.shape[:2]
    cell_width = 420
    cell_height = max(2, round(h * cell_width / w))
    canvas = Image.new("RGB", (cell_width * 3, cell_height + 70), (15, 17, 22))
    draw = ImageDraw.Draw(canvas)
    for j, (data, title) in enumerate(zip((source, virtual, depth_rgb),
                         ("Observed RGB", "SR canonical / frozen RGB", "Virtual depth / holes black"))):
        canvas.paste(Image.fromarray(data).resize((cell_width, cell_height)), (j * cell_width, 50))
        draw.text((j * cell_width + 8, 28), title, fill="white")
    draw.text((8, 8), f"Frame {index + 1}/{count} | camera path preserved | coverage {coverage:.1%} | {UNITS}", fill="white")
    arr = np.asarray(canvas)
    return arr[: arr.shape[0] // 2 * 2, : arr.shape[1] // 2 * 2].copy()


def render_dataset(name, sources, output, args, budget, manifest):
    import rerun as rr
    import rerun.blueprint as rrb

    model_path = output / "model.npz"
    for pin in manifest["input_pins"]:
        path = Path(pin["path"])
        if path.stat().st_size != pin["bytes"] or digest(path) != pin["sha256"]:
            raise RuntimeError(f"Build input changed before render: {path}")
    if digest(model_path) != manifest["model_sha256"]:
        raise RuntimeError("Canonical model changed since build")
    with np.load(model_path) as data:
        points, colors, birth = data["points"], data["colors"], data["birth_frame"]
    with np.load(output / "cameras.npz") as data:
        cameras, intrinsics = data["camera_to_world"], data["K"]
    h, w = manifest["original_size"]
    fps = args.fps or (5 if name == "grass" else 24)
    rr.init(f"sr_static_{name}", spawn=False)
    recording = output / f"{name}_static_world.rrd"
    rr.save(str(recording))
    rr.log("world", rr.ViewCoordinates.RDF, static=True)
    display_stride = max(1, int(np.ceil(len(points) / args.rerun_max_points)))
    rr.log("world/canonical", rr.Points3D(points[::display_stride], colors=colors[::display_stride],
                                         radii=manifest["voxel_size"] * 0.4), static=True)
    rr.log("world/camera_path", rr.LineStrips3D([cameras[:, :3, 3]], colors=[100, 165, 255]), static=True)
    blueprint = rrb.Blueprint(
        rrb.Horizontal(rrb.Spatial3DView(name="SR fixed world (first observed shapes)", origin="/world"),
            rrb.Vertical(rrb.Spatial2DView(name="Observed RGB", origin="/observed"),
                rrb.Spatial2DView(name="Virtual frozen RGB", origin="/world/camera/rgb"),
                rrb.Spatial2DView(name="Virtual camera-z depth", origin="/world/camera/depth")), column_shares=[2, 1]),
        auto_layout=False, auto_views=False, collapse_panels=True)
    rr.send_blueprint(blueprint)
    blueprint.save(f"sr_static_{name}", output / f"{name}_static_world.rbl")
    videos = []
    coverage_rows = []
    representatives = []
    try:
        for index in range(manifest["frame_count"]):
            frame = sources.frame(index)
            rendered = render_points(points, colors, intrinsics[index], cameras[index], (h, w), radius=args.splat_radius)
            valid = rendered["valid"]
            selected_ids = rendered["point_index"]
            reference_support = np.zeros((h, w), bool)
            reference_support[valid] = birth[selected_ids[valid]] == 0
            coverage = float(valid.mean())
            if not valid.any():
                raise RuntimeError(f"Virtual frame {index} has zero model coverage")
            path = output / "rgbd" / f"{index:06d}.npz"
            save_npz(path, budget, rgb=rendered["rgb"], depth=rendered["depth"], valid=valid,
                model_point_index=selected_ids.astype(np.int32), reference_support=reference_support,
                K=intrinsics[index], camera_to_world=cameras[index], frame_index=np.int32(index),
                source_index=np.int32(sources.records[index]["source_index"]),
                metadata=np.asarray(json.dumps({"units": UNITS, "metric_calibrated": False,
                    "depth_kind": "camera_z", "world_is_static": True, "radius_pixels": args.splat_radius,
                    "model_sha256": manifest["model_sha256"], "timestamp_seconds": index / fps})))
            depth_rgb = color_depth(rendered["depth"], valid, manifest["depth_range"])
            composite = panel_image(frame["rgb"], rendered["rgb"], depth_rgb, index, manifest["frame_count"], coverage)
            if not videos:
                videos = [Video(output / f"{name}_static_rgb.mp4", w, h, fps),
                          Video(output / f"{name}_static_depth.mp4", w, h, fps),
                          Video(output / f"{name}_static_rgbd.mp4", composite.shape[1], composite.shape[0], fps)]
            for video, data in zip(videos, (rendered["rgb"], depth_rgb, composite)):
                video.add(data)
            rr.set_time("frame", sequence=index)
            rr.log("world/camera", rr.Transform3D(translation=cameras[index, :3, 3], mat3x3=cameras[index, :3, :3]))
            rr.log("world/camera", rr.Pinhole(image_from_camera=intrinsics[index], resolution=[w, h]))
            rr.log("observed", rr.Image(frame["rgb"]).compress(jpeg_quality=88))
            rr.log("world/camera/rgb", rr.Image(rendered["rgb"]).compress(jpeg_quality=90))
            rr.log("world/camera/depth", rr.DepthImage(np.nan_to_num(rendered["depth"], nan=0)))
            rr.log("coverage", rr.Scalars([coverage]))
            coverage_rows.append({"frame_index": index, "valid_pixels": int(valid.sum()),
                "total_pixels": h * w, "coverage": coverage, "reference_supported_pixels": int(reference_support.sum())})
            if index in {0, manifest["frame_count"] // 2, manifest["frame_count"] - 1}:
                representatives.append(composite)
            print(f"{name} render {index + 1}/{manifest['frame_count']}: coverage={coverage:.3f}", flush=True)
    finally:
        for video in videos:
            video.close()
        rr.disconnect()
    Image.fromarray(np.concatenate(representatives, axis=0)).save(output / f"{name}_static_preview.png")
    if digest(model_path) != manifest["model_sha256"]:
        raise RuntimeError("Rendering modified the canonical model")
    sources.verify()
    manifest.update(render_status="complete", rendered_at=now(), fps=fps, splat_radius=args.splat_radius,
        splat_semantics="fixed pixel footprint, nearest camera-z, no inpainting", render_source_pins=source_code_pins(),
        rerun_display_stride=display_stride, coverage=coverage_rows, sources_verified_after_render=True,
        minimum_coverage=min(r["coverage"] for r in coverage_rows),
        mean_coverage=float(np.mean([r["coverage"] for r in coverage_rows])))
    artifacts = []
    for path in sorted(output.iterdir()):
        if path.suffix in {".mp4", ".png", ".ply", ".rrd", ".rbl", ".npz"}:
            artifacts.append({"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path)})
    manifest["artifacts"] = artifacts
    write_json(output / "manifest.json", manifest)
    budget.check()
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results/depth_detail_oct16"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/sr_static_oct16"))
    parser.add_argument("--dataset", choices=("grass", "railway", "all"), default="all")
    parser.add_argument("--stage", choices=("all", "build", "render"), default="all")
    parser.add_argument("--model-stride", type=int, default=2)
    parser.add_argument("--voxel-size", type=float, default=0, help="0: half a sampled-pixel footprint at median depth")
    parser.add_argument("--depth-tolerance", type=float, default=0.05)
    parser.add_argument("--pixel-tolerance", type=float, default=1.5)
    parser.add_argument("--splat-radius", type=int, default=2)
    parser.add_argument("--max-points", type=int, default=5000000)
    parser.add_argument("--rerun-max-points", type=int, default=250000)
    parser.add_argument("--fps", type=int, default=0, help="0 uses grass5 / railway24")
    parser.add_argument("--max-output-gib", type=float, default=30)
    parser.add_argument("--min-free-gib", type=float, default=35)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly regenerate this consumer's outputs")
    args = parser.parse_args(argv)
    if args.model_stride < 1 or args.voxel_size < 0 or args.max_points < 1 or args.rerun_max_points < 1:
        parser.error("stride and point limits must be positive, voxel size nonnegative")
    if not 0 <= args.splat_radius <= 3 or not 0 <= args.depth_tolerance <= 1 or not 0 <= args.pixel_tolerance <= 3:
        parser.error("invalid splat radius or correspondence tolerances")
    if args.fps < 0 or args.min_free_gib < 0 or args.max_output_gib < 0.01:
        parser.error("invalid fps/storage budget; output budget must be at least 0.01GiB")
    input_root, output_root = args.input_dir.resolve(), args.output_dir.resolve()
    if input_root == output_root or input_root in output_root.parents or output_root in input_root.parents:
        parser.error("Input and output trees must be disjoint")
    output_root.mkdir(parents=True, exist_ok=True)
    budget = Budget(output_root, int(args.max_output_gib * 1024**3), int(args.min_free_gib * 1024**3))
    budget.check()
    datasets = ("grass", "railway") if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        sources = Sources(input_root / dataset)
        output = output_root / dataset / "static_world"
        if args.stage != "render" and output.exists() and any(output.iterdir()) and not args.overwrite:
            parser.error(f"Existing output requires --overwrite: {output}")
        output.mkdir(parents=True, exist_ok=True)
        if args.stage in ("all", "build"):
            manifest = build_dataset(dataset, sources, output, args, budget)
        else:
            manifest = json.loads((output / "manifest.json").read_text())
            if manifest["render_status"] == "complete" and not args.overwrite:
                parser.error("Completed render requires --overwrite")
            if sources.frames["frame_count"] != manifest["frame_count"]:
                raise ValueError("Render input frame count differs from built model")
        if args.stage in ("all", "render"):
            render_dataset(dataset, sources, output, args, budget, manifest)
        print(f"{dataset}: {output / 'manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
