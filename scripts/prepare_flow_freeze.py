#!/usr/bin/env python3
"""Build an appearance-preserving, SR-guided frozen-plant draft for masked completion.

Continuous canonical coordinates are transported through observed total motion.
Unknown surfaces are never reborn as frame-zero geometry. This is a provisional
draft, not a recovered static-world ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_bullettime_controls import source_map, write_video  # noqa: E402
from arrow.static_world import unproject_depth, transform_points  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dilate(mask, radius):
    """Square binary dilation using an integral image, without SciPy/OpenCV."""
    mask = np.asarray(mask, dtype=bool)
    if radius == 0:
        return mask.copy()
    if radius < 0:
        raise ValueError("Negative morphology radius")
    padded = np.pad(mask.astype(np.int32), radius)
    integral = np.pad(padded, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    k = 2 * radius + 1
    sums = integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k]
    return sums > 0


def erode(mask, radius):
    # Outside-image space is unknown, so erosion removes the image border too.
    inside = ~dilate(~np.asarray(mask, bool), radius)
    if radius:
        inside[:radius] = inside[-radius:] = False
        inside[:, :radius] = inside[:, -radius:] = False
    return inside


def foreground_seed(rgb):
    """Observed green pixels only; no morphology-expanded road halo is included."""
    r, g, b = np.moveaxis(rgb.astype(np.float32), -1, 0)
    green = (g > 55) & (g > r * 1.035) & (g > b * 1.10) & ((g - b) > 12)
    h = rgb.shape[0]
    green[:int(.08 * h)] = False
    green[int(.985 * h):] = False
    return green


def foreground_candidate(rgb):
    """Conservative green/stem editing candidate; includes nearby vegetation.

    Pale/dry stems and non-green objects are not guaranteed to be segmented.
    Morphology expands the unknown editing footprint, not reference color trust.
    """
    green = foreground_seed(rgb)
    closed = erode(dilate(green, 10), 10)
    return dilate(closed | green, 6)


def project_world(world, camera_to_world, K):
    inverse = np.linalg.inv(np.asarray(camera_to_world, np.float64))
    q = np.asarray(world, np.float64) @ inverse[:3, :3].T + inverse[:3, 3]
    projected = q @ np.asarray(K, np.float64).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = projected[..., :2] / projected[..., 2, None]
    return uv, q[..., 2]


def total_endpoint(depth, K0, K1, M0, M1, residual, valid):
    """Qt = inv(M1) M0 (unproject(depth,K0) + ego-compensated SR residual)."""
    P = unproject_depth(depth, K0)
    X = np.linalg.inv(np.asarray(M1, np.float64)) @ np.asarray(M0, np.float64)
    Q = (P.astype(np.float64) + residual) @ X[:3, :3].T + X[:3, 3]
    q = Q @ np.asarray(K1, np.float64).T
    with np.errstate(invalid="ignore", divide="ignore"):
        uv = q[..., :2] / q[..., 2, None]
    good = valid & np.isfinite(uv).all(-1) & np.isfinite(Q).all(-1) & (Q[..., 2] > 0)
    return uv, Q[..., 2], good


def splat_front(attributes, uv, z, valid, output_shape, *, observed_depth=None,
                depth_tolerance=.05, layer_tolerance=.002, minimum_mass=.25):
    """Bilinear forward splat restricted to a front-depth layer.

    Four mathematical bilinear corners only; no disk expansion or hole filling.
    The observed-depth gate runs before visibility and attribute interpolation.
    Attributes are continuous values, never shared/discrete anchor IDs.
    """
    attributes = np.asarray(attributes, np.float32)
    uv = np.asarray(uv, np.float64)
    z = np.asarray(z, np.float64)
    valid = np.asarray(valid, bool)
    if (attributes.shape[:-1] != z.shape or uv.shape != z.shape + (2,)
            or valid.shape != z.shape or depth_tolerance < 0 or layer_tolerance < 0):
        raise ValueError("Incompatible splat shapes or tolerances")
    h, w = output_shape
    if min(h, w) <= 0:
        raise ValueError("Invalid output dimensions")
    if observed_depth is not None and np.shape(observed_depth) != (h, w):
        raise ValueError("Observed depth shape mismatch")
    good = valid & np.isfinite(attributes).all(-1) & np.isfinite(uv).all(-1) & np.isfinite(z) & (z > 0)
    good &= (uv[..., 0] > -1) & (uv[..., 0] < w) & (uv[..., 1] > -1) & (uv[..., 1] < h)
    source_ids = np.flatnonzero(good)
    points = uv.reshape(-1, 2)[source_ids]
    depths = z.ravel()[source_ids]
    values = attributes.reshape(-1, attributes.shape[-1])[source_ids]
    base = np.floor(points).astype(np.int64)
    frac = points - base
    ids_list, weights_list, source_list = [], [], []
    for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
        x, y = base[:, 0] + dx, base[:, 1] + dy
        weight = (frac[:, 0] if dx else 1 - frac[:, 0]) * (frac[:, 1] if dy else 1 - frac[:, 1])
        keep = (x >= 0) & (x < w) & (y >= 0) & (y < h) & (weight > 1e-8)
        select = np.flatnonzero(keep)
        dest = y[select] * w + x[select]
        if observed_depth is not None:
            measured = np.asarray(observed_depth).ravel()[dest]
            predicted = depths[select]
            coherent = (np.isfinite(measured) & (measured > 0)
                        & (np.abs(measured - predicted) <= depth_tolerance * np.maximum(measured, predicted)))
            select, dest = select[coherent], dest[coherent]
        ids_list.append(dest)
        weights_list.append(weight[select])
        source_list.append(select)
    ids = np.concatenate(ids_list)
    weights = np.concatenate(weights_list)
    selects = np.concatenate(source_list)
    zmin = np.full(h * w, np.inf, dtype=np.float64)
    np.minimum.at(zmin, ids, depths[selects])
    front = depths[selects] <= zmin[ids] * (1 + layer_tolerance)
    ids, weights, selects = ids[front], weights[front], selects[front]
    mass = np.bincount(ids, weights=weights, minlength=h * w)
    supported = mass >= minimum_mass
    result = np.full((h * w, attributes.shape[-1]), np.nan, dtype=np.float32)
    for channel in range(attributes.shape[-1]):
        sums = np.bincount(ids, weights=weights * values[selects, channel], minlength=h * w)
        result[supported, channel] = (sums[supported] / mass[supported]).astype(np.float32)
    return (result.reshape(h, w, -1), supported.reshape(h, w),
            mass.reshape(h, w).astype(np.float32), zmin.reshape(h, w).astype(np.float32))


def transport_canonical(field, foreground_probability, confidence, endpoint, endpoint_depth,
                        valid, following_depth):
    attributes = np.concatenate([field, foreground_probability[..., None], confidence[..., None]], axis=-1)
    splat, supported, mass, _ = splat_front(
        attributes, endpoint, endpoint_depth,
        valid & np.isfinite(field).all(-1) & (confidence >= .35), following_depth.shape,
        observed_depth=following_depth, depth_tolerance=.05, layer_tolerance=.005,
        minimum_mass=.25)
    # Bilinear mass is image-area coverage, not a transition probability. It is
    # already gated above; multiplying it at every pair invents exponential
    # confidence loss even for a perfectly observed fractional-pixel track.
    # Carry the normalized inherited quality through the binary depth/front
    # gates. This support value is not a calibrated correctness probability.
    following_confidence = np.nan_to_num(splat[..., 4])
    supported &= following_confidence >= .35
    next_field = np.where(supported[..., None], splat[..., :3], np.nan).astype(np.float32)
    next_foreground = np.where(supported, splat[..., 3], 0).astype(np.float32)
    following_confidence[~supported] = 0
    return next_field, next_foreground, following_confidence.astype(np.float32)


def map_mask_native(mask, K_native, K_common, shape):
    x, y = source_map(K_native, K_common, shape)
    h, w = mask.shape
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    output = np.zeros(shape, bool)
    for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
        xx, yy = x0 + dx, y0 + dy
        inside = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        output[inside] |= mask[yy[inside], xx[inside]]
    return output


def sample_depth_safe(depth, K_native, K_common, shape):
    """Ray-preserving depth sample; reject neighborhoods crossing a >5% depth edge."""
    x, y = source_map(K_native, K_common, shape)
    x = np.clip(x, 0, depth.shape[1] - 1)
    y = np.clip(y, 0, depth.shape[0] - 1)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, depth.shape[1] - 1), np.minimum(y0 + 1, depth.shape[0] - 1)
    values = np.stack([depth[y0, x0], depth[y0, x1], depth[y1, x0], depth[y1, x1]])
    valid = np.isfinite(values).all(0) & (values > 0).all(0)
    valid &= values.max(0) - values.min(0) <= .05 * values.max(0)
    fx, fy = x - x0, y - y0
    result = (values[0] * (1 - fx) * (1 - fy) + values[1] * fx * (1 - fy)
              + values[2] * (1 - fx) * fy + values[3] * fx * fy)
    return np.where(valid, result, np.nan).astype(np.float32)


def projected_footprint(initial_world, initial_mask, M, K, shape):
    points = initial_world[initial_mask & np.isfinite(initial_world).all(-1)]
    uv, z = project_world(points, M, K)
    _, valid, _, _ = splat_front(np.ones((len(points), 1), np.float32), uv, z,
                                np.ones(len(points), bool), shape, minimum_mass=1e-8)
    return dilate(valid, 2)


def reference_visibility_depth(initial_world, M, K, shape):
    """Front depth of all observed frame-zero surfaces, used only for visibility."""
    uv, z = project_world(initial_world, M, K)
    _, _, _, front = splat_front(np.ones(initial_world.shape[:2] + (1,), np.float32),
                                uv, z, np.isfinite(initial_world).all(-1), shape,
                                minimum_mass=1e-8)
    return front


def raster_reference(rgb, initial_world, initial_depth, seed, M, K, shape):
    """Rasterize observed adjacent-pixel triangles, with perspective-correct UV.

    All valid surfaces participate in visibility. A visible triangle supplies
    foreground color only if all three original vertices are actual green seed.
    Reject source-depth jumps, foldovers, and excessive projected edge lengths;
    neither morphology nor a splat radius creates reference foreground support.
    """
    h, w = rgb.shape[:2]
    oh, ow = shape
    if (initial_world.shape != (h, w, 3) or initial_depth.shape != (h, w)
            or seed.shape != (h, w) or min(h, w, oh, ow) < 2):
        raise ValueError("Incompatible reference raster shapes")
    projected, z = project_world(initial_world, M, K)
    uv = projected.reshape(-1, 2)
    z = z.ravel()
    src_depth = initial_depth.ravel()
    seed_flat = seed.ravel()
    front = np.full(oh * ow, np.inf, np.float64)
    source_uv = np.full((oh * ow, 2), np.nan, np.float64)
    foreground = np.zeros(oh * ow, bool)
    metrics = {"triangles_total": 2 * (h - 1) * (w - 1), "triangles_accepted": 0,
               "raster_samples": 0, "depth_relative_limit": .05, "max_projected_edge_px": 8}
    for row in range(0, h - 1, 64):
        yy, xx = np.indices((min(64, h - 1 - row), w - 1))
        a = ((yy + row) * w + xx).ravel()
        vertices = np.concatenate([np.stack([a, a + 1, a + w + 1], 1),
                                   np.stack([a, a + w + 1, a + w], 1)])
        points = uv[vertices]
        depths = z[vertices]
        original_depths = src_depth[vertices]
        e1, e2 = points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
        area = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        edge_squared = np.stack([(e1 ** 2).sum(1), (e2 ** 2).sum(1), ((e2 - e1) ** 2).sum(1)], 1)
        good = (np.isfinite(points).all((1, 2)) & np.isfinite(depths).all(1) & (depths > 0).all(1)
                & np.isfinite(original_depths).all(1) & (original_depths > 0).all(1)
                & (original_depths.max(1) - original_depths.min(1) <= .05 * original_depths.max(1))
                & (area > 1e-8) & (edge_squared.max(1) <= 64))
        vertices, points, depths, area = vertices[good], points[good], depths[good], area[good]
        metrics["triangles_accepted"] += len(vertices)
        if not len(vertices):
            continue
        lo = np.ceil(points.min(1) - 1e-7).astype(np.int64)
        hi = np.floor(points.max(1) + 1e-7).astype(np.int64)
        lo = np.maximum(lo, [0, 0])
        hi = np.minimum(hi, [ow - 1, oh - 1])
        size = np.maximum(hi - lo + 1, 0)
        count = size[:, 0] * size[:, 1]
        tri = np.repeat(np.arange(len(vertices)), count)
        if not len(tri):
            continue
        offset = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
        x = lo[tri, 0] + offset % size[tri, 0]
        y = lo[tri, 1] + offset // size[tri, 0]
        q = np.stack([x, y], 1) - points[tri, 0]
        e1 = points[tri, 1] - points[tri, 0]
        e2 = points[tri, 2] - points[tri, 0]
        b1 = (q[:, 0] * e2[:, 1] - q[:, 1] * e2[:, 0]) / area[tri]
        b2 = (e1[:, 0] * q[:, 1] - e1[:, 1] * q[:, 0]) / area[tri]
        bary = np.stack([1 - b1 - b2, b1, b2], 1)
        inside = (bary >= -1e-7).all(1)
        tri, bary, x, y = tri[inside], bary[inside], x[inside], y[inside]
        perspective = bary / depths[tri]
        sample_depth = 1 / perspective.sum(1)
        perspective *= sample_depth[:, None]
        ids = y * ow + x
        metrics["raster_samples"] += len(ids)
        # Deterministic closest sample per pixel, then compare with earlier rows.
        order = np.lexsort((tri, sample_depth, ids))
        ids_sorted = ids[order]
        first = np.r_[True, ids_sorted[1:] != ids_sorted[:-1]] if len(order) else np.zeros(0, bool)
        chosen = order[first]
        chosen = chosen[sample_depth[chosen] < front[ids[chosen]]]
        dest = ids[chosen]
        vert = vertices[tri[chosen]]
        # The texture is a projected source-camera image, not an affine UV
        # parameterization of the 3D plane. Interpolate source homogeneous
        # (u*z_source, v*z_source, z_source), then perform its perspective divide.
        source_weights = perspective[chosen] * src_depth[vert]
        source_weights /= source_weights.sum(1)[:, None]
        source_uv[dest, 0] = (source_weights * (vert % w)).sum(1)
        source_uv[dest, 1] = (source_weights * (vert // w)).sum(1)
        front[dest] = sample_depth[chosen]
        foreground[dest] = seed_flat[vert].all(1)
    known = foreground & np.isfinite(source_uv).all(1)
    output = np.zeros((oh * ow, 3), np.uint8)
    xy = np.clip(source_uv[known], [0, 0], [w - 1, h - 1])
    x0, y0 = np.floor(xy).astype(np.int64).T
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (xy - np.floor(xy)).T
    weights = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], 1)
    texture_seed = np.stack([seed[y0, x0], seed[y0, x1], seed[y1, x0], seed[y1, x1]], 1)
    texture_trusted = ((weights <= 1e-8) | texture_seed).all(1)
    colors = np.stack([rgb[y0, x0], rgb[y0, x1], rgb[y1, x0], rgb[y1, x1]], 1)
    color = (colors * weights[..., None]).sum(1)
    destinations = np.flatnonzero(known)
    output[destinations[texture_trusted]] = np.rint(color[texture_trusted]).clip(0, 255).astype(np.uint8)
    known[destinations[~texture_trusted]] = False
    source_uv[~known] = np.nan
    metrics["visible_surface_fraction"] = float(np.isfinite(front).mean())
    metrics["foreground_fraction"] = float(known.mean())
    return (output.reshape(oh, ow, 3), known.reshape(oh, ow), known.reshape(oh, ow).astype(np.float32),
            source_uv.reshape(oh, ow, 2).astype(np.float32), front.reshape(oh, ow).astype(np.float32), metrics)


def render_foreground(rgb, field, foreground_probability, confidence, current_mask, M, K, shape):
    uv, z = project_world(field, M, K)
    yy, xx = np.indices(rgb.shape[:2], dtype=np.float32)
    colors = rgb.astype(np.float32)
    attributes = np.concatenate([colors, colors ** 2, xx[..., None], yy[..., None], confidence[..., None]], axis=-1)
    good = (np.isfinite(field).all(-1) & (foreground_probability >= .95)
            & (confidence >= .5) & current_mask)
    values, supported, mass, depth = splat_front(attributes, uv, z, good, shape,
                                           layer_tolerance=.002, minimum_mass=.5)
    mean = values[..., :3]
    variance = np.maximum(values[..., 3:6] - mean ** 2, 0)
    # Mixed leaf/road or strong RGB boundaries remain explicitly unknown.
    supported &= np.isfinite(variance).all(-1) & (np.sqrt(variance).max(-1) <= 28)
    output = np.rint(np.nan_to_num(mean)).clip(0, 255).astype(np.uint8)
    conf = np.nan_to_num(values[..., 8]) * np.minimum(mass, 1)
    conf[~supported] = 0
    source_uv = np.where(supported[..., None], values[..., 6:8], np.nan).astype(np.float32)
    return output, supported, conf.astype(np.float32), source_uv, depth


def compose_draft(source, warped, warped_valid, source_foreground, destination_foreground, *, reference=False):
    union = dilate(source_foreground | destination_foreground, 3)
    source_preserved = ~union
    warped_known = warped_valid & erode(union, 2)
    if reference:
        source_preserved = np.ones(union.shape, bool)
        warped_known = np.zeros(union.shape, bool)
    unknown = ~(source_preserved | warped_known)
    draft = np.full_like(source, 127)
    draft[source_preserved] = source[source_preserved]
    draft[warped_known] = warped[warped_known]
    return draft, unknown, source_preserved, warped_known, union


def build(input_dir, rectified_controls, output_dir):
    started = time.monotonic()
    input_dir, rectified_controls, output_dir = map(Path, (input_dir, rectified_controls, output_dir))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output must be empty")
    with np.load(rectified_controls, allow_pickle=False) as data:
        source_rgb = data["rgb"].copy()
        Ks = data["original_K"].copy()
        Ms = data["camera_to_world"].copy()
        common = data["K"].copy()
        native_indices = data["native_indices"].copy()
        fps = float(data["fps"])
    count, oh, ow, _ = source_rgb.shape
    if count != 17 or not np.array_equal(native_indices, np.arange(17)):
        raise ValueError("This bounded pilot requires original frames 0..16")
    if not np.array_equal(common, np.repeat(common[:1], count, axis=0)):
        raise ValueError("Rectification controls need constant K")
    frame_path = input_dir / "frames.json"
    records = json.loads(frame_path.read_text())["frames"][:count]
    pins = {str(frame_path.resolve()): digest(frame_path), str(rectified_controls.resolve()): digest(rectified_controls)}

    def pin(path):
        pins[str(path.resolve())] = digest(path)

    def load(index):
        rgb_path = Path(records[index]["rgb_path"])
        if not rgb_path.is_absolute():
            rgb_path = input_dir / rgb_path
        depth_path = input_dir / "stages" / "final" / f"{index:06d}.npz"
        geometry_path = input_dir / "stages" / "geometry" / f"{index:06d}.npz"
        for path in (rgb_path, depth_path, geometry_path):
            pin(path)
        if records[index].get("rgb_sha256") != pins[str(rgb_path.resolve())]:
            raise ValueError(f"RGB hash mismatch {index}")
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        with np.load(depth_path, allow_pickle=False) as data:
            valid = data["final_valid"] & np.isfinite(data["final_depth"]) & (data["final_depth"] > 0)
            depth = np.where(valid, data["final_depth"], np.nan).astype(np.float32)
        with np.load(geometry_path, allow_pickle=False) as data:
            if not np.array_equal(data["arrow_K"], Ks[index]) or not np.array_equal(data["arrow_camera_to_metric"], Ms[index]):
                raise ValueError("Original geometry differs from rectification controls")
        return rgb, depth, valid

    rgb, depth, depth_valid = load(0)
    initial_world = transform_points(unproject_depth(depth, Ks[0]), Ms[0])
    initial_depth = depth.copy()
    initial_rgb = rgb.copy()
    initial_mask = foreground_candidate(rgb)
    initial_seed = foreground_seed(rgb)
    field = initial_world.copy()
    probability = initial_seed.astype(np.float32)
    confidence = depth_valid.astype(np.float32)
    buffers = {key: [] for key in ("draft_rgb", "edit_mask", "source_preserved_mask", "warped_known_mask",
                                  "foreground_source_mask", "foreground_destination_mask", "foreground_union_mask",
                                  "warp_confidence", "source_uv", "canonical_depth", "geometry_valid",
                                  "observed_background_depth", "reference_projected_mask",
                                  "reference_projected_confidence", "source_uv_frame_index")}
    statistics = []
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_valid = np.isfinite(initial_world).all(-1)
    np.savez_compressed(output_dir / "reference_model.npz", points=initial_world[initial_valid],
                        colors=rgb[initial_valid], native_uv=np.moveaxis(np.indices(depth.shape)[::-1], 0, -1)[initial_valid],
                        foreground_candidate=initial_mask[initial_valid], units=np.array("arrow_provisional"),
                        foreground_seed=initial_seed[initial_valid],
                        reference_frame=np.array(0), metric_calibrated=np.array(False))
    for index in range(count):
        current_mask = foreground_candidate(rgb)
        source_mask = map_mask_native(current_mask, Ks[index], common[index], (oh, ow))
        destination_mask = projected_footprint(initial_world, initial_mask, Ms[index], common[index], (oh, ow))
        warped, supported, warp_conf, source_uv, canonical_depth = render_foreground(
            rgb, field, probability, confidence, current_mask, Ms[index], common[index], (oh, ow))
        reference_rgb, reference_supported, reference_conf, reference_uv, reference_depth, raster_metrics = raster_reference(
            initial_rgb, initial_world, initial_depth, initial_seed, Ms[index], common[index], (oh, ow))
        supported &= canonical_depth <= reference_depth * 1.002
        # A fixed-world appearance legitimately keeps observed frame-zero colors.
        # This is separate provenance, not invented confidence in a lost SR track.
        combined_rgb = np.where(reference_supported[..., None], reference_rgb, warped)
        draft, edit, preserved, known, union = compose_draft(
            source_rgb[index], combined_rgb, supported | reference_supported,
            source_mask, destination_mask, reference=index == 0)
        reference_known = known & reference_supported
        warped_known = known & ~reference_known
        warp_conf[~warped_known] = 0
        reference_conf[~reference_known] = 0
        source_uv = np.where(reference_known[..., None], reference_uv, source_uv)
        source_uv[~known] = np.nan
        source_uv_frame = np.full((oh, ow), -1, np.int16)
        source_uv_frame[warped_known] = index
        source_uv_frame[reference_known] = 0
        observed_depth = sample_depth_safe(depth, Ks[index], common[index], (oh, ow))
        canonical_depth = np.where(reference_known, reference_depth, canonical_depth)
        canonical_depth = (observed_depth.copy() if index == 0 else
                           np.where(known, canonical_depth, np.nan).astype(np.float32))
        geometry_valid = np.isfinite(canonical_depth) & ~edit
        canonical_depth[~geometry_valid] = np.nan
        observed_background_depth = np.where(preserved & ~union, observed_depth, np.nan).astype(np.float32)
        for key, value in zip(buffers, (draft, edit, preserved, warped_known, source_mask, destination_mask,
                                       union, warp_conf, source_uv, canonical_depth, geometry_valid,
                                       observed_background_depth, reference_known, reference_conf,
                                       source_uv_frame), strict=True):
            buffers[key].append(value)
        row = {"frame": index, "canonical_supported_fraction_native": float((confidence > 0).mean()),
               "source_foreground_fraction": float(source_mask.mean()),
               "destination_foreground_fraction": float(destination_mask.mean()),
               "source_preserved_fraction": float(preserved.mean()), "warped_known_fraction": float(warped_known.mean()),
               "reference_projected_fraction": float(reference_known.mean()),
               "edit_fraction": float(edit.mean()), "reference_exact": index == 0,
               "reference_triangle_raster": raster_metrics}
        statistics.append(row)
        print(json.dumps(row), flush=True)
        if index < count - 1:
            next_rgb, next_depth, next_valid = load(index + 1)
            path = input_dir / "stages" / "sr" / f"{index:06d}.npz"
            pin(path)
            with np.load(path, allow_pickle=False) as sr:
                if not np.array_equal(sr["sr_pair_index"], [index, index + 1]):
                    raise ValueError("Scene-flow pair mismatch")
                endpoint, endpoint_z, endpoint_valid = total_endpoint(
                    depth, Ks[index], Ks[index + 1], Ms[index], Ms[index + 1],
                    sr["sr_flow"], sr["sr_valid"] & depth_valid)
            field, probability, confidence = transport_canonical(
                field, probability, confidence, endpoint, endpoint_z, endpoint_valid, next_depth)
            rgb, depth, depth_valid = next_rgb, next_depth, next_valid
    arrays: dict[str, Any] = {key: np.stack(value) for key, value in buffers.items()}
    assert np.array_equal(arrays["draft_rgb"][0], source_rgb[0])
    assert np.all(arrays["draft_rgb"][arrays["edit_mask"]] == 127)
    assert np.array_equal(arrays["draft_rgb"][arrays["source_preserved_mask"]], source_rgb[arrays["source_preserved_mask"]])
    manifest = {
        "status": "draft_prepared_not_quality_accepted", "frames": count, "shape_hw": [oh, ow], "fps": fps,
        "input_dir": str(input_dir.resolve()), "rectified_controls": str(rectified_controls.resolve()),
        "units": "arrow_provisional", "pose_quality": "original provisional ARROW; no PnP correction",
        "reference_frame": 0, "method": "Reference-appearance foreground projection plus continuous SR correspondence support. "
        "Dense actual frame0 foreground RGB/XYZ is triangle-rasterized under each current original camera, without requiring surviving tracks. "
        "Source-grid triangles reject >5% source-depth jumps, foldover/degenerate area, and projected edges >8px. "
        "All surfaces compete in an exact front-depth buffer; trusted foreground triangles require all three actual green-seed vertices. "
        "Source-camera homogeneous coordinates are interpolated with target inverse-depth correction before source perspective divide and RGB bilinear sampling. "
        "Every positive-weight bilinear source-color donor must also be an actual green-seed pixel, including the fourth quad corner. "
        "Its provenance is reference_projected_mask; no road halo from mask morphology is trusted as reference color. "
        "Continuous native canonical XYZ field is also transported through total SR endpoints by "
        "bilinear front-layer splat; matched current foreground colors rendered at canonical points under current original camera. "
        "Reference-projected color has priority; surviving current matches fill only remaining supported pixels. "
        "No shared anchor IDs, no negated-flow shortcut, no new canonical births.",
        "reference_visibility": "Nearest valid reference triangle supplies the front surface; current-warp depth must be at most "
                                "0.2% behind it. Missing reference triangles do not certify unknown-surface visibility.",
        "equations": "P=depth*K_t^-1*[u,v,1]; Q_next=inv(M_next)*M_t*(P+sr_flow); endpoint=project(K_next,Q_next). "
        "C_0=M_0*P_0; transport(C_t) along endpoint; output_uv=project(K_common,inv(M_t)*C_t).",
        "thresholds": {"transport_depth_relative": .05, "transport_front_layer_relative": .005,
                       "render_front_layer_relative": .002, "transport_min_mass": .25,
                       "transport_min_confidence": .35, "render_min_confidence": .5,
                       "reference_foreground_probability": .95, "render_min_mass": .5,
                       "render_max_color_std": 28, "reference_triangle_depth_relative": .05,
                       "reference_triangle_max_edge_px": 8, "union_dilate_model_px": 3,
                       "unknown_boundary_ring_model_px": 2},
        "confidence_semantics": "Transport quality is normalized inherited donor support after binary depth/visibility gates. "
                                "Bilinear mass is a per-step minimum-coverage gate, never multiplied cumulatively. "
                                "It is not a calibrated correspondence-accuracy probability; rendering additionally uses local coverage/color consistency.",
        "foreground_mask": "RGB green candidate g>55,g>1.035r,g>1.10b,g-b>12, y in[.08H,.985H); "
                           "native closing radius10,dilation6. Includes nearby green vegetation in ROI; dry/pale stems may escape.",
        "destination_mask": "Project ALL initial valid foreground points under each current camera regardless of live tracking/visibility; "
                            "four-corner footprint then dilation2. Union with current source footprint, dilation3.",
        "known_policy": "Outside foreground union, source RGB preserved byte-for-byte. Inside, only supported warped foreground; "
                        "reference projection has priority and current-warp is separate. Remaining pixels gray127/edit=true. "
                        "frame0 is explicit exact-reference exception. source_uv_frame_index distinguishes reference0/current colors; -1 elsewhere.",
        "depth_contract": "canonical_depth is front canonical camera-z only at geometry_valid warped-known pixels; "
                          "frame0 uses reference source depth with a 5% neighbor-edge rejection. Unknown/edit pixels remain NaN. "
                          "observed_background_depth is separately sampled CURRENT observed depth only outside the editing union; "
                          "it is not claimed to be the canonical model. reference_model.npz stores frame0 dense points/colors once.",
        "limitations": ["Reference triangles approximate observed surfaces between adjacent pixels; depth uncertainty can distort leaf shape.",
                        "Tracking, geometry and masks are estimates; freeze correctness is unverified.",
                        "Unsupported/disoccluded/new leaf surfaces stay unknown for generation.",
                        "Background identity preservation assumes outside-mask pixels are static; this is not semantic ground truth.",
                        "Color can vary with current observations; this freezes supported geometry, not illumination."],
        "statistics": statistics, "input_sha256": pins,
        "script_sha256": digest(__file__), "control_helper_sha256": digest(Path(__file__).with_name("prepare_bullettime_controls.py")),
    }
    np.savez_compressed(output_dir / "controls.npz", **arrays, source_rgb=source_rgb,
                        reference_rgb=source_rgb[0], fps=np.array(fps), K=common, camera_to_world=Ms,
                        native_indices=native_indices, original_K=Ks, known_mask=~arrays["edit_mask"],
                        metadata=np.array(json.dumps(manifest, allow_nan=False)))
    write_video(output_dir / "draft.mp4", arrays["draft_rgb"], fps)
    mask_rgb = np.repeat(arrays["edit_mask"][..., None], 3, axis=-1).astype(np.uint8) * 255
    write_video(output_dir / "edit_mask.mp4", mask_rgb, fps)
    comparison = np.concatenate([source_rgb, arrays["draft_rgb"], mask_rgb], axis=2)
    write_video(output_dir / "comparison.mp4", comparison, fps)
    chosen = [0, 4, 8, 12, 16]
    board = Image.new("RGB", (ow * 3 // 2, (oh // 2 + 24) * len(chosen)), "#202020")
    draw = ImageDraw.Draw(board)
    for row_index, index in enumerate(chosen):
        image = Image.fromarray(comparison[index]).resize((ow * 3 // 2, oh // 2), Image.Resampling.BILINEAR)
        y = row_index * (oh // 2 + 24)
        draw.text((4, y + 5), f"{index}: source | warped draft | unknown white", fill="white")
        board.paste(image, (0, y + 24))
    board.save(output_dir / "preview.png")
    for path, expected in pins.items():
        if digest(path) != expected:
            raise RuntimeError(f"Input changed: {path}")
    manifest["source_verified_after_build"] = True
    manifest["wall_seconds"] = time.monotonic() - started
    manifest["artifacts"] = {p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
                             for p in output_dir.iterdir() if p.is_file()}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--rectified-controls", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.input_dir, args.rectified_controls, args.output_dir)
    print(json.dumps({"status": result["status"], "wall_seconds": result["wall_seconds"]}))


if __name__ == "__main__":
    main()
