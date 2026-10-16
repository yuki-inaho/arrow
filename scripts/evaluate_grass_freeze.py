#!/usr/bin/env python3
"""CPU diagnostics and unmodified-pixel comparisons; never a quality acceptance gate.

Requires cv2 (SIFT), PIL, NumPy and ffmpeg/ffprobe on PATH.
Masks are tailored to this grass clip and must be reviewed for other scenes. Generated inputs are raw
PNG frames. Only the legacy native image is fitted to a display tile; source and
model images are shown at their exact common resolution. All scores use raw data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

cv2 = importlib.import_module("cv2")

BG_BOXES = ((0.65, 0.04, 0.98, 0.22), (0.88, 0.37, 0.98, 0.56),
            (0.02, 0.78, 0.25, 0.98))
GRASS_BOX = (0.02, 0.12, 0.99, 0.91)
MIN_MATCHES = 12
MIN_INLIERS = 8


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def stats(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "median": None, "p90": None, "mean": None}
    return {"count": int(len(values)), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)), "mean": float(np.mean(values))}


def load_frames(folder, expected):
    folder = Path(folder)
    actual = sorted(folder.glob("*.png"))
    desired = [folder / f"frame_{index:04d}.png" for index in range(expected)]
    if actual != desired:
        raise ValueError(f"Exact PNG sequence required at {folder}: expected {expected}, got {len(actual)}")
    arrays, records = [], []
    for path in actual:
        with Image.open(path) as image:
            if image.mode != "RGB":
                raise ValueError(f"RGB PNG required without implicit conversion: {path}: {image.mode}")
            array = np.asarray(image).copy()
        arrays.append(array)
        records.append({"path": str(path.resolve()), "file_sha256": digest(path),
                        "raw_rgb_sha256": array_digest(array)})
    return np.stack(arrays), records


def ssim_rgb(a, b):
    """Classic 11x11 Gaussian population SSIM, RGB channel mean, 5px border removed."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    if min(a.shape[:2]) < 11:
        raise ValueError("SSIM requires dimensions >= 11")
    def blur(x):
        return cv2.GaussianBlur(x, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT)
    ma, mb = blur(a), blur(b)
    va = np.maximum(blur(a * a) - ma * ma, 0)
    vb = np.maximum(blur(b * b) - mb * mb, 0)
    cov = blur(a * b) - ma * mb
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    value = ((2 * ma * mb + c1) * (2 * cov + c2)
             / ((ma * ma + mb * mb + c1) * (va + vb + c2)))
    return float(np.mean(value[5:-5, 5:-5]))


def box_mask(shape, boxes):
    height, width = shape
    result = np.zeros(shape, dtype=bool)
    for x0, y0, x1, y1 in boxes:
        result[int(y0 * height):int(y1 * height), int(x0 * width):int(x1 * width)] = True
    return result


def masks(source_rgb):
    """Only source determines masks, so generated artifacts cannot choose their ROI."""
    hsv = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2HSV)
    green = ((hsv[..., 0] >= 24) & (hsv[..., 0] <= 95)
             & (hsv[..., 1] >= 45) & (hsv[..., 2] >= 30))
    background = box_mask(source_rgb.shape[:2], BG_BOXES)
    exclude = cv2.dilate(green.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    background &= ~exclude
    grass = green & box_mask(source_rgb.shape[:2], [GRASS_BOX])
    grass = cv2.erode(grass.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return background, grass


def fit_background_homography(gray0, gray1, bg0, bg1):
    sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.02)
    k0, d0 = sift.detectAndCompute(gray0, bg0.astype(np.uint8) * 255)
    k1, d1 = sift.detectAndCompute(gray1, bg1.astype(np.uint8) * 255)
    result = {"keypoints_0": len(k0), "keypoints_1": len(k1), "ratio_matches": 0,
              "inliers": 0, "status": "insufficient_matches"}
    if d0 is None or d1 is None or len(d1) < 2:
        return None, result
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(d0, d1, k=2)
    backward = matcher.knnMatch(d1, d0, k=2)
    rev = {pair[0].queryIdx: pair[0].trainIdx for pair in backward
           if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance}
    matches = [pair[0] for pair in forward if len(pair) == 2
               and pair[0].distance < 0.7 * pair[1].distance
               and rev.get(pair[0].trainIdx) == pair[0].queryIdx]
    result["ratio_matches"] = len(matches)
    if len(matches) < MIN_MATCHES:
        return None, result
    p0 = np.asarray([k0[m.queryIdx].pt for m in matches], dtype=np.float32)
    p1 = np.asarray([k1[m.trainIdx].pt for m in matches], dtype=np.float32)
    cv2.setRNGSeed(42)
    homography, inlier = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0,
                                         maxIters=5000, confidence=0.999)
    if homography is None or inlier is None or not np.isfinite(homography).all():
        result["status"] = "homography_fit_failed"
        return None, result
    inlier = inlier[:, 0].astype(bool)
    count = int(inlier.sum())
    result["inliers"] = count
    result["inlier_ratio"] = count / len(matches)
    if count < MIN_INLIERS:
        result["status"] = "insufficient_inliers"
        return None, result
    fitted = cv2.perspectiveTransform(p0[None], homography)[0]
    result["inlier_reprojection_px"] = stats(np.linalg.norm(fitted - p1, axis=1)[inlier])
    hull = cv2.convexHull(p0[inlier])
    coverage = float(cv2.contourArea(hull) / gray0.size)
    result["inlier_convex_hull_area_fraction"] = coverage
    height, width = gray0.shape
    inlier_xy = p0[inlier] / np.array([width, height])
    box_counts = [int(((inlier_xy[:, 0] >= x0) & (inlier_xy[:, 0] < x1)
                       & (inlier_xy[:, 1] >= y0) & (inlier_xy[:, 1] < y1)).sum())
                  for x0, y0, x1, y1 in BG_BOXES]
    result["inliers_per_background_box"] = box_counts
    if coverage < 0.05 or sum(count >= 4 for count in box_counts) < 2:
        result["status"] = "insufficient_spatial_support"
        return None, result
    result["status"] = "diagnostic_available"
    result["homography"] = homography.tolist()
    result["fit_is_not_held_out_validation"] = True
    return homography, result


def motion_pair(rgb0, rgb1, source0, source1):
    gray0 = cv2.cvtColor(rgb0, cv2.COLOR_RGB2GRAY)
    gray1 = cv2.cvtColor(rgb1, cv2.COLOR_RGB2GRAY)
    bg0, grass = masks(source0)
    bg1, _ = masks(source1)
    homography, fit = fit_background_homography(gray0, gray1, bg0, bg1)
    result = {"registration": fit, "source_mask_pixels": {"background": int(bg0.sum()),
                                                          "grass": int(grass.sum())}}
    if homography is None:
        result["residual_flow"] = None
        return result
    # Dense motion is measured on each variant; background H is estimated independently.
    h, w = gray0.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    grid = np.stack([xx, yy], axis=-1)
    projected = cv2.perspectiveTransform(grid.reshape(1, -1, 2), homography).reshape(h, w, 2)
    inverse_projected = cv2.perspectiveTransform(grid.reshape(1, -1, 2), np.linalg.inv(homography)).reshape(h, w, 2)
    # Initialize with the independently fitted camera-plane motion. Uninitialized
    # Farneback aliases repeated asphalt under the source's large adjacent shifts.
    kwargs = dict(pyr_scale=0.5, levels=5, winsize=25, iterations=5,
                  poly_n=7, poly_sigma=1.5, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    flow = cv2.calcOpticalFlowFarneback(gray0, gray1, projected - grid, **kwargs)
    reverse = cv2.calcOpticalFlowFarneback(gray1, gray0, inverse_projected - grid, **kwargs)
    target = grid + flow
    back = cv2.remap(reverse, target[..., 0], target[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    fb_error = np.linalg.norm(flow + back, axis=-1)
    valid = ((target[..., 0] >= 1) & (target[..., 0] < w - 2)
             & (target[..., 1] >= 1) & (target[..., 1] < h - 2)
             & (fb_error <= 2 + 0.05 * np.linalg.norm(flow, axis=-1)))
    residual = np.linalg.norm(target - projected, axis=-1)
    result["residual_flow"] = {}
    for name, mask in (("background", bg0), ("grass", grass)):
        result["residual_flow"][name] = {
            "pixels_after_consistency": int((mask & valid).sum()),
            "consistency_fraction": float((mask & valid).sum() / max(mask.sum(), 1)),
            "residual_px_per_frame_pair": stats(residual[mask & valid]),
            "raw_flow_px_per_frame_pair": stats(np.linalg.norm(flow, axis=-1)[mask & valid]),
            "forward_backward_error_px": stats(fb_error[mask & valid])}
    bg_median = result["residual_flow"]["background"]["residual_px_per_frame_pair"]["median"]
    result["motion_warning"] = ("background_residual_above_3px_do_not_interpret_as_grass_motion"
                                if bg_median is None or bg_median > 3 else None)
    return result


def font(size=16):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def fitted_tile(rgb, width, height):
    image = Image.fromarray(rgb)
    if image.size == (width, height):
        return image
    scale = min(width / image.width, height / image.height)
    size = (round(image.width * scale), round(image.height * scale))
    image = image.resize(size, Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (width, height), (24, 24, 24))
    tile.paste(image, ((width - size[0]) // 2, (height - size[1]) // 2))
    return tile


def board(sequences, titles, index, fps, h, w):
    header, footer = 60, 32
    canvas = Image.new("RGB", (w * len(sequences), h + header + footer), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for column, (name, sequence) in enumerate(sequences.items()):
        x = column * w
        draw.text((x + 8, 6), titles[name], fill="white", font=font(15))
        draw.text((x + 8, 30), f"frame {index:02d} | {index / fps:.1f} s", fill=(210, 210, 210), font=font(14))
        canvas.paste(fitted_tile(sequence[index], w, h), (x, header))
    draw.text((8, h + header + 7), "Generated RGB only | pixel/motion diagnostics do not establish 3D or visual acceptance",
              fill="white", font=font(12))
    return canvas


def video_encode(path, images, fps):
    first = np.asarray(images[0])
    h, w = first.shape[:2]
    command = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
               "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(path)]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        assert process.stdin is not None and process.stderr is not None
        for image in images:
            process.stdin.write(np.asarray(image).tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode()
        status = process.wait()
        if status:
            raise RuntimeError(f"ffmpeg exit {status}: {stderr}")
    probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate",
                            "-of", "json", str(path)], check=True, capture_output=True, text=True)
    stream = json.loads(probe.stdout)["streams"][0]
    if int(stream["nb_read_frames"]) != len(images):
        raise RuntimeError("Encoded comparison frame count does not match raw sequence")
    return stream


def run(args):
    cv2.setNumThreads(2)
    controls_path = args.controls.resolve()
    with np.load(controls_path, allow_pickle=False) as controls:
        source = controls["rgb"].copy()
        fps = float(controls["fps"])
        indices = controls["native_indices"].astype(int)
    if source.dtype != np.uint8 or source.ndim != 4 or source.shape[-1] != 3:
        raise ValueError("Controls RGB must be uint8 [T,H,W,3]")
    count, h, w, _ = source.shape
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid controls FPS")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be unoccupied: {output}")
    sequences = {"source": source}
    titles = {"source": "SOURCE RGB (exact rectified input)",
              "old_reference": "OLD reference render (native view)",
              "source_time": "MODEL / time advances", "freeze": "MODEL / time frozen"}
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "diagnostic_only_no_acceptance", "helper_sha256": digest(__file__),
                "controls": {"path": str(controls_path), "sha256": digest(controls_path),
                             "raw_rgb_sha256": array_digest(source), "shape": list(source.shape),
                             "fps": fps, "native_indices": indices.tolist()},
                "generated_frames": {}, "metrics": {}, "motion": {},
                "mask_policy": {"background_boxes_normalized_xyxy": BG_BOXES,
                                "grass_box_normalized_xyxy": GRASS_BOX,
                                "grass_hsv_cv2": {"hue": [24, 95], "saturation_min": 45, "value_min": 30},
                                "source_drives_all_variant_masks": True,
                                "rationale": "Inspected native/rectified source 0,8,16; upper/right and lower-left asphalt patches, with source green dilated 9x9 excluded. Homography requires >=4 inliers in >=2 separated patches and inlier hull >=5% of image to prevent extrapolation from a small textured patch."},
                "limitations": ["MAE/SSIM measure aligned RGB similarity, not static-world correctness.",
                                "Old render has native intrinsics; it is displayed with an aspect-preserving fit only and excluded from pixel metrics.",
                                "Homography is a planar asphalt approximation and cannot remove depth parallax on raised leaves.",
                                "Residual flow is 2D pixels per adjacent pair, not metric rigid/non-rigid scene flow.",
                                "Independent homography fitting can hide camera mismatch; inspect frame-by-frame source likeness too.",
                                "Source-derived grass masks can miss a hallucinated/displaced plant; inspect actual video.",
                                "Repeated texture, occlusion, low texture, and Farneback error affect motion statistics.",
                                "No metric is used as an acceptance threshold."]}
    if args.old_rgbd:
        old, old_receipts = [], []
        for index in indices:
            path = args.old_rgbd / f"{index:06d}.npz"
            with np.load(path, allow_pickle=False) as data:
                rgb = data["rgb"].copy()
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError(f"Invalid old RGB: {path}")
            old.append(rgb)
            old_receipts.append({"path": str(path.resolve()), "sha256": digest(path)})
        sequences["old_reference"] = np.stack(old)
        manifest["old_receipts"] = old_receipts
    if not args.source_only:
        if not args.source_time or not args.freeze:
            raise ValueError("Both --source-time and --freeze are required unless --source-only")
        for name in ("source_time", "freeze"):
            generated, receipts = load_frames(getattr(args, name), count)
            if generated.shape != source.shape:
                raise ValueError(f"{name} shape {generated.shape} != controls {source.shape}; no implicit resizing")
            sequences[name] = generated
            manifest["generated_frames"][name] = receipts
            per_frame = [{"frame": i, "mae_255": float(np.mean(np.abs(generated[i].astype(np.float32) - source[i]))),
                          "ssim_rgb": ssim_rgb(source[i], generated[i])} for i in range(count)]
            manifest["metrics"][name] = {"per_frame": per_frame,
                "mae_mean_255": float(np.mean([p["mae_255"] for p in per_frame])),
                "ssim_mean": float(np.mean([p["ssim_rgb"] for p in per_frame]))}
    output.mkdir(parents=True)
    for name, sequence in sequences.items():
        if name == "old_reference":
            continue
        pairs = []
        for index in range(count - 1):
            value = motion_pair(sequence[index], sequence[index + 1], source[index], source[index + 1])
            pairs.append({"frames": [index, index + 1], **value})
        manifest["motion"][name] = {"pairs": pairs,
            "available_pairs": sum(p["residual_flow"] is not None for p in pairs),
            "total_pairs": len(pairs)}
    images = [board(sequences, titles, i, fps, h, w) for i in range(count)]
    contact_indices = sorted(set(np.linspace(0, count - 1, min(5, count), dtype=int).tolist()))
    # Contact shares a single display resize for every method and row.
    contact_width = min(1472, images[0].width)
    contact_height = round(images[0].height * contact_width / images[0].width)
    contact = Image.new("RGB", (contact_width, contact_height * len(contact_indices)))
    for row, index in enumerate(contact_indices):
        contact.paste(images[index].resize((contact_width, contact_height), Image.Resampling.LANCZOS), (0, row * contact_height))
        images[index].save(output / f"comparison_{index:04d}.png")
    contact.save(output / "comparison_contact.png")
    manifest["video_probe"] = video_encode(output / "comparison.mp4", images, fps)
    mask_panels = []
    for index in contact_indices:
        bg, grass = masks(source[index])
        preview = source[index].copy()
        preview[bg] = (preview[bg].astype(np.float32) * .5 + np.array([0, 120, 255]) * .5).astype(np.uint8)
        preview[grass] = (preview[grass].astype(np.float32) * .5 + np.array([255, 0, 120]) * .5).astype(np.uint8)
        mask_panels.append(preview)
    Image.fromarray(np.concatenate(mask_panels, axis=1)).save(output / "mask_preview_only.png")
    manifest["display"] = {"contact_indices": contact_indices,
                           "native_generated_pixels_unmodified": True,
                           "video_compression": "H.264 CRF18; metric inputs are raw arrays/PNG",
                           "mask_preview_only": "Blue = asphalt; magenta = source grass; overlay is not used in comparisons or scores"}
    manifest["artifacts"] = {p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
                             for p in output.iterdir() if p.is_file()}
    (output / "diagnostics.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": manifest["status"], "output_dir": str(output),
                      "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "per_frame"} for k, v in manifest["metrics"].items()},
                      "motion_available_pairs": {k: v["available_pairs"] for k, v in manifest["motion"].items()}}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", required=True, type=Path)
    parser.add_argument("--source-time", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--old-rgbd", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
