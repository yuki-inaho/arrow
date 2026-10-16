#!/usr/bin/env python3
"""Prepare a short BulletTime pilot with constant intrinsics and explicit time units.

RGB resampling uses integer pixel centers and a NumPy bilinear sampler. Camera
poses remain the original provisional ARROW estimates; no pose fitting occurs.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw


MODEL_TIME_FPS = 30.0


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_intrinsics(K: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("Intrinsics must be a finite 3x3 matrix")
    if not np.allclose(K[2], [0, 0, 1], atol=1e-12, rtol=0):
        raise ValueError("Intrinsics must have homogeneous row [0, 0, 1]")
    if K[0, 0] <= 0 or K[1, 1] <= 0 or abs(K[1, 0]) > 1e-12:
        raise ValueError("Expected positive focal lengths and upper-triangular intrinsics")
    return K


def resize_affine(native_size: tuple[int, int], output_size: tuple[int, int]) -> np.ndarray:
    """Integer-center scaling: u_out = sx * (u_native + 0.5) - 0.5."""
    nh, nw = native_size
    oh, ow = output_size
    if min(nh, nw, oh, ow) <= 0:
        raise ValueError("Image dimensions must be positive")
    sx, sy = ow / nw, oh / nh
    return np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])


def source_map(K_source: np.ndarray, K_common: np.ndarray,
               output_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Map output pixel centers to source centers while preserving camera rays."""
    mapping = validate_intrinsics(K_source) @ np.linalg.inv(validate_intrinsics(K_common))
    yy, xx = np.indices(output_size, dtype=np.float64)
    x = mapping[0, 0] * xx + mapping[0, 1] * yy + mapping[0, 2]
    y = mapping[1, 0] * xx + mapping[1, 1] * yy + mapping[1, 2]
    return x, y


def choose_common_intrinsics(Ks: np.ndarray, native_size: tuple[int, int],
                             output_size: tuple[int, int],
                             max_zoom: float = 1.05) -> tuple[np.ndarray, float]:
    """Zoom scaled frame-0 K just enough to keep every mapped center in bounds."""
    Ks = np.asarray(Ks, dtype=np.float64)
    if Ks.ndim != 3 or Ks.shape[1:] != (3, 3) or len(Ks) == 0:
        raise ValueError("Expected one or more 3x3 intrinsics matrices")
    for K in Ks:
        validate_intrinsics(K)
    if not np.isfinite(max_zoom) or not 1 <= max_zoom <= 1.05:
        raise ValueError("max_zoom must be within [1, 1.05]")
    base = resize_affine(native_size, output_size) @ Ks[0]
    nh, nw = native_size
    oh, ow = output_size
    cx, cy = (ow - 1) / 2, (oh - 1) / 2
    corners = np.array([[0, 0, 1], [ow - 1, 0, 1],
                        [0, oh - 1, 1], [ow - 1, oh - 1, 1]], dtype=float)

    def candidate(zoom):
        Z = np.array([[zoom, 0, cx * (1 - zoom)],
                      [0, zoom, cy * (1 - zoom)], [0, 0, 1]])
        return Z @ base

    def inside(zoom):
        inverse = np.linalg.inv(candidate(zoom))
        for K in Ks:
            mapped = corners @ (K @ inverse).T
            if (mapped[:, 0].min() < 0 or mapped[:, 0].max() > nw - 1
                    or mapped[:, 1].min() < 0 or mapped[:, 1].max() > nh - 1):
                return False
        return True

    if inside(1.0):
        return candidate(1.0), 1.0
    if not inside(max_zoom):
        raise ValueError("Common intrinsics require zoom above the allowed 1.05 cap")
    low, high = 1.0, max_zoom
    for _ in range(50):
        mid = (low + high) / 2
        if inside(mid):
            high = mid
        else:
            low = mid
    return candidate(high), high


def bilinear_rgb(rgb: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample RGB at explicitly specified integer-center coordinates, without padding."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected uint8 HxWx3 RGB")
    h, w = rgb.shape[:2]
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid sampling maps")
    tolerance = 1e-9  # Matrix roundoff only; real out-of-bounds pixels fail.
    if (x.min() < -tolerance or x.max() > w - 1 + tolerance
            or y.min() < -tolerance or y.max() > h - 1 + tolerance):
        raise ValueError("Sampling map leaves source pixel-center bounds")
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    x0, y0 = np.floor(x).astype(np.intp), np.floor(y).astype(np.intp)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    wx, wy = (x - x0)[..., None], (y - y0)[..., None]
    result = ((1 - wy) * ((1 - wx) * rgb[y0, x0] + wx * rgb[y0, x1])
              + wy * ((1 - wx) * rgb[y1, x0] + wx * rgb[y1, x1]))
    return np.rint(result).clip(0, 255).astype(np.uint8)


def time_controls(count: int, fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if count < 5 or (count - 1) % 4 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Require T=4k+1 >=5 and positive playback fps")
    indices = np.arange(count, dtype=np.float64)
    actual = indices / fps
    # BulletTime's time embedding multiplies this value by hardcoded fps=30.
    # It must recover frame indices, independently of the source playback rate.
    source = (indices / MODEL_TIME_FPS).astype(np.float32)
    return source, np.zeros(count, dtype=np.float32), actual


def write_video(path: Path, frames: np.ndarray, fps: float) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=Fraction(str(fps)).limit_denominator())
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "16", "preset": "fast"}
        for rgb in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def write_native_contact(path: Path, frames: list[np.ndarray]) -> None:
    indices = np.linspace(0, len(frames) - 1, min(5, len(frames)), dtype=int)
    tile_w = 216
    tile_h = round(frames[0].shape[0] * tile_w / frames[0].shape[1])
    board = Image.new("RGB", (tile_w * len(indices), tile_h + 28), "#202020")
    draw = ImageDraw.Draw(board)
    for col, index in enumerate(indices):
        thumb = Image.fromarray(frames[index]).resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        board.paste(thumb, (tile_w * col, 28))
        draw.text((tile_w * col + 6, 7), f"Native frame {index}", fill="white")
    board.save(path)


def prepare_controls(input_dir: Path, output_dir: Path, *, count: int = 17,
                     height: int = 640, width: int = 368, fps: float = 5.0,
                     max_zoom: float = 1.05) -> dict:
    source_time, target_time, actual_time = time_controls(count, fps)
    if height <= 0 or width <= 0 or height % 16 or width % 16:
        raise ValueError("Output dimensions must be positive multiples of 16")
    input_dir, output_dir = input_dir.resolve(), output_dir.resolve()
    if input_dir == output_dir or input_dir in output_dir.parents:
        raise ValueError("Output must be separate from the original input directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory must be empty")
    frames_path = input_dir / "frames.json"
    frames_hash = digest(frames_path)
    metadata = json.loads(frames_path.read_text())
    records = metadata["frames"][:count]
    if len(records) != count or [r["frame_index"] for r in records] != list(range(count)):
        raise ValueError("Require contiguous source frames beginning at frame zero")
    native_size = tuple(metadata["original_size"])
    if len(native_size) != 2:
        raise ValueError("Expected original_size=[height,width]")
    Ks, Ms, rgbs, sources = [], [], [], []
    for index, record in enumerate(records):
        rgb_path = Path(record["rgb_path"])
        if not rgb_path.is_absolute():
            rgb_path = input_dir / rgb_path
        geo_path = input_dir / "stages" / "geometry" / f"{index:06d}.npz"
        rgb_hash, geo_hash = digest(rgb_path), digest(geo_path)
        if record.get("rgb_sha256") is not None and record["rgb_sha256"] != rgb_hash:
            raise ValueError(f"RGB source hash mismatch at frame {index}")
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        if rgb.shape[:2] != native_size:
            raise ValueError("RGB dimensions do not match frames.json")
        with np.load(geo_path, allow_pickle=False) as data:
            if not bool(data["arrow_K_valid"]) or not bool(data["arrow_pose_valid"]):
                raise ValueError(f"Invalid source calibration at frame {index}")
            K = validate_intrinsics(data["arrow_K"]).copy()
            M = np.asarray(data["arrow_camera_to_metric"], dtype=np.float64).copy()
        if (M.shape != (4, 4) or not np.isfinite(M).all()
                or not np.allclose(M[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
                or not np.allclose(M[:3, :3].T @ M[:3, :3], np.eye(3), atol=2e-4, rtol=0)
                or not np.isclose(np.linalg.det(M[:3, :3]), 1, atol=2e-4)):
            raise ValueError(f"Invalid camera-to-world rigid transform at frame {index}")
        Ks.append(K)
        Ms.append(M)
        rgbs.append(rgb)
        sources.append({"frame_index": index, "source_index": record["source_index"],
                        "original_timestamp": record.get("timestamp"),
                        "actual_timestamp_seconds": float(actual_time[index]),
                        "model_source_time": float(source_time[index]),
                        "model_target_time": 0.0, "rgb_path": str(rgb_path),
                        "rgb_sha256": rgb_hash, "geometry_path": str(geo_path),
                        "geometry_sha256": geo_hash, "original_K": K.tolist(),
                        "original_camera_to_world": M.tolist()})
    Ks, Ms = np.asarray(Ks), np.asarray(Ms)
    common, zoom = choose_common_intrinsics(Ks, native_size, (height, width), max_zoom)
    rectified, boundary = [], []
    for index, rgb in enumerate(rgbs):
        K = np.asarray(Ks[index], dtype=np.float64)
        x, y = source_map(K, common, (height, width))
        mapped_valid = ((x >= -1e-9) & (x <= native_size[1] - 1 + 1e-9)
                        & (y >= -1e-9) & (y <= native_size[0] - 1 + 1e-9))
        if not mapped_valid.all():
            raise ValueError(f"Rectification has invalid mapped pixels at frame {index}")
        rectified.append(bilinear_rgb(rgb, x, y))
        boundary.append({"frame_index": index, "valid_pixels": int(mapped_valid.sum()),
                         "total_pixels": height * width, "valid_fraction": 1.0,
                         "source_x_min": float(x.min()), "source_x_max": float(x.max()),
                         "source_y_min": float(y.min()), "source_y_max": float(y.max())})
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_array = np.stack(rectified)
    controls_path = output_dir / "controls.npz"
    np.savez_compressed(
        controls_path, rgb=rgb_array, camera_to_world=Ms, K=np.repeat(common[None], count, axis=0),
        source_time=source_time, target_time=target_time, fps=np.array(fps, dtype=np.float64),
        actual_timestamps=actual_time, model_time_fps=np.array(MODEL_TIME_FPS),
        original_K=Ks, original_camera_to_world=Ms.copy(),
        native_indices=np.arange(count, dtype=np.int64),
        source_indices=np.array([r["source_index"] for r in records], dtype=np.int64),
        original_rgb_sha256=np.array([r["rgb_sha256"] for r in sources]),
        original_geometry_sha256=np.array([r["geometry_sha256"] for r in sources]),
        native_size=np.array(native_size, dtype=np.int64), unit=np.array("arrow_provisional"),
    )
    video_path = output_dir / "source_rectified.mp4"
    contact_path = output_dir / "native_contact.png"
    write_video(video_path, rgb_array, fps)
    write_native_contact(contact_path, rgbs)
    if digest(frames_path) != frames_hash:
        raise RuntimeError("frames.json changed during preparation")
    for source in sources:
        if (digest(Path(source["rgb_path"])) != source["rgb_sha256"]
                or digest(Path(source["geometry_path"])) != source["geometry_sha256"]):
            raise RuntimeError("An original RGB or geometry source changed during preparation")
    manifest = {
        "status": "prepared_controls_only", "dataset": metadata.get("dataset"),
        "frame_count": count, "native_size_hw": list(native_size), "output_size_hw": [height, width],
        "original_fps": fps, "playback_fps": fps, "fps_provenance": "explicit CLI input",
        "model_time_fps": MODEL_TIME_FPS,
        "time_contract": "source_time=i/30; target_time=0; actual_timestamps=i/playback_fps; "
                         "model_time=actual_timestamp*playback_fps/30. Model time normalization "
                         "preserves frame-index embeddings; playback is not retimed.",
        "K_common": common.tolist(), "zoom": zoom, "max_zoom": max_zoom,
        "resize_affine": resize_affine(native_size, (height, width)).tolist(),
        "pixel_center_convention": "Pixel (row=v,col=u) center is (u,v), integer coordinates",
        "map_equations": "u_source_h=K_t @ inv(K_common) @ [u_out,v_out,1]; "
                         "K_common=Z_zoom @ A_halfpixel @ K_0; "
                         "A_halfpixel: u_out=sx*(u_native+0.5)-0.5, likewise y. "
                         "Z_zoom scales around output center ((W-1)/2,(H-1)/2).",
        "resampling": "NumPy bilinear at explicit source centers; round-to-nearest-even uint8; "
                      "no padding, invalid maps fail; original files untouched",
        "units": "arrow_provisional", "metric_calibrated": False,
        "pose_quality": "provisional; original ARROW camera_to_metric preserved exactly as float64; "
                        "no discontinuous PnP corrections used",
        "source_frames_json": str(frames_path), "source_frames_json_sha256": frames_hash,
        "sources": sources, "boundary_validity": boundary,
        "artifacts": {path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
                      for path in (controls_path, video_path, contact_path)},
        "script_sha256": digest(Path(__file__)),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--width", type=int, default=368)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--max-zoom", type=float, default=1.05)
    args = parser.parse_args()
    result = prepare_controls(args.input_dir, args.output_dir, count=args.frames,
                              height=args.height, width=args.width, fps=args.fps,
                              max_zoom=args.max_zoom)
    print(json.dumps({"frame_count": result["frame_count"], "zoom": result["zoom"],
                      "output_dir": str(args.output_dir.resolve()),
                      "boundary_valid_fraction": min(b["valid_fraction"]
                                                     for b in result["boundary_validity"])}))


if __name__ == "__main__":
    main()
