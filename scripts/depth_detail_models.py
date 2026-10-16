#!/usr/bin/env python3
"""深度詳細化パイプラインの段階実装（pipeline担当 scope）。

正本: temp/workdoc_Oct16-2026_depth_workflow.md（2026-10-16 最新ユーザー方針）。
主経路: ARROW clip (低解像度depth) -> clip Sim3 scale で尺度統一 -> 元RGB画素中心の
fullres ARROW anchor + pixel K -> LingBot-Depth (実モデル, 専用venv) -> ARROW anchor への
robust log ratio 再固定 -> gray edge + depth由来normal で NDR (backend=numpy 明示) ->
camera-compensated scene flow (rgbd_sceneflow 実API) -> edge/depth guided scene flow SR ->
edge/support/trust ゲート付き時間融合 -> 共通range の比較映像 + final 点群RRD。

単位は arrow_provisional（metric_calibrated=false）。全 depth / camera translation /
3D scene flow は後段で単一 scalar 乗算により一括校正できる構造を保つ。

このモジュールは run_depth_detail.py からも、各モデル専用 venv の worker subprocess からも
実行される。worker サブコマンドは外部 venv の python で起動されるため、
arrow パッケージに依存しない純 numpy ヘルパーだけを使う（core 契約を使う段階は
arrow venv 側の worker であり、その場合のみ arrow.depth_detail を import する）。

worker 一覧:
  ingest            入力列挙 + 元RGBをPNGへ一度だけ書き出し frames.json を保存
  anchor            ARROW clip NPZ -> fullres anchor depth (Sim3 scale適用) を生成
  geometry          ARROW pose/K -> reconstruct基準(0)へ変換 -> pixel K / metric poses
  lingbot           LingBot-Depth 実モデル推論 (LingBot venv)
  lingbot_fix       LingBot生出力を ARROW anchor へ robust log ratio 再固定 (arrow venv)
  prep              gray edge map + depth由来3D局所差分normalを生成 (arrow venv)
  ndr               normal_depth_refiner (backend=numpy 明示, NDR venv)
  sceneflow_prep    低解像度RGBDペアを作り z-buffer 再投影で camera 補償 (arrow venv)
  sceneflow_refine  観測残差flow + scene_flow_refinement.refine 実装 (sceneflow venv)
  sceneflow_post    低解像度結果を元解像度へ復元し、trust/edge/visibility gateを適用
  sr                scene flow super-resolution (core.superresolve_scene_flow)
  temporal          前後候補の補償済み時間融合 + 共通支持maskでの残差比較
  final             per-frame NPZ pack (frames/NNNNNN.npz) + final.npz + 共通depth range
  render            比較MP4/代表PNG/final点群RRD + ffprobe/RRD検証記録
  moge              MoGe v3 (OPTIONAL比較専用; 主経路では実行しない)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
UNITS = "arrow_provisional"
FORMAT_VERSION = 1
SCENEFLOW_PYTHONPATH = Path("/home/kasm-user/Desktop/rgbd_sceneflow_sandbox/scene_flow_primitives/src")

# --- CHUNK BOUNDARY: utilities continue ---


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n")
    os.replace(temporary, path)


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def natural_key(path: Path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def list_image_files(directory: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    return sorted(
        (p for p in Path(directory).iterdir() if p.is_file() and p.suffix.lower() in suffixes),
        key=natural_key,
    )


def read_rgb(path: Path) -> np.ndarray:
    """RGB uint8 [H, W, 3] を返す。PIL 優先、無ければ cv2（PNGなので両者で同一値）。"""
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except ImportError:
        import cv2  # type: ignore

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"failed to read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_png(rgb: np.ndarray, path: Path) -> None:
    rgb = np.asarray(rgb, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image  # type: ignore

        Image.fromarray(rgb).save(path, format="PNG")
        return
    except ImportError:
        import cv2  # type: ignore

        if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise ValueError(f"failed to write image: {path}")


def enumerate_source(source: Path, *, decode_frames: bool) -> list[dict]:
    """入力フレームを一意列挙する。フォルダは自然順、動画は decode 順。

    Returns: [{"source_index", "name", "original_size": (H, W), "timestamp"}]
    decode_frames=False の動画では総フレーム数を stream 情報から取る（dry-run 用）。
    """
    source = Path(source)
    if source.is_dir():
        frames = []
        for index, path in enumerate(list_image_files(source)):
            frames.append(
                {
                    "source_index": index,
                    "name": path.name,
                    "path": str(path),
                    "original_size": None,
                    "timestamp": None,
                }
            )
        return frames
    import av

    frames = []
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        count = int(stream.frames or 0)
        if not count and stream.duration and stream.time_base and stream.average_rate:
            count = round(float(stream.duration * stream.time_base) * float(stream.average_rate))
        if not count:
            raise SystemExit(f"動画のフレーム数を取得できません: {source}")
        for index, frame in enumerate(container.decode(stream)):
            timestamp = None
            if frame.pts is not None and frame.time_base is not None:
                timestamp = float(frame.pts * frame.time_base)
            frames.append(
                {
                    "source_index": index,
                    "name": f"{source.name}#{index:06d}",
                    "path": str(source),
                    "original_size": None,
                    "timestamp": timestamp,
                }
            )
            if not decode_frames and len(frames) >= count:
                break
    if not decode_frames:
        frames = [
            {"source_index": i, "name": f"{source.name}#{i:06d}", "path": str(source),
             "original_size": None, "timestamp": None}
            for i in range(count)
        ]
    return frames


# ---- 座標・K 規約ヘルパー -------------------------------------------------


def scale_k_affine(K: np.ndarray, scale_xy, offset_xy) -> np.ndarray:
    """old = scale * new + offset のとき、old画素Kをnew画素Kへ変換する。

    fx_new = fx_old / s、cx_new = (cx_old - t) / s（軸ごと）。
    """
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError("K must be 3x3")
    sx, sy = float(scale_xy[0]), float(scale_xy[1])
    tx, ty = float(offset_xy[0]), float(offset_xy[1])
    if sx <= 0 or sy <= 0:
        raise ValueError("scale must be positive")
    out = K.copy()
    out[0, 0] = K[0, 0] / sx
    out[1, 1] = K[1, 1] / sy
    out[0, 2] = (K[0, 2] - tx) / sx
    out[1, 2] = (K[1, 2] - ty) / sy
    return out


def moge_norm_to_pixel_k(K_norm: np.ndarray, width: int, height: int) -> np.ndarray:
    """MoGe規約 (uv [0,1], pixel=uv*size-0.5) の正規化K -> 整数画素中心K。"""
    K_norm = np.asarray(K_norm, dtype=np.float64)
    K = K_norm.copy()
    K[0, 0] = K_norm[0, 0] * width
    K[1, 1] = K_norm[1, 1] * height
    K[0, 2] = K_norm[0, 2] * width - 0.5
    K[1, 2] = K_norm[1, 2] * height - 0.5
    return K


def pixel_k_to_lingbot_norm(K: np.ndarray, width: int, height: int) -> np.ndarray:
    """LingBot規約 (u=arange, fx'=fx/W, cx'=cx/W) の正規化Kへ変換。"""
    K = np.asarray(K, dtype=np.float64)
    out = K.copy()
    out[0, 0] = K[0, 0] / width
    out[1, 1] = K[1, 1] / height
    out[0, 2] = K[0, 2] / width
    out[1, 2] = K[1, 2] / height
    return out


def downscale_geometry(height: int, width: int, long_side: int) -> dict:
    """長辺 long_side への縮小形状と画素中心スケール。"""
    scale = float(long_side) / float(max(height, width))
    low_h = max(2, int(round(height * scale)))
    low_w = max(2, int(round(width * scale)))
    sx = low_w / float(width)
    sy = low_h / float(height)
    # low = sx * high + (0.5*sx - 0.5)
    return {
        "low_size": [low_h, low_w],
        "sx": sx,
        "sy": sy,
        "offset": [0.5 * sx - 0.5, 0.5 * sy - 0.5],
    }


def resize_rgb_low(rgb: np.ndarray, low_size) -> np.ndarray:
    low_h, low_w = int(low_size[0]), int(low_size[1])
    try:
        from PIL import Image  # type: ignore

        image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
        if image.size == (low_w, low_h):
            return np.asarray(image, dtype=np.uint8)
        return np.asarray(image.resize((low_w, low_h), Image.Resampling.BILINEAR), dtype=np.uint8)
    except ImportError:
        import cv2  # type: ignore

        if rgb.shape[0] == low_h and rgb.shape[1] == low_w:
            return np.asarray(rgb, dtype=np.uint8)
        return cv2.resize(np.asarray(rgb, dtype=np.uint8), (low_w, low_h), interpolation=cv2.INTER_LINEAR)


def inverse_affine_2x3(affine) -> np.ndarray:
    """3x2 相当（[2,3] or [3,3]）の逆アフィンを [2,3] で返す。"""
    affine = np.asarray(affine, dtype=np.float64)
    if affine.shape == (3, 3):
        affine = affine[:2]
    if affine.shape != (2, 3):
        raise ValueError("affine must be [2,3] or [3,3]")
    linear = affine[:, :2]
    translation = affine[:, 2]
    inverse_linear = np.linalg.inv(linear)
    return np.concatenate([inverse_linear, (-inverse_linear @ translation)[:, None]], axis=1)


def map_points_affine(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    affine = np.asarray(affine, dtype=np.float64)
    return points @ affine[:, :2].T + affine[:, 2]


def upscale_geometry(high_height: int, high_width: int, low_size) -> dict:
    """低解像度 -> 高解像度の画素中心アフィン（low = sx*high + offset）。"""
    low_h, low_w = int(low_size[0]), int(low_size[1])
    sx = low_w / float(high_width)
    sy = low_h / float(high_height)
    return {"low_size": [low_h, low_w], "sx": sx, "sy": sy,
            "offset": [0.5 * sx - 0.5, 0.5 * sy - 0.5]}


def upsample_bilinear_masked(array: np.ndarray, out_shape, valid: np.ndarray | None = None,
                             support_threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """画素中心規約の bilinear 拡大（valid のみを正規化畳み込みで使う）。

    valid=None は全画素有効。戻り値 (upsampled, valid_up):
    valid_up は bilinear 支持の重み付き和が support_threshold 以上の画素のみ True。
    """
    array = np.asarray(array, dtype=np.float64)
    low_h, low_w = array.shape[:2]
    out_h, out_w = int(out_shape[0]), int(out_shape[1])
    if valid is None:
        valid = np.ones((low_h, low_w), dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if valid.shape != (low_h, low_w):
        raise ValueError("valid must match array shape")

    sx = low_w / float(out_w)
    sy = low_h / float(out_h)
    ys = (np.arange(out_h, dtype=np.float64) + 0.5) * sy - 0.5
    xs = (np.arange(out_w, dtype=np.float64) + 0.5) * sx - 0.5
    y0 = np.clip(np.floor(ys).astype(np.int64), 0, low_h - 1)
    x0 = np.clip(np.floor(xs).astype(np.int64), 0, low_w - 1)
    y1 = np.clip(y0 + 1, 0, low_h - 1)
    x1 = np.clip(x0 + 1, 0, low_w - 1)
    wy = np.clip(ys - y0, 0.0, 1.0)
    wx = np.clip(xs - x0, 0.0, 1.0)

    def gather(values):
        v00 = values[np.ix_(y0, x0)]
        v01 = values[np.ix_(y0, x1)]
        v10 = values[np.ix_(y1, x0)]
        v11 = values[np.ix_(y1, x1)]
        return ((1 - wy)[:, None] * ((1 - wx)[None, :] * v00 + wx[None, :] * v01)
                + wy[:, None] * ((1 - wx)[None, :] * v10 + wx[None, :] * v11))

    weight = gather(valid.astype(np.float64))
    if array.ndim == 2:
        numerator = gather(np.where(valid, array, 0.0))
    else:
        numerator = np.stack(
            [gather(np.where(valid, array[..., c], 0.0)) for c in range(array.shape[-1])], axis=-1
        )
    out_valid = weight >= float(support_threshold)
    with np.errstate(invalid="ignore", divide="ignore"):
        if array.ndim == 2:
            out = np.where(out_valid, numerator / np.maximum(weight, 1e-12), np.nan)
        else:
            out = np.where(out_valid[..., None], numerator / np.maximum(weight, 1e-12)[..., None], np.nan)
    return out, out_valid


def upsample_nearest_bool(mask: np.ndarray, out_shape) -> np.ndarray:
    """bool マスクの最近傍拡大（画素中心規約）。"""
    mask = np.asarray(mask, dtype=bool)
    high_h, high_w = int(out_shape[0]), int(out_shape[1])
    geometry = upscale_geometry(high_h, high_w, mask.shape[:2])
    sx, sy = geometry["sx"], geometry["sy"]
    ox, oy = geometry["offset"]
    ys = np.clip(np.rint(sy * np.arange(high_h, dtype=np.float64) + oy), 0, mask.shape[0] - 1).astype(np.int64)
    xs = np.clip(np.rint(sx * np.arange(high_w, dtype=np.float64) + ox), 0, mask.shape[1] - 1).astype(np.int64)
    return mask[np.ix_(ys, xs)]


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    if not np.any(finite):
        raise ValueError("weighted_median: no usable values")
    values, weights = values[finite], weights[finite]
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    if cumulative[-1] <= 0:
        raise ValueError("weighted_median: zero total weight")
    cutoff = 0.5 * cumulative[-1]
    return float(values[int(np.searchsorted(cumulative, cutoff, side="left"))])


def normals_from_depth(depth: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """depth と pixel K から 3D 局所差分で camera 座標 normal を作る。

    central difference（端は片側差分）、4近傍の有効画素のみ使用。
    camera 座標は x右 / y下 / z前。normal は z<=0（カメラ側）へそろえる。
    Returns (normals [H,W,3] float64, valid [H,W] bool)。
    """
    depth = np.asarray(depth, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    h, w = depth.shape
    valid = np.isfinite(depth) & (depth > 0)
    z = np.where(valid, depth, 0.0)
    u = np.arange(w, dtype=np.float64)[None, :].repeat(h, axis=0)
    v = np.arange(h, dtype=np.float64)[:, None].repeat(w, axis=1)
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    points = np.stack([x, y, z], axis=-1)

    def shifted(dx: int, dy: int) -> tuple[np.ndarray, np.ndarray]:
        out = np.zeros_like(points)
        ok = np.zeros((h, w), dtype=bool)
        sy = slice(max(0, dy), h + min(0, dy))
        ty = slice(max(0, -dy), h + min(0, -dy))
        sx = slice(max(0, dx), w + min(0, dx))
        tx = slice(max(0, -dx), w + min(0, -dx))
        out[ty, tx] = points[sy, sx]
        ok[ty, tx] = valid[sy, sx]
        return out, ok

    minus_y, ok_my = shifted(0, -1)
    plus_y, ok_py = shifted(0, +1)
    minus_x, ok_mx = shifted(-1, 0)
    plus_x, ok_px = shifted(+1, 0)

    dv = np.where(ok_py[..., None], plus_y - points, points - minus_y)
    dv_ok = ok_py | ok_my
    du = np.where(ok_px[..., None], plus_x - points, points - minus_x)
    du_ok = ok_px | ok_mx
    usable = valid & dv_ok & du_ok
    normals = np.cross(du, dv)
    length = np.linalg.norm(normals, axis=-1)
    usable &= np.isfinite(length) & (length > 1e-12)
    normals = normals / np.where(length[..., None] > 1e-12, length[..., None], 1.0)
    flip = normals[..., 2] > 0
    normals = np.where(flip[..., None], -normals, normals)
    normals[~usable] = np.nan
    return normals, usable


def sample_bilinear_masked_at(array: np.ndarray, valid: np.ndarray, ys: np.ndarray, xs: np.ndarray,
                              support_threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """連続座標 (ys, xs) で valid のみを使う bilinear サンプル（画素中心規約）。"""
    array = np.asarray(array, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    h, w = array.shape[:2]
    y0 = np.clip(np.floor(ys).astype(np.int64), 0, h - 1)
    x0 = np.clip(np.floor(xs).astype(np.int64), 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    wy = np.clip(ys - y0, 0.0, 1.0)
    wx = np.clip(xs - x0, 0.0, 1.0)

    def gather(values):
        v00 = values[y0, x0]
        v01 = values[y0, x1]
        v10 = values[y1, x0]
        v11 = values[y1, x1]
        return ((1 - wy) * ((1 - wx) * v00 + wx * v01)
                + wy * ((1 - wx) * v10 + wx * v11))

    weight = gather(valid.astype(np.float64))
    numerator = gather(np.where(valid, array, 0.0))
    support = weight >= float(support_threshold)
    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.where(support, numerator / np.maximum(weight, 1e-12), np.nan)
    return values, support


def motion_candidate_from_sr(depth0: np.ndarray, K0: np.ndarray, depth1: np.ndarray, K1: np.ndarray,
                             target_from_source: np.ndarray, sr_flow: np.ndarray, sr_valid: np.ndarray,
                             trust_abs: float, trust_rel: float) -> dict:
    """supervisor式: SR 3D変位から frame0 深度候補を correspondences 付きで作る。

    P1_0 = D0 * r0 + S (S=SR 3D変位, camera0 basis) -> P1_1 = T @ P1_0 -> (q1, zpred1) 投影
    -> D1obs = mask-aware bilinear -> P1obs_0 = inv(T) @ [D1obs * r1(q1)]
    -> candidate = (P1obs_0 - S).z。

    support = SR valid AND D0 valid AND in-bounds AND D1obs支持 AND occlusion/positive z
    (|D1obs - zpred1| <= max(trust_abs, trust_rel*|zpred1|))。同pixel比較はしない。
    """
    depth0 = np.asarray(depth0, dtype=np.float64)
    depth1 = np.asarray(depth1, dtype=np.float64)
    K0 = np.asarray(K0, dtype=np.float64)
    K1 = np.asarray(K1, dtype=np.float64)
    T = np.asarray(target_from_source, dtype=np.float64)
    flow = np.asarray(sr_flow, dtype=np.float64)
    sr_valid = np.asarray(sr_valid, dtype=bool)
    h, w = depth0.shape
    if flow.shape != (h, w, 3):
        raise ValueError("sr_flow must be [H, W, 3] with the same H/W as depth0")
    valid0 = np.isfinite(depth0) & (depth0 > 0) & sr_valid & np.isfinite(flow).all(axis=-1)
    valid1 = np.isfinite(depth1) & (depth1 > 0)

    u = np.arange(w, dtype=np.float64)[None, :].repeat(h, axis=0)
    v = np.arange(h, dtype=np.float64)[:, None].repeat(w, axis=1)
    ray0_x = (u - K0[0, 2]) / K0[0, 0]
    ray0_y = (v - K0[1, 2]) / K0[1, 1]
    point0 = np.stack([ray0_x * depth0, ray0_y * depth0, depth0], axis=-1)
    point1 = point0 + flow

    x = point1[..., 0]
    y = point1[..., 1]
    z = point1[..., 2]
    x1 = T[0, 0] * x + T[0, 1] * y + T[0, 2] * z + T[0, 3]
    y1 = T[1, 0] * x + T[1, 1] * y + T[1, 2] * z + T[1, 3]
    z1 = T[2, 0] * x + T[2, 1] * y + T[2, 2] * z + T[2, 3]
    front = valid0 & (z1 > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        qx = np.where(front, K1[0, 0] * x1 / z1 + K1[0, 2], np.nan)
        qy = np.where(front, K1[1, 1] * y1 / z1 + K1[1, 2], np.nan)
    inside = (
        front & np.isfinite(qx) & np.isfinite(qy)
        & (qx >= -0.5) & (qx <= w - 0.5) & (qy >= -0.5) & (qy <= h - 0.5)
    )
    qy_safe = np.where(inside, qy, 0.0)
    qx_safe = np.where(inside, qx, 0.0)
    d1_observed, d1_support = sample_bilinear_masked_at(depth1, valid1, qy_safe, qx_safe)
    tolerance = np.maximum(float(trust_abs), float(trust_rel) * np.abs(z1))
    consistent = np.isfinite(d1_observed) & (np.abs(d1_observed - z1) <= tolerance)
    support = inside & d1_support & consistent

    inverse = np.linalg.inv(T)
    ray1_x = (qx_safe - K1[0, 2]) / K1[0, 0]
    ray1_y = (qy_safe - K1[1, 2]) / K1[1, 1]
    observed1 = np.stack(
        [ray1_x * np.nan_to_num(d1_observed, nan=0.0),
         ray1_y * np.nan_to_num(d1_observed, nan=0.0),
         np.nan_to_num(d1_observed, nan=0.0),
         np.ones((h, w))],
        axis=-1,
    ).reshape(-1, 4)
    observed0 = (observed1 @ inverse.T).reshape(h, w, 4)
    candidate_z = observed0[..., 2] - flow[..., 2]
    candidate = np.where(support, candidate_z, np.nan)
    candidate_ok = support & np.isfinite(candidate) & (candidate > 0)
    candidate = np.where(candidate_ok, candidate, np.nan)
    return {
        "candidate": candidate.astype(np.float32),
        "support": candidate_ok,
        "zpred1": np.where(support, z1, np.nan).astype(np.float32),
        "d1_observed": np.where(support, d1_observed, np.nan).astype(np.float32),
        "q1": np.stack([np.where(support, qx, np.nan), np.where(support, qy, np.nan)], axis=-1),
    }


def reproject_zpred1(depth0: np.ndarray, K0: np.ndarray, target_from_source: np.ndarray,
                     sr_flow: np.ndarray) -> np.ndarray:
    """融合後深度から再投影 z (P1_1 の z) を再計算する（残差 after 用）。"""
    depth0 = np.asarray(depth0, dtype=np.float64)
    h, w = depth0.shape
    u = np.arange(w, dtype=np.float64)[None, :].repeat(h, axis=0)
    v = np.arange(h, dtype=np.float64)[:, None].repeat(w, axis=1)
    ray0_x = (u - K0[0, 2]) / K0[0, 0]
    ray0_y = (v - K0[1, 2]) / K0[1, 1]
    point0 = np.stack([ray0_x * depth0, ray0_y * depth0, depth0], axis=-1) + sr_flow
    x, y, z = point0[..., 0], point0[..., 1], point0[..., 2]
    T = np.asarray(target_from_source, dtype=np.float64)
    z1 = T[2, 0] * x + T[2, 1] * y + T[2, 2] * z + T[2, 3]
    return z1


def splat_scene_flow(depth: np.ndarray, K_source: np.ndarray, K_target: np.ndarray,
                     target_from_source: np.ndarray, flow: np.ndarray,
                     valid: np.ndarray | None = None) -> dict:
    """P1_0 = D0*r0 + S の全XYZを target カメラへ z-buffer scatter する。

    統括 REVIEW_BACKWARD_XY: depth のみの再投影では S_x/S_y が無視されるため、
    3D変位 S を含む点を変換し target 画素へ scatter する。
    戻り値 dict: depth（NaN invalid), valid, source_hit（z-buffer勝者の source index）。
    """
    depth = np.asarray(depth, dtype=np.float64)
    K_source = np.asarray(K_source, dtype=np.float64)
    K_target = np.asarray(K_target, dtype=np.float64)
    T = np.asarray(target_from_source, dtype=np.float64)
    flow = np.asarray(flow, dtype=np.float64)
    h, w = depth.shape
    if flow.shape != (h, w, 3):
        raise ValueError("flow must be [H, W, 3] matching depth")
    if valid is None:
        valid = np.ones((h, w), dtype=bool)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(depth) & (depth > 0) \
        & np.isfinite(flow).all(axis=-1)
    out_h, out_w = h, w
    rows, cols = np.nonzero(valid)
    z_src = depth[rows, cols]
    x = (cols - K_source[0, 2]) * z_src / K_source[0, 0]
    y = (rows - K_source[1, 2]) * z_src / K_source[1, 1]
    pts = np.stack([x, y, z_src], axis=-1) + flow[rows, cols]
    hom = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    moved = hom @ T.T
    z_t = moved[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        u = K_target[0, 0] * moved[:, 0] / z_t + K_target[0, 2]
        v = K_target[1, 1] * moved[:, 1] / z_t + K_target[1, 2]
    ok = (z_t > 0) & np.isfinite(u) & np.isfinite(v)
    flat_index = np.arange(rows.size)
    col_i = np.rint(u[ok]).astype(np.int64)
    row_i = np.rint(v[ok]).astype(np.int64)
    z_ok = z_t[ok]
    src_ok = flat_index[ok]
    inside = (col_i >= 0) & (col_i < out_w) & (row_i >= 0) & (row_i < out_h)
    col_i, row_i, z_ok, src_ok = col_i[inside], row_i[inside], z_ok[inside], src_ok[inside]
    depth_out = np.full((out_h, out_w), np.nan)
    valid_out = np.zeros((out_h, out_w), dtype=bool)
    source_hit = np.full((out_h, out_w), -1, dtype=np.int64)
    if z_ok.size:
        order = np.argsort(z_ok, kind="stable")
        flat = row_i[order] * out_w + col_i[order]
        unique_index = np.unique(flat, return_index=True)
        first = unique_index[1]
        win_r = row_i[order][first]
        win_c = col_i[order][first]
        depth_out[win_r, win_c] = z_ok[order][first]
        valid_out[win_r, win_c] = True
        source_hit[win_r, win_c] = src_ok[order][first]
    return {"depth": depth_out, "valid": valid_out, "source_hit": source_hit}


# --- 可視化・動画 ---------------------------------------------------------

_TURBO_ANCHORS = np.array(
    [
        [48, 18, 59], [64, 60, 128], [69, 101, 181], [65, 142, 218], [57, 174, 224],
        [47, 207, 205], [47, 231, 168], [66, 244, 120], [101, 246, 78], [142, 236, 53],
        [186, 219, 45], [221, 195, 44], [243, 167, 44], [253, 137, 39], [254, 108, 35],
        [243, 84, 29], [220, 64, 26], [182, 49, 22],
    ],
    dtype=np.float64,
)


def _turbo_lut(steps: int = 256) -> np.ndarray:
    anchors = _TURBO_ANCHORS
    positions = np.linspace(0.0, 1.0, len(anchors))
    xs = np.linspace(0.0, 1.0, steps)
    return np.stack([np.interp(xs, positions, anchors[:, c]) for c in range(3)], axis=-1)


_TURBO_LUT = _turbo_lut()


def colorize_scalar(array: np.ndarray, vmin: float, vmax: float, valid: np.ndarray | None = None,
                    invalid_color=(20, 20, 25)) -> np.ndarray:
    """スカラー[0]を共通 range で turbo 色 [H,W,3] uint8 にする。invalid は暗色。"""
    array = np.asarray(array, dtype=np.float64)
    if valid is None:
        valid = np.isfinite(array)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(array)
    span = float(vmax) - float(vmin)
    if not np.isfinite(span) or span <= 1e-12:
        span = 1.0
    with np.errstate(invalid="ignore"):
        normalized = np.clip((array - float(vmin)) / span, 0.0, 1.0)
    normalized = np.where(valid, normalized, 0.0)
    index = np.rint(normalized * (len(_TURBO_LUT) - 1)).astype(np.int64)
    colored = _TURBO_LUT[index].astype(np.uint8)
    colored[~valid] = np.asarray(invalid_color, dtype=np.uint8)
    return colored


def magnitude_xyz(flow_xyz: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(flow_xyz, dtype=np.float64), axis=-1)


class VideoWriter:
    """PyAV による H.264 MP4 ライター（yuv420p、偶数寸法へ内部調整）。"""

    def __init__(self, path: Path, width: int, height: int, fps: float, codec: str = "libx264"):
        from fractions import Fraction
        from typing import cast

        import av  # type: ignore[import-not-found]

        self._av = av
        self.width = int(width) - (int(width) % 2)
        self.height = int(height) - (int(height) % 2)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(path), mode="w")
        rate = Fraction(int(round(fps * 1000)), 1000)
        self.stream = cast("av.VideoStream", self.container.add_stream(codec, rate=rate))
        self.stream.width = self.width
        self.stream.height = self.height
        self.stream.pix_fmt = "yuv420p"
        self.codec = codec
        self.frames = 0

    def append(self, rgb: np.ndarray) -> None:
        rgb = np.asarray(rgb, dtype=np.uint8)[: self.height, : self.width]
        frame = self._av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.frames += 1

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


# --- NPZ 入出力 ------------------------------------------------------------


def load_npz_arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def save_npz_atomic(path: Path, arrays: dict, metadata: dict | None = None, compressed: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(arrays)
    if metadata is not None:
        payload["metadata"] = np.array(json.dumps(metadata, ensure_ascii=False))
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as stream:
        saver = np.savez_compressed if compressed else np.savez
        saver(stream, **payload)
    os.replace(temporary, path)


def stage_metadata(stage: str, dataset: str, model: str, params: dict, input_sha256,
                   argv: list, status: str = "complete", extras: dict | None = None) -> dict:
    metadata = {
        "format_version": FORMAT_VERSION,
        "stage": stage,
        "dataset": dataset,
        "model": model,
        "params": params,
        "input_sha256": input_sha256,
        "argv": list(argv),
        "status": status,
        "created_at": utc_now(),
        "units": UNITS,
        "metric_calibrated": False,
    }
    if extras:
        metadata.update(extras)
    return metadata


def read_frame_list(frames_json: Path) -> dict:
    payload = load_json(frames_json)
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"frames.json に frames がありません: {frames_json}")
    return payload


def load_frame_npz(dataset_dir: Path, stage: str, frame_index: int) -> dict | None:
    path = dataset_dir / "stages" / stage / f"{frame_index:06d}.npz"
    if not path.is_file():
        return None
    return load_npz_arrays(path)


def save_stage_frame(dataset_dir: Path, stage: str, frame_index: int, arrays: dict,
                     metadata: dict, compressed: bool = True) -> Path:
    path = dataset_dir / "stages" / stage / f"{frame_index:06d}.npz"
    save_npz_atomic(path, arrays, metadata, compressed=compressed)
    return path


# --- workers: ingest / anchor / geometry ---------------------------------


def load_clip_depth(clip_path: Path) -> dict:
    """ARROW clip NPZ から depth クエリ（reference=source の自カメラ光軸z）を読む。"""
    with np.load(clip_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        group = next(entry for entry in metadata["groups"] if entry["task"] == "depth")
        start, count = int(group["start"]), int(group["count"])
        if group.get("reference_frame") != "source":
            raise ValueError(f"depth queries の reference_frame が source ではありません: {clip_path}")
        xy = np.asarray(data["queries.xy"][0, start : start + count], dtype=np.float64)
        points = np.asarray(data["predictions.points"][0, start : start + count], dtype=np.float64)
    frames = len(metadata["frame_info"])
    per_frame = count // frames
    if per_frame * frames != count:
        raise ValueError(f"depth query 数がフレーム数で割り切れません: {clip_path}")
    return {
        "metadata": metadata,
        "xy": xy.reshape(frames, per_frame, 2),
        "depth": points[:, 2].reshape(frames, per_frame),
        "frames": frames,
        "per_frame": per_frame,
    }


def clip_frame_infos(clip_path: Path) -> list[dict]:
    with np.load(clip_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
    return metadata["frame_info"]


def worker_ingest(args: argparse.Namespace) -> int:
    """入力列挙 + 元RGBをPNGへ一度だけ書き出す（全venvで同一画素にするため）。"""
    dataset_dir = Path(args.dataset_dir)
    source = Path(args.source)
    frames_dir = dataset_dir / "frames_rgb"
    frames = []
    if source.is_dir():
        from PIL import Image, ImageOps  # type: ignore

        for path in list_image_files(source):
            with Image.open(path) as image:
                rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8)
            frames.append({"path": str(path), "name": path.name, "rgb": rgb, "timestamp": None})
    else:
        import av
        from PIL import Image  # type: ignore

        with av.open(str(source)) as container:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                timestamp = None
                if frame.pts is not None and frame.time_base is not None:
                    timestamp = float(frame.pts * frame.time_base)
                rotation = round(frame.rotation) % 360
                image = frame.to_image(width=frame.width, height=frame.height)
                if rotation == 90:
                    image = image.transpose(Image.Transpose.ROTATE_270)
                elif rotation == 180:
                    image = image.transpose(Image.Transpose.ROTATE_180)
                elif rotation == 270:
                    image = image.transpose(Image.Transpose.ROTATE_90)
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
                frames.append({"path": str(source), "name": f"{source.name}#{len(frames):06d}",
                               "rgb": rgb, "timestamp": timestamp})
    if not frames:
        raise SystemExit(f"入力フレームがありません: {source}")

    sizes = {tuple(rgb.shape[:2]) for rgb in (f["rgb"] for f in frames)}
    if len(sizes) != 1:
        raise SystemExit(f"元画像サイズが一意ではありません: {sorted(sizes)}")
    height, width = tuple(next(iter(sizes)))

    records = []
    for index, item in enumerate(frames):
        png_path = frames_dir / f"{index:06d}.png"
        if not png_path.is_file():
            write_png(item["rgb"], png_path)
        records.append(
            {
                "frame_index": index,
                "source_index": index,
                "name": item["name"],
                "relative_path": f"frames/{index:06d}.npz",
                "rgb_path": str(png_path),
                "rgb_sha256": sha256_file(png_path) if png_path.is_file() else sha256_bytes(item["rgb"].tobytes()),
                "rgb_sha256_source": "ingested_png_sha256",
                "height": int(height),
                "width": int(width),
                "timestamp": item["timestamp"],
                "source": str(item["path"]),
            }
        )
    payload = {
        "dataset": args.dataset,
        "source": str(source),
        "frame_count": len(records),
        "original_size": [int(height), int(width)],
        "frames": records,
        "created_at": utc_now(),
    }
    atomic_write_json(Path(args.frames_json), payload)
    atomic_write_json(
        dataset_dir / "configs" / "stage_configs" / "ingest.json",
        {
            "dataset": args.dataset,
            "source": str(source),
            "frame_count": len(records),
            "original_size": [int(height), int(width)],
            "argv": list(sys.argv),
            "created_at": utc_now(),
        },
    )
    print(f"ingest: {len(records)} フレーム ({width}x{height}) -> {frames_dir}")
    return 0


def worker_anchor(args: argparse.Namespace) -> int:
    """ARROW depth を元RGB画素中心へ復元し clip Sim3 scale で尺度統一する。"""
    dataset_dir = Path(args.dataset_dir)
    frames_payload = read_frame_list(Path(args.frames))
    frames = frames_payload["frames"]
    height, width = (int(v) for v in frames_payload["original_size"])
    alignments = load_json(Path(args.alignments))
    align_entries = {entry["name"]: entry for entry in alignments["clips"]}
    clips_manifest = load_json(Path(args.clips_dir) / "manifest.json")
    entries = sorted(
        (entry for entry in clips_manifest["clips"] if entry.get("status") == "done"),
        key=lambda entry: entry["start"],
    )
    if not entries:
        raise SystemExit("完了済みクリップがありません")

    scale_by_frame = np.full(len(frames), np.nan, dtype=np.float64)
    clip_by_frame = np.full(len(frames), -1, dtype=np.int64)
    collisions_total = 0
    per_clip = []
    for clip_entry in entries:
        clip_path = Path(args.clips_dir) / clip_entry["name"]
        loaded = load_clip_depth(clip_path)
        align = align_entries.get(clip_entry["name"])
        if align is None:
            raise SystemExit(f"alignments.json に {clip_entry['name']} がありません")
        clip_scale = float(align["scale"])
        if not np.isfinite(clip_scale) or clip_scale <= 0:
            raise SystemExit(f"clip scale が不正: {clip_entry['name']} -> {clip_scale}")
        frame_info = loaded["metadata"]["frame_info"]
        clip_start = int(clip_entry["start"])
        collision_count = 0
        for local in range(loaded["frames"]):
            info = frame_info[local]
            source_index = int(info.get("index", clip_start + local))
            if not 0 <= source_index < len(frames):
                raise SystemExit(f"source_index が範囲外: {clip_entry['name']} local={local}")
            record = frames[source_index]
            if [int(record["height"]), int(record["width"])] != [height, width]:
                raise SystemExit("frames.json と clip の original_size が不一致")
            affine = np.asarray(info["original_to_input"], dtype=np.float64)
            inverse = inverse_affine_2x3(affine)
            xy = loaded["xy"][local]
            depth = loaded["depth"][local]
            finite = np.isfinite(depth) & (depth > 0) & np.isfinite(xy).all(axis=1)
            original_xy = map_points_affine(xy[finite], inverse)
            depth = depth[finite]
            inside = (
                (original_xy[:, 0] >= -0.5) & (original_xy[:, 0] <= width - 0.5)
                & (original_xy[:, 1] >= -0.5) & (original_xy[:, 1] <= height - 0.5)
            )
            original_xy = original_xy[inside]
            depth = depth[inside]
            cols = np.clip(np.rint(original_xy[:, 0]).astype(np.int64), 0, width - 1)
            rows = np.clip(np.rint(original_xy[:, 1]).astype(np.int64), 0, height - 1)
            scaled = depth * clip_scale
            buffer = np.full((height, width), np.nan, dtype=np.float32)
            flat = rows * width + cols
            order = np.argsort(flat, kind="stable")
            flat_sorted = flat[order]
            unique_index = np.unique(flat_sorted, return_index=True)
            first = unique_index[1]
            winners = order[first]
            collision_count += int(len(flat) - len(winners))
            buffer[rows[winners], cols[winners]] = scaled[winners].astype(np.float32)
            valid = np.isfinite(buffer) & (buffer > 0)
            scale_by_frame[source_index] = clip_scale
            clip_by_frame[source_index] = entries.index(clip_entry)
            save_stage_frame(
                dataset_dir,
                "anchor",
                source_index,
                {
                    "arrow_depth": buffer.astype(np.float32),
                    "arrow_valid": valid,
                    "arrow_scale": np.float64(clip_scale),
                    "arrow_clip_index": np.int32(entries.index(clip_entry)),
                    "arrow_support_count": np.int64(int(valid.sum())),
                },
                stage_metadata(
                    "anchor",
                    args.dataset,
                    "ifradlin/arrow",
                    {
                        "clip": clip_entry["name"],
                        "clip_scale": clip_scale,
                        "clip_note": align.get("note", ""),
                        "query_stride": loaded["metadata"]["groups"][0].get("stride"),
                        "depth_reference": "source",
                    },
                    {"clips_manifest": sha256_file(Path(args.clips_dir) / "manifest.json"),
                     "alignments": sha256_file(Path(args.alignments))},
                    sys.argv,
                ),
            )
        collisions_total += collision_count
        per_clip.append(
            {
                "name": clip_entry["name"],
                "start": clip_entry["start"],
                "end": clip_entry["end"],
                "scale": clip_scale,
                "note": align.get("note", ""),
                "collisions": collision_count,
            }
        )
    missing = np.nonzero(~np.isfinite(scale_by_frame))[0]
    if missing.size:
        raise SystemExit(f"anchor 未生成フレーム: {missing.tolist()}")
    atomic_write_json(
        dataset_dir / "configs" / "stage_configs" / "anchor.json",
        {
            "dataset": args.dataset,
            "method": "depth_queries(source) -> pixel-centre 復元 -> clip Sim3 scale 統一",
            "clips": per_clip,
            "collisions_total": collisions_total,
            "units": UNITS,
            "metric_calibrated": False,
            "argv": list(sys.argv),
            "created_at": utc_now(),
        },
    )
    print(f"anchor: {len(frames)} フレーム, clips={len(entries)}, collisions={collisions_total}")
    return 0


def worker_geometry(args: argparse.Namespace) -> int:
    """ARROW pose(reference=middle) -> reconstruct基準(0)変換 + pixel K(元解像度)。"""
    dataset_dir = Path(args.dataset_dir)
    frames_payload = read_frame_list(Path(args.frames))
    frames = frames_payload["frames"]
    height, width = (int(v) for v in frames_payload["original_size"])
    alignments = load_json(Path(args.alignments))
    align_entries = {entry["name"]: entry for entry in alignments["clips"]}

    from dataclasses import fields

    from arrow.tasks.pose import estimate_poses
    from arrow.types import Predictions

    from numpy import array as np_array

    clips_manifest = load_json(Path(args.clips_dir) / "manifest.json")
    entries = sorted(
        (entry for entry in clips_manifest["clips"] if entry.get("status") == "done"),
        key=lambda entry: entry["start"],
    )
    summary = []
    for clip_entry in entries:
        clip_path = Path(args.clips_dir) / clip_entry["name"]
        with np.load(clip_path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            pose_entry = next(entry for entry in metadata["groups"] if entry["task"] == "pose")
            pose_start, pose_count = int(pose_entry["start"]), int(pose_entry["count"])
            predictions = {
                field.name: np_array(
                    data[f"predictions.{field.name}"][0:1, pose_start : pose_start + pose_count]
                )
                for field in fields(Predictions)
            }
            xy = np_array(data["queries.xy"][0:1, pose_start : pose_start + pose_count])
        import torch

        pose_predictions = Predictions(**{key: torch.from_numpy(value) for key, value in predictions.items()})
        frame_info = metadata["frame_info"]
        clip_frames = len(frame_info)
        poses = estimate_poses(
            pose_predictions,
            torch.from_numpy(xy),
            num_frames=clip_frames,
            reference_frame=pose_entry["reference_frame"],
            **pose_entry["pose"],
        )
        camera_to_reference = poses.camera_to_reference[0].numpy().astype(np.float64)
        valid = poses.valid[0].numpy().astype(bool)
        intrinsics = poses.intrinsics[0].numpy().astype(np.float64)
        reference_index = int(pose_entry["reference_frame"])
        reference_absolute = int(frame_info[reference_index].get("index", clip_entry["start"] + reference_index))

        align = align_entries.get(clip_entry["name"])
        if align is None:
            raise SystemExit(f"alignments.json に {clip_entry['name']} がありません")
        clip_scale = float(align["scale"])
        rotation = np.asarray(align["rotation"], dtype=np.float64)
        translation = np.asarray(align["translation"], dtype=np.float64)

        p0 = camera_to_reference[0]
        pose0_valid = bool(bool(valid[0]) and np.all(np.isfinite(p0)))
        p0_inv = np.linalg.inv(p0) if pose0_valid else None
        for local in range(clip_frames):
            source_index = int(frame_info[local].get("index", clip_entry["start"] + local))
            affine = np.asarray(frame_info[local]["original_to_input"], dtype=np.float64)
            k_input = intrinsics[local]
            k_valid = bool(np.all(np.isfinite(k_input))) and k_input[0, 0] > 0 and k_input[1, 1] > 0
            if k_valid:
                k_original = scale_k_affine(
                    k_input,
                    (affine[0, 0], affine[1, 1]),
                    (affine[0, 2], affine[1, 2]),
                )
                k_valid = bool(
                    np.all(np.isfinite(k_original)) and k_original[0, 0] > 1.0 and k_original[1, 1] > 1.0
                    and 0.0 <= k_original[0, 2] <= width - 1 and 0.0 <= k_original[1, 2] <= height - 1
                )
            else:
                k_original = np.full((3, 3), np.nan)
            camera_to_reference_f = camera_to_reference[local] if valid[local] else np.full((4, 4), np.nan)
            if pose0_valid and valid[local]:
                camera_to_reconstruct = p0_inv @ camera_to_reference_f
                # scaled depth + rigid c2w + scaled translation（SO3 det=1 を保つ）
                metric = np.eye(4)
                metric[:3, :3] = rotation @ camera_to_reconstruct[:3, :3]
                metric[:3, 3] = clip_scale * (rotation @ camera_to_reconstruct[:3, 3]) + translation
            else:
                camera_to_reconstruct = np.full((4, 4), np.nan)
                metric = np.full((4, 4), np.nan)
            save_stage_frame(
                dataset_dir,
                "geometry",
                source_index,
                {
                    "arrow_K": k_original.astype(np.float32),
                    "arrow_K_valid": np.bool_(k_valid),
                    "arrow_pose_valid": np.bool_(bool(valid[local])),
                    "arrow_pose_frame0_valid": np.bool_(pose0_valid),
                    "arrow_pose_reference_index": np.int64(reference_absolute),
                    "arrow_camera_to_reference": camera_to_reference_f.astype(np.float32),
                    "arrow_camera_to_reconstruct": camera_to_reconstruct.astype(np.float32),
                    "arrow_camera_to_metric": metric.astype(np.float32),
                },
                stage_metadata(
                    "geometry",
                    args.dataset,
                    "ifradlin/arrow",
                    {
                        "clip": clip_entry["name"],
                        "pose_reference_frame": reference_index,
                        "pose_reference_absolute_index": reference_absolute,
                        "pose_settings": pose_entry["pose"],
                        "reconstruct_reference": 0,
                        "pose_to_reconstruct": "P0^-1 @ camera_to_reference (P0=clip先頭フレーム)",
                        "clip_scale": clip_scale,
                    },
                    {"clips_manifest": sha256_file(Path(args.clips_dir) / "manifest.json"),
                     "alignments": sha256_file(Path(args.alignments))},
                    sys.argv,
                ),
            )
        summary.append(
            {
                "name": clip_entry["name"],
                "pose_valid": int(valid.sum()),
                "pose_frames": clip_frames,
                "pose_frame0_valid": pose0_valid,
                "reference_index": reference_index,
                "K_valid": int(np.isfinite(intrinsics[:, 0, 0]).sum()),
            }
        )
    atomic_write_json(
        dataset_dir / "configs" / "stage_configs" / "geometry.json",
        {
            "dataset": args.dataset,
            "clips": summary,
            "conventions": {
                "pose_reference": "middle (clip内 frames//2)",
                "reconstruct_reference": 0,
                "camera_to_metric": "rigid c2w (SO3) + clip_scale 乗算 translation; スケール済み深度と組み合わせる",
                "calibration": "後段で単一 scalar 乗算のみで depth/translation/3D flow を一括校正可能",
            },
            "units": UNITS,
            "metric_calibrated": False,
            "argv": list(sys.argv),
            "created_at": utc_now(),
        },
    )
    print(f"geometry: {len(entries)} clips, {len(frames)} frames")
    return 0


def worker_align(args: argparse.Namespace) -> int:
    """重複フレームの再構成点から clip 間 Sim(3) を推定する（ref0 基準 adapter）。

    既存 align_clip_scenes.py の estimate_chain / world_depth_range / world_extents を
    再利用する。既知のバイアス（camera_to_reference=middle を reconstruct=0 と混用）を
    避けるため、geometry stage が既に計算した inv(P0)@camera_to_reference (ref0 基準) を
    cameras として渡す。元スクリプトは変更しない。
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from align_clip_scenes import estimate_chain, world_depth_range, world_extents  # noqa: E402

    from dataclasses import fields

    import torch
    from arrow.tasks.pose import estimate_poses
    from arrow.types import Predictions

    dataset_dir = Path(args.dataset_dir)
    clips_manifest = load_json(Path(args.clips_dir) / "manifest.json")
    entries = sorted(
        (entry for entry in clips_manifest["clips"] if entry.get("status") == "done"),
        key=lambda entry: entry["start"],
    )

    clips = []
    for entry in entries:
        clip_path = Path(args.clips_dir) / entry["name"]
        with np.load(clip_path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            reconstruct_entry = next(
                item for item in metadata["groups"] if item["task"] == "reconstruct"
            )
            if reconstruct_entry.get("reference_frame") != 0:
                raise SystemExit(
                    f"reconstruct の reference_frame が 0 ではありません: {entry['name']}"
                )
            start, count = int(reconstruct_entry["start"]), int(reconstruct_entry["count"])
            points = np.asarray(data["predictions.points"][0, start : start + count], dtype=np.float64)
            depth_entry = next(item for item in metadata["groups"] if item["task"] == "depth")
            d_start, d_count = int(depth_entry["start"]), int(depth_entry["count"])
            depth_z = np.asarray(
                data["predictions.points"][0, d_start : d_start + d_count, 2], dtype=np.float64
            )
            pose_entry = next(item for item in metadata["groups"] if item["task"] == "pose")
            pose_start, pose_count = int(pose_entry["start"]), int(pose_entry["count"])
            pose_predictions = {
                field.name: data[f"predictions.{field.name}"][0:1, pose_start : pose_start + pose_count]
                for field in fields(Predictions)
            }
            pose_xy = np.asarray(data["queries.xy"][0:1, pose_start : pose_start + pose_count])
        frame_info = metadata["frame_info"]
        clip_frames = len(frame_info)
        if count % clip_frames or d_count % clip_frames:
            raise SystemExit(f"クエリ数がフレーム数で割り切れません: {entry['name']}")
        grid = points.reshape(clip_frames, count // clip_frames, 3)
        depth = depth_z.reshape(clip_frames, d_count // clip_frames)

        # align worker は geometry stage に依存しない（実行順: align -> geometry）。
        # ここで pose を推定し、inv(P0) で reconstruct(ref0) 基準へ変換してから Sim(3) を解く。
        poses = estimate_poses(
            Predictions(**{key: torch.from_numpy(value) for key, value in pose_predictions.items()}),
            torch.from_numpy(pose_xy),
            num_frames=clip_frames,
            reference_frame=pose_entry["reference_frame"],
            **pose_entry["pose"],
        )
        camera_to_reference = poses.camera_to_reference[0].numpy().astype(np.float64)
        pose_valid = poses.valid[0].numpy().astype(bool)
        p0 = camera_to_reference[0]
        pose0_valid = bool(bool(pose_valid[0]) and np.all(np.isfinite(p0)))
        if not pose0_valid:
            raise SystemExit(
                f"clip 先頭フレームの pose が無効で ref0 変換できません: {entry['name']}"
            )
        p0_inv = np.linalg.inv(p0)
        cameras = np.full((clip_frames, 4, 4), np.nan, dtype=np.float64)
        valid = np.zeros(clip_frames, dtype=bool)
        for local in range(clip_frames):
            if pose_valid[local] and np.all(np.isfinite(camera_to_reference[local])):
                cameras[local] = p0_inv @ camera_to_reference[local]
                valid[local] = True
        clips.append(
            {
                "name": entry["name"],
                "start": int(entry["start"]),
                "end": int(entry["end"]),
                "points": grid,
                "depth": depth,
                "cameras": {
                    "camera_to_reference": cameras,
                    "valid": valid,
                    "basis": "reconstruct ref0 = inv(P0) @ camera_to_reference(middle)",
                },
            }
        )
    transforms = estimate_chain(clips)
    depth_range = world_depth_range(clips, transforms)
    extents = world_extents(clips, transforms)
    output = {
        "created_at": utc_now(),
        "overlap_frames": int(clips_manifest.get("signature", {}).get("clip_overlap", 0)),
        "clips": [
            {
                "name": clip["name"],
                "start": clip["start"],
                "end": clip["end"],
                "scale": transform["scale"],
                "rotation": np.asarray(transform["rotation"]).round(8).tolist(),
                "translation": np.asarray(transform["translation"]).round(8).tolist(),
                "note": clip.get("note", ""),
            }
            for clip, transform in zip(clips, transforms)
        ],
        "depth_range_world": depth_range,
        "world_bounds": extents,
        "notes": (
            "重複フレームの再構成点群の密対応から Umeyama/RANSAC で推定。"
            "cameras は inv(P0)@camera_to_reference で reconstruct(ref0) 基準に統一済み "
            "(align_clip_scenes の関数を再使用、元スクリプトは無変更)。"
        ),
        "units": UNITS,
        "metric_calibrated": False,
        "argv": list(sys.argv),
    }
    alignments_path = dataset_dir / "stages" / "align" / "alignments.json"
    atomic_write_json(alignments_path, output)
    atomic_write_json(
        dataset_dir / "configs" / "stage_configs" / "align.json",
        {
            "dataset": args.dataset,
            "method": "align_clip_scenes.estimate_chain (ref0 camera adapter)",
            "clip_scales": [item["scale"] for item in output["clips"]],
            "clip_notes": {item["name"]: item["note"] for item in output["clips"]},
            "depth_range_world": depth_range,
            "world_bounds": extents,
            "alignments_path": str(alignments_path),
            "units": UNITS,
            "metric_calibrated": False,
            "argv": list(sys.argv),
            "created_at": utc_now(),
        },
    )
    for item in output["clips"]:
        print(f"align: {item['name']}: scale {item['scale']:.6f}")
    return 0


def downsample_bilinear_masked(array: np.ndarray, valid: np.ndarray | None, low_size,
                               support_threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """画素中心規約の bilinear 縮小（valid のみを正規化畳み込みで使う）。"""
    array = np.asarray(array, dtype=np.float64)
    high_h, high_w = array.shape[:2]
    low_h, low_w = int(low_size[0]), int(low_size[1])
    if valid is None:
        valid = np.ones((high_h, high_w), dtype=bool)
    valid = np.asarray(valid, dtype=bool)

    sx = low_w / float(high_w)
    sy = low_h / float(high_h)
    offset = (0.5 * sx - 0.5, 0.5 * sy - 0.5)
    ys = (np.arange(low_h, dtype=np.float64) - offset[1]) / sy
    xs = (np.arange(low_w, dtype=np.float64) - offset[0]) / sx

    y0 = np.clip(np.floor(ys).astype(np.int64), 0, high_h - 1)
    x0 = np.clip(np.floor(xs).astype(np.int64), 0, high_w - 1)
    y1 = np.clip(y0 + 1, 0, high_h - 1)
    x1 = np.clip(x0 + 1, 0, high_w - 1)
    wy = np.clip(ys - y0, 0.0, 1.0)
    wx = np.clip(xs - x0, 0.0, 1.0)

    def gather(values):
        v00 = values[np.ix_(y0, x0)]
        v01 = values[np.ix_(y0, x1)]
        v10 = values[np.ix_(y1, x0)]
        v11 = values[np.ix_(y1, x1)]
        return ((1 - wy)[:, None] * ((1 - wx)[None, :] * v00 + wx[None, :] * v01)
                + wy[:, None] * ((1 - wx)[None, :] * v10 + wx[None, :] * v11))

    weight = gather(valid.astype(np.float64))
    if array.ndim == 2:
        numerator = gather(np.where(valid, array, 0.0))
    else:
        numerator = np.stack(
            [gather(np.where(valid, array[..., c], 0.0)) for c in range(array.shape[-1])], axis=-1
        )
    out_valid = weight >= float(support_threshold)
    with np.errstate(invalid="ignore", divide="ignore"):
        if array.ndim == 2:
            out = np.where(out_valid, numerator / np.maximum(weight, 1e-12), np.nan)
        else:
            out = np.where(out_valid[..., None], numerator / np.maximum(weight, 1e-12)[..., None], np.nan)
    return out, out_valid


def camera_transform_between(pose_a: np.ndarray, pose_b: np.ndarray) -> np.ndarray:
    """camera_to_reference 姿勢2つから b カメラ←a カメラ変換を作る (inv(Pb) @ Pa)。"""
    return np.linalg.inv(np.asarray(pose_b, dtype=np.float64)) @ np.asarray(pose_a, dtype=np.float64)


# --- workers: lingbot / lingbot_fix / prep / ndr ---------------------------


def worker_lingbot(args: argparse.Namespace) -> int:
    """LingBot-Depth 実モデルを ARROW anchor depth に対して実行する。"""
    import torch

    # device 検査は model package の import より前に行う（欠依存やCUDA不可を明示diagnosticで fail）
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "lingbot: device=cuda を指定されたが CUDA が利用できません。"
            "指定を黙って CPU へ変更しない（明示エラー）。"
        )
    from mdm.model.v2 import MDMModel  # type: ignore[import-not-found]

    dataset_dir = Path(args.dataset_dir)
    frames_payload = read_frame_list(Path(args.frames))
    frames = frames_payload["frames"]
    model = MDMModel.from_pretrained(args.model).to(device)

    processed = 0
    skipped = []
    for record in frames:
        frame_index = int(record["frame_index"])
        anchor = load_frame_npz(dataset_dir, "anchor", frame_index)
        geometry = load_frame_npz(dataset_dir, "geometry", frame_index)
        if anchor is None or geometry is None:
            raise SystemExit(f"anchor/geometry が未生成: frame {frame_index}")
        rgb = read_rgb(Path(record["rgb_path"]))
        height, width = rgb.shape[:2]
        depth_anchor = np.asarray(anchor["arrow_depth"], dtype=np.float32)
        valid_anchor = np.asarray(anchor["arrow_valid"], dtype=bool)
        K = np.asarray(geometry["arrow_K"], dtype=np.float64)
        K_valid = bool(geometry["arrow_K_valid"]) and np.all(np.isfinite(K))

        if not K_valid or not np.any(valid_anchor):
            arrays = {
                "lingbot_depth_in": depth_anchor,
                "lingbot_depth_raw": np.full((height, width), np.nan, dtype=np.float32),
                "lingbot_valid": np.zeros((height, width), dtype=bool),
                "lingbot_mask": np.zeros((height, width), dtype=bool),
                "lingbot_K_norm": np.full((3, 3), np.nan, dtype=np.float32),
                "lingbot_K_pixel": K.astype(np.float32),
                "lingbot_support_count": np.int64(0),
            }
            metadata = stage_metadata(
                "lingbot", args.dataset, args.model,
                {"resolution_level": int(args.resolution_level), "use_fp16": True,
                 "skip_reason": "k_invalid_or_no_anchor"},
                {"anchor": frame_index}, sys.argv, status="skipped",
            )
            save_stage_frame(dataset_dir, "lingbot", frame_index, arrays, metadata)
            skipped.append({"frame_index": frame_index, "reason": "k_invalid_or_no_anchor"})
            continue

        K_norm = pixel_k_to_lingbot_norm(K, width, height)
        image_tensor = (
            torch.from_numpy(np.ascontiguousarray(rgb)).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        )
        depth_model_input = np.where(valid_anchor, depth_anchor, 0.0).astype(np.float32)
        depth_tensor = torch.from_numpy(depth_model_input).to(device).float()
        intrinsics_tensor = torch.from_numpy(K_norm.astype(np.float32)).to(device).unsqueeze(0)
        with torch.no_grad():
            output = model.infer(
                image_tensor,
                depth_in=depth_tensor,
                apply_mask=True,
                intrinsics=intrinsics_tensor,
                resolution_level=int(args.resolution_level),
                use_fp16=True,
            )
        depth_raw = output["depth"]
        if depth_raw.dim() == 3:
            depth_raw = depth_raw.squeeze(0)
        mask = output.get("mask")
        mask_np = (
            np.asarray(mask.squeeze(0).cpu().numpy(), dtype=bool)
            if mask is not None else np.ones((height, width), dtype=bool)
        )
        depth_np = np.asarray(depth_raw.cpu().numpy(), dtype=np.float32)
        valid_out = np.isfinite(depth_np) & (depth_np > 0)
        arrays = {
            "lingbot_depth_in": depth_anchor.astype(np.float32),
            "lingbot_depth_raw": depth_np.astype(np.float32),
            "lingbot_valid": valid_out,
            "lingbot_mask": mask_np,
            "lingbot_K_norm": K_norm.astype(np.float32),
            "lingbot_K_pixel": K.astype(np.float32),
            "lingbot_support_count": np.int64(int(valid_out.sum())),
        }
        metadata = stage_metadata(
            "lingbot", args.dataset, args.model,
            {
                "resolution_level": int(args.resolution_level),
                "use_fp16": True,
                "apply_mask": True,
                "intrinsics_convention": "fx/W, fy/H, cx/W, cy/H (u=arange pixel-centre)",
                "device": str(device),
                "torch": torch.__version__,
            },
            {"anchor_frame": frame_index}, sys.argv,
        )
        save_stage_frame(dataset_dir, "lingbot", frame_index, arrays, metadata)
        processed += 1
        if processed % 10 == 0:
            print(f"lingbot: {processed}/{len(frames)} frames", flush=True)
        del image_tensor, depth_tensor, output, depth_raw
    print(f"lingbot: 完了 {processed}, skipped {len(skipped)}")
    atomic_write_json(
        dataset_dir / "configs" / "stage_configs" / "lingbot.json",
        {
            "dataset": args.dataset,
            "model": args.model,
            "model_kind": "LingBot-Depth (MDMModel.infer, depth_in=ARROW anchor)",
            "resolution_level": int(args.resolution_level),
            "use_fp16": True,
            "device": str(device),
            "torch": torch.__version__,
            "skipped": skipped,
            "units": UNITS,
            "metric_calibrated": False,
            "argv": list(sys.argv),
            "created_at": utc_now(),
        },
    )
    return 0


def worker_lingbot_fix(args: argparse.Namespace) -> int:
    """LingBot生出力を ARROW anchor に robust log ratio で再固定する (per-frame係数)。"""
    from arrow.depth_detail import fit_scale, regularize_scales

    dataset_dir = Path(args.dataset_dir)
    frames_payload = read_frame_list(Path(args.frames))
    frames = frames_payload["frames"]
    count = len(frames)
    ratios = np.ones(count, dtype=np.float64)
    weights = np.zeros(count, dtype=np.float64)
    residuals = np.full(count, np.nan)
    inliers = np.zeros(count, dtype=np.int64)
    reasons = []
    for record in frames:
        frame_index = int(record["frame_index"])
        anchor = load_frame_npz(dataset_dir, "anchor", frame_index)
        lingbot = load_frame_npz(dataset_dir, "lingbot", frame_index)
        if anchor is None or lingbot is None:
            ratios[frame_index] = 1.0
            weights[frame_index] = 0.0
            reasons.append({"frame_index": frame_index, "reason": "missing_stage"})
            continue
        depth_anchor = np.asarray(anchor["arrow_depth"], dtype=np.float64)
        valid_anchor = np.asarray(anchor["arrow_valid"], dtype=bool)
        raw = np.asarray(lingbot["lingbot_depth_raw"], dtype=np.float64)
        valid_raw = np.asarray(lingbot["lingbot_valid"], dtype=bool)
        mask = valid_anchor & valid_raw
        try:
            fit = fit_scale(raw, depth_anchor, mask=mask, min_points=int(args.min_points))
        except ValueError as error:
            ratios[frame_index] = 1.0
            weights[frame_index] = 0.0
            reasons.append({"frame_index": frame_index, "reason": f"fit_scale: {error}"})
            continue
        ratios[frame_index] = float(fit["scale"])
        weights[frame_index] = float(fit["inliers"])
        residuals[frame_index] = float(fit["median_abs_log_residual"])
        inliers[frame_index] = int(fit["inliers"])
    if float(weights.sum()) <= 0.0:
        raise SystemExit("lingbot_fix: 全フレームで対応不足 (対応不足はエラー契約)")
    fixed_scales = regularize_scales(ratios, weights, strength=float(args.strength))
    per_frame = []
    for record in frames:
        frame_index = int(record["frame_index"])
        lingbot = load_frame_npz(dataset_dir, "lingbot", frame_index)
        if lingbot is None:
            continue
        raw = np.asarray(lingbot["lingbot_depth_raw"], dtype=np.float64)
        valid_raw = np.asarray(lingbot["lingbot_valid"], dtype=bool)
        fix = float(fixed_scales[frame_index])
        depth_fixed = np.where(valid_raw, raw * fix, np.nan).astype(np.float32)
        arrays = {
            "lingbot_depth": depth_fixed,
            "lingbot_scale_fix": np.float64(fix),
            "lingbot_scale_raw": np.float64(ratios[frame_index]),
            "lingbot_fix_weight": np.float64(weights[frame_index]),
            "lingbot_fix_inliers": np.int64(inliers[frame_index]),
            "lingbot_fix_residual": np.float64(residuals[frame_index]),
        }
        metadata = stage_metadata(
            "lingbot_fix", args.dataset,
            "arrow.depth_detail.fit_scale+regularize_scales",
            {"strength": float(args.strength), "min_points": int(args.min_points),
             "target": "arrow anchor (arrow_provisional)"},
            {"source": "lingbot_depth_raw"}, sys.argv,
        )
        save_stage_frame(dataset_dir, "lingbot_fix", frame_index, arrays, metadata)
        per_frame.append(
            {
                "frame_index": frame_index,
                "raw_ratio": float(ratios[frame_index]),
                "fixed": fix,
                "weight": float(weights[frame_index]),
                "inliers": int(inliers[frame_index]),
                "residual": float(residuals[frame_index]) if np.isfinite(residuals[frame_index]) else None,
            }
        )
    valid_scales = np.asarray([r["fixed"] for r in per_frame], dtype=np.float64)
    atomic_write_json(
        dataset_dir / "configs" / "stage_configs" / "lingbot_fix.json",
        {
            "dataset": args.dataset,
            "method": "robust median log ratio (fit_scale) + log regularization (strength)",
            "strength": float(args.strength),
            "min_points": int(args.min_points),
            "fix_stats": {
                "median": float(np.median(valid_scales)),
                "min": float(valid_scales.min()),
                "max": float(valid_scales.max()),
            },
            "per_frame": per_frame,
            "insufficient_frames": reasons,
            "units": UNITS,
            "metric_calibrated": False,
            "argv": list(sys.argv),
            "created_at": utc_now(),
        },
    )
    print(f"lingbot_fix: {len(per_frame)} frames, median fix {np.median(valid_scales):.6f}")
    return 0


def worker_prep(args: argparse.Namespace) -> int:
    """gray edge map と depth由来 normal（3D局所差分 + pixel K）を作る。"""
    from arrow.depth_detail import gray_edges

    dataset_dir = Path(args.dataset_dir)
    frames_payload = read_frame_list(Path(args.frames))
    frames = frames_payload["frames"]
    normal_frames = 0
    for record in frames:
        frame_index = int(record["frame_index"])
        rgb = read_rgb(Path(record["rgb_path"]))
        height, width = rgb.shape[:2]
        edges = np.asarray(gray_edges(rgb), dtype=np.float32)
        lingbot = load_frame_npz(dataset_dir, "lingbot_fix", frame_index)
        geometry = load_frame_npz(dataset_dir, "geometry", frame_index)
        if lingbot is None or geometry is None:
            raise SystemExit(f"lingbot_fix/geometry が未生成: frame {frame_index}")
        depth = np.asarray(lingbot["lingbot_depth"], dtype=np.float64)
        K = np.asarray(geometry["arrow_K"], dtype=np.float64)
        K_valid = bool(bool(geometry["arrow_K_valid"]) and np.all(np.isfinite(K)))
        if K_valid:
            normals, normal_valid = normals_from_depth(depth, K)
        else:
            normals = np.full((height, width, 3), np.nan)
            normal_valid = np.zeros((height, width), dtype=bool)
        normal_frames += int(normal_valid.any())
        arrays = {
            "prep_edges": edges,
            "prep_normal": normals.astype(np.float32),
            "prep_normal_valid": normal_valid,
            "prep_normal_count": np.int64(int(normal_valid.sum())),
        }
        metadata = stage_metadata(
            "prep", args.dataset, "arrow.depth_detail.gray_edges + depth-derived normals",
            {
                "edge_map": "[0,1] gray gradient (core.gray_edges); 高勾配を深度境界と同一視しない",
                "normal_method": "unproject(depth,K) の 4近傍3D局所差分 cross(du,dv), z<=0 へそろえ, 正規化",
                "k_valid": K_valid,
            },
            {"lingbot_fix": frame_index, "geometry": frame_index}, sys.argv,
        )
        save_stage_frame(dataset_dir, "prep", frame_index, arrays, metadata)
    atomic_write_json(
        dataset_dir / "configs" / "stage_configs" / "prep.json",
        {
            "dataset": args.dataset,
            "edge_source": "ingested RGB -> core.gray_edges (max正規化 [0,1])",
            "normal_source": "lingbot_fix depth + arrow_K (pixel centre) 3D局所差分",
            "frames_with_normals": normal_frames,
            "units": UNITS,
            "metric_calibrated": False,
            "argv": list(sys.argv),
            "created_at": utc_now(),
        },
    )
    print(f"prep: {len(frames)} frames, normals in {normal_frames}")
    return 0


def worker_ndr(args: argparse.Namespace) -> int:
    """normal_depth_refiner（backend=numpy 明示）で LingBot 深度を法線誘導精緻化する。"""
    from dataclasses import asdict

    from normal_depth_refiner.api import refine_depth  # type: ignore[import-not-found]
    from normal_depth_refiner.config import CameraIntrinsics, RefinementConfig  # type: ignore[import-not-found]

    dataset_dir = Path(args.dataset_dir)
    frames_payload = read_frame_list(Path(args.frames))
    frames = frames_payload["frames"]
    config = RefinementConfig().with_updates(
        iterations=int(args.iterations),
        trust_abs=float(args.trust_abs),
        trust_rel=float(args.trust_rel),
        min_depth=float(args.min_depth),
        max_depth=float(args.max_depth),
    )
    backend = str(args.backend)
    if backend == "numpy" and not args.allow_numpy:
        raise SystemExit("ndr: backend=numpy は --allow-numpy を明示しない限り使用しません")
    skipped = []
    processed = 0
    for record in frames:
        frame_index = int(record["frame_index"])
        lingbot = load_frame_npz(dataset_dir, "lingbot_fix", frame_index)
        prep = load_frame_npz(dataset_dir, "prep", frame_index)
        geometry = load_frame_npz(dataset_dir, "geometry", frame_index)
        if lingbot is None or prep is None or geometry is None:
            raise SystemExit(f"stage 未生成: frame {frame_index}")
        depth = np.asarray(lingbot["lingbot_depth"], dtype=np.float64)
        depth_valid = np.isfinite(depth) & (depth > 0)
        normals = np.asarray(prep["prep_normal"], dtype=np.float64)
        normal_valid = np.asarray(prep["prep_normal_valid"], dtype=bool)
        edges = np.asarray(prep["prep_edges"], dtype=np.float64)
        K = np.asarray(geometry["arrow_K"], dtype=np.float64)
        K_valid = bool(geometry["arrow_K_valid"]) and np.all(np.isfinite(K))
        height, width = depth.shape
        if not K_valid or not np.any(depth_valid & normal_valid):
            arrays = {
                "ndr_depth_in": depth.astype(np.float32),
                "ndr_depth": np.full((height, width), np.nan, dtype=np.float32),
                "ndr_valid": np.zeros((height, width), dtype=bool),
                "ndr_support_count": np.int64(0),
            }
            metadata = stage_metadata(
                "ndr", args.dataset, "normal_depth_refiner",
                {"skip_reason": "k_invalid_or_no_support", "backend": backend,
                 "iterations": int(args.iterations),
                 "trust_abs": float(args.trust_abs), "trust_rel": float(args.trust_rel)},
                {"lingbot_fix": frame_index}, sys.argv, status="skipped",
            )
            save_stage_frame(dataset_dir, "ndr", frame_index, arrays, metadata)
            skipped.append({"frame_index": frame_index, "reason": "k_invalid_or_no_support"})
            continue
        mask = depth_valid & normal_valid
        refined = refine_depth(
            depth=depth,
            normals=np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0),
            intrinsics=CameraIntrinsics(fx=float(K[0, 0]), fy=float(K[1, 1]),
                                        cx=float(K[0, 2]), cy=float(K[1, 2])),
            mask=mask,
            edge_map=edges,
            config=config,
            backend=backend,
        )
        refined = np.asarray(refined, dtype=np.float64)
        valid_out = mask & np.isfinite(refined) & (refined > 0)
        out = np.where(valid_out, refined, np.nan).astype(np.float32)
        arrays = {
            "ndr_depth_in": depth.astype(np.float32),
            "ndr_depth": out.astype(np.float32),
            "ndr_valid": valid_out,
            "ndr_support_count": np.int64(int(valid_out.sum())),
        }
        metadata = stage_metadata(
            "ndr", args.dataset, "normal_depth_refiner",
            {"backend": backend, "iterations": int(args.iterations), "config": asdict(config),
             "mask": "depth_valid & normal_valid (未支持はvalid=false)",
             "units": UNITS},
            {"lingbot_fix": frame_index}, sys.argv,
        )
        save_stage_frame(dataset_dir, "ndr", frame_index, arrays, metadata)
        processed += 1
        if processed % 10 == 0:
            print(f"ndr: {processed}/{len(frames)} frames", flush=True)
    atomic_write_json(
        dataset_dir / "configs" / "stage_configs" / "ndr.json",
        {
            "dataset": args.dataset,
            "backend": backend,
            "backend_policy": "rust 明示 (統括 benchmark: 約6.1倍, max_abs_diff 7.36e-13)。失敗時 fallback 禁止",
            "iterations": int(args.iterations),
            "config": asdict(config),
            "trust_policy": {
                "trust_abs": float(args.trust_abs),
                "trust_rel": float(args.trust_rel),
                "units": UNITS,
                "note": "workdoc 開始値 trust_abs=0.05, trust_rel=0.05 (暫定ARROW単位)",
            },
            "depth_bounds_policy": {
                "min_depth": float(args.min_depth), "max_depth": float(args.max_depth),
                "max_depth_source": args.max_depth_source,
            },
            "skipped": skipped,
            "processed": processed,
            "units": UNITS,
            "metric_calibrated": False,
            "argv": list(sys.argv),
            "created_at": utc_now(),
        },
    )
    print(f"ndr: 完了 {processed}, skipped {len(skipped)}")
    return 0


# --- workers: sceneflow prep / refine / post -------------------------------


def build_frame_clip_map(clips_manifest: dict) -> list[list[int]]:
    """frame -> そのframeを含む clip 番号（start順）の一覧。"""
    entries = sorted(
        (entry for entry in clips_manifest["clips"] if entry.get("status") == "done"),
        key=lambda entry: entry["start"],
    )
    frame_count = max(int(entry["end"]) for entry in entries)
    mapping: list[list[int]] = [[] for _ in range(frame_count)]
    for index, entry in enumerate(entries):
        for frame in range(int(entry["start"]), int(entry["end"])):
            mapping[frame].append(index)
    return mapping


def assign_pairs(clips_manifest: dict) -> tuple[list[dict], dict]:
    """連続フレームペアを一意な clip へ割り当てる（重複は低index優先、跨clipは除外計上）。"""
    mapping = build_frame_clip_map(clips_manifest)
    frame_count = len(mapping)
    pairs = []
    cross_clip_skipped = 0
    for frame in range(frame_count - 1):
        common = [index for index in mapping[frame] if index in mapping[frame + 1]]
        if not common:
            cross_clip_skipped += 1
            continue
        pairs.append({"pair_index": [frame, frame + 1], "clip_index": int(min(common))})
    return pairs, {"pairs": len(pairs), "cross_clip_skipped": cross_clip_skipped,
                   "frame_count": frame_count}


def worker_sceneflow_prep(args: argparse.Namespace) -> int:
    """低解像度RGBDペアを作り、z-buffer 再投影で camera を補償する (core.reproject_depth)。"""
    from arrow.depth_detail import reproject_depth

    dataset_dir = Path(args.dataset_dir)
    frames_payload = read_frame_list(Path(args.frames))
    frames = frames_payload["frames"]
    height, width = (int(v) for v in frames_payload["original_size"])
    clips_manifest = load_json(Path(args.clips_dir) / "manifest.json")
    pairs, pair_stats = assign_pairs(clips_manifest)
    geometry_low = downscale_geometry(height, width, int(args.long_side))
    low_size = geometry_low["low_size"]
    sx, sy = geometry_low["sx"], geometry_low["sy"]
    offset_x, offset_y = geometry_low["offset"]

    inputs_dir = dataset_dir / "stages" / "sceneflow" / "pairs" / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    skipped = []
    written = []
    for pair in pairs:
        frame_a, frame_b = pair["pair_index"]
        geometry_a = load_frame_npz(dataset_dir, "geometry", frame_a)
        geometry_b = load_frame_npz(dataset_dir, "geometry", frame_b)
        ndr_a = load_frame_npz(dataset_dir, "ndr", frame_a)
        ndr_b = load_frame_npz(dataset_dir, "ndr", frame_b)
        if geometry_a is None or geometry_b is None or ndr_a is None or ndr_b is None:
            raise SystemExit(f"sceneflow_prep: stage 未生成 pair {frame_a}-{frame_b}")
        K_a = np.asarray(geometry_a["arrow_K"], dtype=np.float64)
        K_b = np.asarray(geometry_b["arrow_K"], dtype=np.float64)
        metric_a = np.asarray(geometry_a["arrow_camera_to_metric"], dtype=np.float64)
        metric_b = np.asarray(geometry_b["arrow_camera_to_metric"], dtype=np.float64)
        pose_valid = (
            bool(geometry_a["arrow_pose_valid"]) and bool(geometry_b["arrow_pose_valid"])
            and bool(geometry_a["arrow_pose_frame0_valid"]) and bool(geometry_b["arrow_pose_frame0_valid"])
            and np.all(np.isfinite(metric_a)) and np.all(np.isfinite(metric_b))
        )
        K_valid = bool(geometry_a["arrow_K_valid"]) and bool(geometry_b["arrow_K_valid"])
        if not pose_valid:
            skipped.append({"pair_index": pair["pair_index"], "reason": "pose_invalid"})
            continue
        if not K_valid or not np.all(np.isfinite(K_a)) or not np.all(np.isfinite(K_b)):
            skipped.append({"pair_index": pair["pair_index"], "reason": "k_invalid"})
            continue
        depth_a = np.asarray(ndr_a["ndr_depth"], dtype=np.float64)
        valid_a = np.asarray(ndr_a["ndr_valid"], dtype=bool)
        depth_b = np.asarray(ndr_b["ndr_depth"], dtype=np.float64)
        valid_b = np.asarray(ndr_b["ndr_valid"], dtype=bool)
        if not np.any(valid_a) or not np.any(valid_b):
            skipped.append({"pair_index": pair["pair_index"], "reason": "empty_depth"})
            continue

        rgb_a = read_rgb(Path(frames[frame_a]["rgb_path"]))
        rgb_b = read_rgb(Path(frames[frame_b]["rgb_path"]))
        rgb_a_low = resize_rgb_low(rgb_a, low_size)
        rgb_b_low = resize_rgb_low(rgb_b, low_size)
        depth_a_low, valid_a_low = downsample_bilinear_masked(depth_a, valid_a, low_size)
        depth_b_low, valid_b_low = downsample_bilinear_masked(depth_b, valid_b, low_size)
        # high = (1/sx) * low - offset/sx の画素中心規約
        K_low_a = scale_k_affine(K_a, (1.0 / sx, 1.0 / sy), (-offset_x / sx, -offset_y / sy))
        K_low_b = scale_k_affine(K_b, (1.0 / sx, 1.0 / sy), (-offset_x / sx, -offset_y / sy))
        # per-frame fit 差はあり得るため、明らかな破綻のみ除外し drift は記録する
        drift = float(np.max(np.abs((K_low_b - K_low_a) / np.maximum(np.abs(K_low_a), 1e-9))))
        if not np.isfinite(drift) or drift > 0.25:
            skipped.append({"pair_index": pair["pair_index"], "reason": "intrinsics_drift",
                            "drift": drift})
            continue
        # source=b (frame b), target=a (frame a): 補償は camera b -> camera a
        target_from_source = camera_transform_between(metric_b, metric_a)
        compensated = reproject_depth(
            depth=depth_b_low,
            rgb=rgb_b_low,
            K_source=K_low_b,
            K_target=K_low_a,
            target_from_source=target_from_source,
            output_shape=low_size,
        )
        depth_b_in_a = np.asarray(compensated["depth"], dtype=np.float32)
        rgb_b_in_a = np.asarray(compensated["rgb"], dtype=np.uint8)
        valid_b_in_a = np.asarray(compensated["valid"], dtype=bool)

        coverage = float(valid_b_in_a.mean())
        arrays = {
            "rgb0_low": rgb_a_low,
            "rgb1_low": rgb_b_low,
            "rgb1_in_0": rgb_b_in_a,
            "depth0_low": depth_a_low.astype(np.float32),
            "depth1_low": depth_b_low.astype(np.float32),
            "depth1_in_0": depth_b_in_a.astype(np.float32),
            "valid0_low": valid_a_low,
            "valid1_low": valid_b_low,
            "valid1_in_0": valid_b_in_a,
            "K_low": K_low_a,
            "K_low_source": K_low_b,
            "pair_index": np.asarray(pair["pair_index"], dtype=np.int64),
            "clip_index": np.int32(pair["clip_index"]),
            "low_size": np.asarray(low_size, dtype=np.int32),
        }
        name = f"pair_{frame_a:06d}_{frame_b:06d}.npz"
        save_npz_atomic(
            inputs_dir / name,
            arrays,
            {
                "pair_index": pair["pair_index"],
                "clip_index": pair["clip_index"],
                "target_from_source": target_from_source.tolist(),
                "basis": "arrow_camera_to_metric (world metric, rigid rotation + scaled translation) の inv(M_a)@M_b",
                "coverage": coverage,
                "K_drift": drift,
                "units": UNITS,
            },
        )
        written.append(
            {"pair_index": pair["pair_index"], "clip_index": pair["clip_index"],
             "file": str(inputs_dir / name), "coverage": coverage, "K_drift": drift,
             "valid0_frac": float(valid_a_low.mean()), "valid1_frac": float(valid_b_low.mean())}
        )
    coverage_values = [item["coverage"] for item in written]
    pair_list = {
        "dataset": args.dataset,
        "long_side": int(args.long_side),
        "low_size": low_size,
        "pairs": written,
        "skipped": skipped,
        "counts": {
            **pair_stats,
            "written": len(written),
            "skipped_pose": sum(1 for item in skipped if item["reason"] == "pose_invalid"),
            "skipped_k": sum(1 for item in skipped if item["reason"] == "k_invalid"),
            "skipped_empty": sum(1 for item in skipped if item["reason"] == "empty_depth"),
        },
        "coverage_stats": {
            "mean": float(np.mean(coverage_values)) if coverage_values else None,
            "median": float(np.median(coverage_values)) if coverage_values else None,
            "min": float(np.min(coverage_values)) if coverage_values else None,
        },
        "units": UNITS,
        "argv": list(sys.argv),
        "created_at": utc_now(),
    }
    atomic_write_json(dataset_dir / "stages" / "sceneflow" / "pairs.json", pair_list)
    atomic_write_json(
        dataset_dir / "configs" / "stage_configs" / "sceneflow_prep.json",
        {
            "dataset": args.dataset,
            "compensation": "core.reproject_depth (z-buffer, occlusion, out-of-bounds) 低解像度",
            "long_side": int(args.long_side),
            "low_size": low_size,
            "pixel_centre_K": "high = (1/sx)*low - offset/sx の画素中心規約で変換",
            "pose_note": "同一clip内の camera_to_reference から inv(P_b)@P_a。推定pose不良は除外計上",
            **pair_list["counts"],
            "coverage_stats": pair_list["coverage_stats"],
            "units": UNITS,
            "metric_calibrated": False,
            "argv": list(sys.argv),
            "created_at": utc_now(),
        },
    )
    print(f"sceneflow_prep: pairs={len(written)} skipped={len(skipped)} "
          f"cross_clip={pair_stats['cross_clip_skipped']}")
    return 0


def worker_sceneflow_refine(args: argparse.Namespace) -> int:
    """観測残差flow + scene_flow_refinement.refine（明示Config）を低解像度で実行する。"""
    import cv2  # type: ignore[import-not-found]
    from dataclasses import asdict

    from scene_flow_refinement.api import Config, refine  # type: ignore[import-not-found]

    if not (0 < float(args.min_depth) < float(args.max_depth)):
        raise SystemExit(
            f"sceneflow config の min/max が不正: {args.min_depth} {args.max_depth}"
        )
    if not (0 < float(args.max_motion) < float(args.min_depth)):
        raise SystemExit(
            "sceneflow config: max_motion < min_depth が必要（C++ solver制約: "
            f"max_motion={args.max_motion} min_depth={args.min_depth}）。"
            "runner は max_motion = min(frac*median, 0.5*min_depth) で導出する。"
        )

    dataset_dir = Path(args.dataset_dir)
    pairs_path = dataset_dir / "stages" / "sceneflow" / "pairs.json"
    if not pairs_path.is_file():
        raise SystemExit(
            f"sceneflow_refine: prepared pairs.json がありません: {pairs_path}"
            "（先に sceneflow_prep を実行）"
        )
    pair_list = load_json(pairs_path)
    outputs_dir = dataset_dir / "stages" / "sceneflow" / "pairs" / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import replace

    config = replace(Config(), min_depth=float(args.min_depth), max_depth=float(args.max_depth),
                     max_motion=float(args.max_motion), trust_sigma=float(args.trust_sigma))
    if not (0 < config.min_depth < config.max_depth):
        raise SystemExit(f"sceneflow config の min/max が不正: {config.min_depth} {config.max_depth}")

    # 1pair目開始前の全入力 preflight（範囲決定の正本=実入力。clamp/除外で値を変えない）
    scan_values = []
    for item in pair_list["pairs"]:
        pair = item["pair_index"]
        try:
            with np.load(item["file"], allow_pickle=False) as data:
                for key in ("depth0_low", "depth1_in_0"):
                    arr = np.asarray(data[key], dtype=np.float64)
                    positive = arr[np.isfinite(arr) & (arr > 0)]
                    if positive.size == 0:
                        continue
                    bad = positive[(positive < config.min_depth) | (positive > config.max_depth)]
                    if bad.size:
                        raise SystemExit(
                            f"sceneflow_refine preflight: pair {pair} の {key} に範囲外深度 "
                            f"{float(bad.min()):.6g}..{float(bad.max()):.6g} "
                            f"(config [{config.min_depth:.6g}, {config.max_depth:.6g}])。"
                            "値のclamp/除外はしないため、bounds導出を確認してください"
                        )
                    scan_values.append(positive)
        except (OSError, ValueError, KeyError) as error:
            raise SystemExit(
                f"sceneflow_refine preflight: pair {pair} の入力を読めません: "
                f"{type(error).__name__}: {error}"
            ) from error
    if not scan_values:
        raise SystemExit(
            f"sceneflow_refine preflight: finite positive な solver 入力がありません: {pairs_path}"
        )
    scan_merged = np.concatenate(scan_values)
    scan_stats = {
        "pair_count": len(pair_list["pairs"]),
        "value_count": int(scan_merged.size),
        "global_min": float(scan_merged.min()),
        "global_max": float(scan_merged.max()),
        "median": float(np.median(scan_merged)),
    }

    def process_pair(entry: dict) -> dict:
        """1 pair の処理（各pair固有NPZのみ書き、共有状態は持たない）。"""
        pair = entry["pair_index"]
        with np.load(entry["file"], allow_pickle=False) as data:
            rgb0_low = data["rgb0_low"]
            rgb1_in_0 = data["rgb1_in_0"]
            depth0_low = np.asarray(data["depth0_low"], dtype=np.float64)
            depth1_in_0 = np.asarray(data["depth1_in_0"], dtype=np.float64)
            valid0 = np.asarray(data["valid0_low"], dtype=bool)
            valid1 = np.asarray(data["valid1_in_0"], dtype=bool)
            K_low = np.asarray(data["K_low"], dtype=np.float64)
            low_size = [int(v) for v in data["low_size"]]
        gray0 = (0.299 * rgb0_low[..., 0] + 0.587 * rgb0_low[..., 1] + 0.114 * rgb0_low[..., 2]).astype(np.uint8)
        gray1 = (0.299 * rgb1_in_0[..., 0] + 0.587 * rgb1_in_0[..., 1] + 0.114 * rgb1_in_0[..., 2]).astype(np.uint8)
        flow = cv2.calcOpticalFlowFarneback(gray0, gray1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow = np.asarray(flow, dtype=np.float64)

        clipped = int(
            np.count_nonzero(valid0 & ((depth0_low <= config.min_depth) | (depth0_low >= config.max_depth)))
            + np.count_nonzero(valid1 & ((depth1_in_0 <= config.min_depth) | (depth1_in_0 >= config.max_depth)))
        )

        start = time.perf_counter()
        result = refine(
            rgb0=rgb0_low.astype(np.float64) / 255.0,
            rgb1=rgb1_in_0.astype(np.float64) / 255.0,
            depth0=depth0_low,
            depth1=depth1_in_0,
            flow=flow,
            intrinsics=np.asarray([K_low[0, 0], K_low[1, 1], K_low[0, 2], K_low[1, 2]], dtype=np.float64),
            config=config,
        )
        elapsed = time.perf_counter() - start
        stats = result.get("stats", {})
        arrays = {
            "depth0_refined": np.asarray(result["depth0"], dtype=np.float32),
            "depth1_refined": np.asarray(result["depth1"], dtype=np.float32),
            "support": np.asarray(result["support"], dtype=bool),
            "scene_flow": np.asarray(result["scene_flow"], dtype=np.float32),
            "flow_observed": flow.astype(np.float32),
            "match_flow": np.asarray(result["match_flow"], dtype=np.float32),
            "initial_scene_flow": np.asarray(result["initial_scene_flow"], dtype=np.float32),
            "geometry_residual": np.asarray(result["geometry_residual"], dtype=np.float32),
            "rigid_weight": np.asarray(result["rigid_weight"], dtype=np.float32),
            "stats_values": np.asarray(list(stats.values()), dtype=np.float64),
            "pair_index": np.asarray(pair, dtype=np.int64),
            "low_size": np.asarray(low_size, dtype=np.int32),
            "clipped_points": np.int64(clipped),
        }
        name = f"pair_{pair[0]:06d}_{pair[1]:06d}.npz"
        save_npz_atomic(
            outputs_dir / name,
            arrays,
            {
                "pair_index": pair,
                "stats_keys": list(stats.keys()),
                "config": asdict(config),
                "units": UNITS,
                "elapsed_seconds": elapsed,
                "clipped_points": clipped,
            },
        )
        return {
                "pair_index": pair,
                "file": str(outputs_dir / name),
                "support_frac": float(np.asarray(result["support"], dtype=bool).mean()),
                "scene_flow_norm_median": float(np.median(np.linalg.norm(result["scene_flow"], axis=-1))),
                "clipped_points": clipped,
                "elapsed_seconds": elapsed,
                "stats": stats,
            }

    workers = max(1, int(args.workers))
    entries: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [(pool.submit(process_pair, item), item["pair_index"]) for item in pair_list["pairs"]]
        for index, (future, pair) in enumerate(pending, 1):
            try:
                entries.append(future.result())
            except Exception as error:
                raise SystemExit(
                    f"sceneflow_refine: pair {pair} が失敗 (workers={workers}): "
                    f"{type(error).__name__}: {error}"
                ) from error
            if index % 10 == 0 or index == len(pending):
                print(f"sceneflow_refine: {index}/{len(pending)} pairs (workers={workers})", flush=True)
    clipped_total = sum(int(entry["clipped_points"]) for entry in entries)
    clipped_frames = sum(1 for entry in entries if int(entry["clipped_points"]) > 0)
    atomic_write_json(
        dataset_dir / "configs" / "stage_configs" / "sceneflow_refine.json",
        {
            "dataset": args.dataset,
            "api": "scene_flow_refinement.refine (rgbd_sceneflow 実API, 固定カメラ前提を補償で満たす)",
            "config": asdict(config),
            "config_policy": {
                "min_depth": float(args.min_depth),
                "max_depth": float(args.max_depth),
                "max_motion": float(args.max_motion),
                "trust_sigma": float(args.trust_sigma),
                "units": UNITS,
                "note": "default max_depth=5m の切捨ては黙認せず、実データ分位で明示設定",
            },
            "clipped_points": clipped_total,
            "clipped_frames": clipped_frames,
            "workers": workers,
            "input_scan": scan_stats,
            "preflight": "全pairを1pair目開始前に範囲検査（clamp/除外なし; 範囲外は即fail）",
            "bounds_source": "runner --min-depth/--max-depth/--max-motion（全prepared入力scan導出値）",
            "pairs": entries,
            "environment": {
                "python": sys.version.split()[0],
                "cv2": cv2.__version__,
                "pythonpath": os.environ.get("PYTHONPATH", ""),
            },
            "units": UNITS,
            "metric_calibrated": False,
            "argv": list(sys.argv),
            "created_at": utc_now(),
        },
    )
    print(f"sceneflow_refine: {len(entries)} pairs, clipped_points={clipped_total}")
    return 0


def worker_sceneflow_post(args: argparse.Namespace) -> int:
    """低解像度refine結果を元解像度へ戻し、trust/edge/visibility gateで支持を決める。"""
    dataset_dir = Path(args.dataset_dir)
    frames_payload = read_frame_list(Path(args.frames))
    frames = frames_payload["frames"]
    height, width = (int(v) for v in frames_payload["original_size"])
    pair_list = load_json(dataset_dir / "stages" / "sceneflow" / "pairs.json")
    trust_abs = float(args.trust_abs)
    trust_rel = float(args.trust_rel)

    written = []
    rejected_pixels = 0
    support_pixels = 0
    outputs_dir = dataset_dir / "stages" / "sceneflow" / "pairs" / "outputs"
    for entry in pair_list["pairs"]:
        pair = entry["pair_index"]
        output_path = outputs_dir / f"pair_{pair[0]:06d}_{pair[1]:06d}.npz"
        if not output_path.is_file():
            raise SystemExit(f"sceneflow_refine 出力がありません: {output_path}")
        with np.load(output_path, allow_pickle=False) as data:
            depth0_refined = np.asarray(data["depth0_refined"], dtype=np.float64)
            depth1_refined = np.asarray(data["depth1_refined"], dtype=np.float64)
            support = np.asarray(data["support"], dtype=bool)
        ndr_a = load_frame_npz(dataset_dir, "ndr", pair[0])
        if ndr_a is None:
            raise SystemExit(f"sceneflow_post: ndr stage 未生成 frame {pair[0]}")
        current = np.asarray(ndr_a["ndr_depth"], dtype=np.float64)
        current_valid = np.asarray(ndr_a["ndr_valid"], dtype=bool) & np.isfinite(current) & (current > 0)

        with np.load(entry["file"], allow_pickle=False) as data:
            depth0_low = np.asarray(data["depth0_low"], dtype=np.float64)
            valid0_low = np.asarray(data["valid0_low"], dtype=bool)
            valid1_in_0 = np.asarray(data["valid1_in_0"], dtype=bool)
        prep = load_frame_npz(dataset_dir, "prep", pair[0])
        if prep is None:
            raise SystemExit(f"sceneflow_post: prep 未生成 frame {pair[0]}")
        edges = np.asarray(prep["prep_edges"], dtype=np.float64)

        # 細部保持: delta0 = depth0_refined - depth0_low(NDR入力) を solver support 付きで拡大し、
        # 高解像度 NDR に加算する（低解像度 refined で NDR を置換しない）
        delta_low = np.where(
            support & valid0_low & np.isfinite(depth0_refined) & np.isfinite(depth0_low),
            depth0_refined - depth0_low, np.nan,
        )
        delta_up, solver_support_up = upsample_bilinear_masked(
            delta_low, (height, width), valid=support & valid0_low
        )
        compensation_valid = upsample_nearest_bool(valid1_in_0, (height, width))

        # next 候補（compensated grid の情報; temporal 候補には使わない）
        depth1_up, valid1_up = upsample_bilinear_masked(depth1_refined, (height, width))
        support_up = upsample_nearest_bool(support, (height, width))
        trust = np.maximum(trust_abs, trust_rel * np.abs(np.where(current_valid, current, np.nan)))
        edge_weight = 1.0 - np.clip(edges, 0.0, 1.0)
        trust_ok = np.isfinite(delta_up) & (np.abs(delta_up) <= trust)
        correction_support = solver_support_up & trust_ok & current_valid
        delta_eff = np.where(correction_support, delta_up * edge_weight, 0.0)
        own_depth = np.where(current_valid, current + delta_eff, np.nan).astype(np.float32)
        own_valid = current_valid
        fallback_mask = current_valid & ~correction_support
        fallback_count = int(fallback_mask.sum())

        next_gate = valid1_up & support_up & compensation_valid
        within_next = np.isfinite(depth1_up) & (np.abs(depth1_up - current) <= trust)
        next_valid = next_gate & within_next & (depth1_up > 0)
        rejected_pixels += int(np.count_nonzero(solver_support_up & ~correction_support))
        support_pixels += int(support_up.sum())

        next_depth = np.where(next_valid, depth1_up, np.nan).astype(np.float32)
        arrays = {
            "sceneflow_depth": own_depth,
            "sceneflow_valid": own_valid,
            "sceneflow_correction_support": correction_support,
            "sceneflow_correction_count": np.int64(int(correction_support.sum())),
            "sceneflow_solver_support": solver_support_up,
            "sceneflow_solver_support_count": np.int64(int(solver_support_up.sum())),
            "sceneflow_fallback_mask": fallback_mask,
            "sceneflow_fallback_count": np.int64(fallback_count),
            "sceneflow_support_count": np.int64(int(own_valid.sum())),
            "sceneflow_next_depth": next_depth,
            "sceneflow_next_valid": next_valid,
            "sceneflow_next_support_count": np.int64(int(next_valid.sum())),
            "sceneflow_compensation_valid": compensation_valid,
            "sceneflow_pair_index": np.asarray(pair, dtype=np.int64),
            "sceneflow_support_frac": np.float64(float(support_up.mean())),
            "sceneflow_depth_source": np.asarray("pair"),
        }
        metadata = stage_metadata(
            "sceneflow_post", args.dataset,
            "scene_flow_refinement residual delta + core gates",
            {
                "pair": pair,
                "trust_abs": trust_abs, "trust_rel": trust_rel, "units": UNITS,
                "detail_policy": "delta0=refined-low をsolver support付き拡大; NDR+delta*(1-edges)へtrust gate",
                "correction_support_count": int(correction_support.sum()),
                "solver_support_count": int(solver_support_up.sum()),
                "fallback_count": fallback_count,
                "next_note": "sceneflow_next_* は compensated grid 情報であり temporal 候補には使わない",
                "upsample": "masked bilinear (0.5支持閾値) / supportは最近傍",
            },
            {"pairs_output": output_path.name}, sys.argv,
        )
        save_stage_frame(dataset_dir, "sceneflow_post", pair[0], arrays, metadata)
        written.append(
            {
                "pair_index": pair,
                "correction_pixels": int(correction_support.sum()),
                "solver_support_pixels": int(solver_support_up.sum()),
                "next_valid_pixels": int(next_valid.sum()),
            }
        )

    # ペア先頭になれないフレーム（各clipの末尾）は NDR 深度で明示 fallback
    covered = {entry["pair_index"][0] for entry in written}
    fallback = []
    for record in frames:
        frame_index = int(record["frame_index"])
        if frame_index in covered:
            continue
        ndr = load_frame_npz(dataset_dir, "ndr", frame_index)
        if ndr is None:
            raise SystemExit(f"sceneflow_post fallback: ndr stage 未生成 frame {frame_index}")
        depth = np.asarray(ndr["ndr_depth"], dtype=np.float64)
        valid = np.asarray(ndr["ndr_valid"], dtype=bool) & np.isfinite(depth) & (depth > 0)
        arrays = {
            "sceneflow_depth": np.where(valid, depth, np.nan).astype(np.float32),
            "sceneflow_valid": valid,
            "sceneflow_correction_support": np.zeros((height, width), dtype=bool),
            "sceneflow_correction_count": np.int64(0),
            "sceneflow_solver_support": np.zeros((height, width), dtype=bool),
            "sceneflow_solver_support_count": np.int64(0),
            "sceneflow_fallback_mask": valid,
            "sceneflow_fallback_count": np.int64(int(valid.sum())),
            "sceneflow_support_count": np.int64(int(valid.sum())),
            "sceneflow_next_depth": np.full((height, width), np.nan, dtype=np.float32),
            "sceneflow_next_valid": np.zeros((height, width), dtype=bool),
            "sceneflow_next_support_count": np.int64(0),
            "sceneflow_compensation_valid": np.zeros((height, width), dtype=bool),
            "sceneflow_pair_index": np.asarray([-1, -1], dtype=np.int64),
            "sceneflow_support_frac": np.float64(0.0),
            "sceneflow_depth_source": np.asarray("ndr_fallback"),
        }
        metadata = stage_metadata(
            "sceneflow_post", args.dataset, "ndr fallback",
            {"reason": "no pair as first frame (clip末尾)", "units": UNITS},
            {"ndr": frame_index}, sys.argv, status="fallback",
        )
        save_stage_frame(dataset_dir, "sceneflow_post", frame_index, arrays, metadata)
        fallback.append(frame_index)

    total_pixels = height * width * max(1, len(written))
    atomic_write_json(
        dataset_dir / "configs" / "stage_configs" / "sceneflow_post.json",
        {
            "dataset": args.dataset,
            "gates": {"trust_abs": trust_abs, "trust_rel": trust_rel,
                      "note": "trust は current (NDR) 基準。support/visibility とAND",
                      "units": UNITS},
            "detail_policy": ("delta0=refined-low をsolver support付きで拡大し NDR+delta*(1-edges) へ"
                              "trust gate。無支持/棄却は NDR 保持。sceneflow_valid==ndr_valid"
                              " (support inflation なし)。correction/solver support を別mask/countで保存"),
            "correction_pixels_total": int(sum(entry["correction_pixels"] for entry in written)),
            "solver_support_pixels_total": int(sum(entry["solver_support_pixels"] for entry in written)),
            "pairs": written,
            "fallback_frames": fallback,
            "rejected_pixel_fraction": float(rejected_pixels / max(1, total_pixels)),
            "support_fraction_mean": float(support_pixels / max(1, total_pixels)),
            "upsample": "masked bilinear (support閾値0.5), support/visibilityは最近傍",
            "units": UNITS,
            "metric_calibrated": False,
            "argv": list(sys.argv),
            "created_at": utc_now(),
        },
    )
    print(f"sceneflow_post: pairs={len(written)} fallback={len(fallback)} "
          f"rejected={rejected_pixels / max(1, total_pixels):.4f}")
    return 0


# --- workers: sr / temporal / final / render --------------------------------


def worker_sr(args: argparse.Namespace) -> int:
    """scene flow super-resolution: 低解像度3D変位を元解像度へ持ち上げる (core実装)。"""
    from arrow.depth_detail import superresolve_scene_flow

    dataset_dir = Path(args.dataset_dir)
    frames_payload = read_frame_list(Path(args.frames))
    frames = frames_payload["frames"]
    height, width = (int(v) for v in frames_payload["original_size"])
    pair_list = load_json(dataset_dir / "stages" / "sceneflow" / "pairs.json")
    outputs_dir = dataset_dir / "stages" / "sceneflow" / "pairs" / "outputs"

    stacked_flow = []
    stacked_valid = []
    stacked_low = []
    stacked_low_valid = []
    stacked_pair = []
    stacked_low_shape = []
    entries = []
    bilinear_diffs = []
    bilinear_rel = []
    low_consistency = []
    sr_valid_fractions = []
    for entry in pair_list["pairs"]:
        pair = entry["pair_index"]
        with np.load(outputs_dir / f"pair_{pair[0]:06d}_{pair[1]:06d}.npz", allow_pickle=False) as data:
            low_flow = np.asarray(data["scene_flow"], dtype=np.float64)
            support = np.asarray(data["support"], dtype=bool)
            low_size = [int(v) for v in data["low_size"]]
        ndr_frame = load_frame_npz(dataset_dir, "ndr", pair[0])
        if ndr_frame is None:
            raise SystemExit(f"ndr 未生成: frame {pair[0]}")
        # high_depth guide は元高解像度 NDR（coarse再拡大は使わない）
        high_depth = np.asarray(ndr_frame["ndr_depth"], dtype=np.float64)
        high_valid = np.asarray(ndr_frame["ndr_valid"], dtype=bool) & np.isfinite(high_depth) & (high_depth > 0)
        with np.load(entry["file"], allow_pickle=False) as data:
            valid0_low = np.asarray(data["valid0_low"], dtype=bool)
            valid1_in_0 = np.asarray(data["valid1_in_0"], dtype=bool)
        rgb = read_rgb(Path(frames[pair[0]]["rgb_path"]))
        # low_valid = 実solver support AND low depth 支持（両フレーム）
        low_valid = support & valid0_low & valid1_in_0 & np.isfinite(low_flow).all(axis=-1)
        result = superresolve_scene_flow(
            low_flow=np.where(low_valid[..., None], low_flow, np.nan),
            high_depth=np.where(high_valid, high_depth, np.nan),
            high_rgb=rgb,
            low_valid=low_valid,
        )
        flow = np.asarray(result["flow"], dtype=np.float64)
        valid = np.asarray(result["valid"], dtype=bool) & np.isfinite(flow).all(axis=-1)
        flow = np.where(valid[..., None], flow, np.nan)
        bilinear_flow, bilinear_valid = upsample_bilinear_masked(
            np.where(low_valid[..., None], low_flow, np.nan), (height, width), valid=low_valid
        )
        joint = valid & bilinear_valid
        if int(joint.sum()) > 0:
            diff = np.abs(flow[joint] - bilinear_flow[joint])
            bilinear_diffs.append(float(diff.mean()))
            magnitude = np.linalg.norm(bilinear_flow[joint], axis=-1)
            denominator = float(magnitude.mean())
            bilinear_rel.append(float(diff.mean() / denominator) if denominator > 1e-12 else np.nan)
        sr_low_flow, sr_low_valid = downsample_bilinear_masked(flow, valid, low_size)
        low_joint = sr_low_valid & low_valid
        if int(low_joint.sum()) > 0:
            low_consistency.append(float(np.abs(sr_low_flow[low_joint] - low_flow[low_joint]).mean()))
        sr_valid_fractions.append(float(valid.mean()))

        arrays = {
            "sr_flow": np.where(valid[..., None], flow, np.nan).astype(np.float32),
            "sr_valid": valid,
            "sr_support_count": np.int64(int(valid.sum())),
            "sr_flow_low": low_flow.astype(np.float32),
            "sr_low_valid": low_valid,
            "sr_pair_index": np.asarray(pair, dtype=np.int64),
        }
        metadata = stage_metadata(
            "sr", args.dataset, "arrow.depth_detail.superresolve_scene_flow",
            {
                "pair": pair,
                "low_size": low_size,
                "note": "3D変位は解像度比で倍率変更しない (contract (d))",
                "valid": "high depth正有限 AND 深度支持重み>0 AND low支持",
                "units": UNITS,
            },
            {"pairs_output": f"pair_{pair[0]:06d}_{pair[1]:06d}.npz"}, sys.argv,
        )
        save_stage_frame(dataset_dir, "sr", pair[0], arrays, metadata)
        stacked_flow.append(arrays["sr_flow"])
        stacked_valid.append(valid)
        stacked_low.append(low_flow.astype(np.float32))
        stacked_low_valid.append(low_valid)
        stacked_pair.append(np.asarray(pair, dtype=np.int64))
        stacked_low_shape.append(np.asarray([low_size[0], low_size[1]], dtype=np.int32))
        entries.append(
            {
                "pair_index": pair,
                "valid_fraction": float(valid.mean()),
                "low_valid_fraction": float(low_valid.mean()),
            }
        )
        if len(entries) % 10 == 0:
            print(f"sr: {len(entries)}/{len(pair_list['pairs'])} pairs", flush=True)
    if not stacked_flow:
        raise SystemExit("sr: 有効なペアがありません")

    stacked = {
        "pair_index": np.stack(stacked_pair, axis=0),
        "flow3d": np.stack(stacked_flow, axis=0),
        "flow3d_valid": np.stack(stacked_valid, axis=0),
        "low_flow": np.stack(stacked_low, axis=0),
        "low_shape": np.stack(stacked_low_shape, axis=0),
    }
    save_npz_atomic(
        dataset_dir / "stages" / "sr.npz",
        stacked,
        stage_metadata(
            "sr", args.dataset, "arrow.depth_detail.superresolve_scene_flow",
            {"pairs": len(stacked_flow), "units": UNITS},
            {"pairs_json": "stages/sceneflow/pairs.json"}, sys.argv,
        ),
    )
    def _agg(values):
        finite = [v for v in values if np.isfinite(v)]
        if not finite:
            return None
        return {"mean": float(np.mean(finite)), "median": float(np.median(finite)), "count": len(finite)}

    # pair 起点にならないフレーム（各clip末尾）には明示的な空エントリを書く（キー欠落を作らない）
    covered = {item["pair_index"][0] for item in pair_list["pairs"]}
    empty_frames = []
    low_size = [int(v) for v in pair_list.get("low_size", [0, 0])]
    for record in frames:
        frame_index = int(record["frame_index"])
        if frame_index in covered:
            continue
        if low_size[0] >= 2 and low_size[1] >= 2:
            flow_low = np.full((low_size[0], low_size[1], 3), np.nan, dtype=np.float32)
            low_valid = np.zeros((low_size[0], low_size[1]), dtype=bool)
        else:
            flow_low = np.full((1, 1, 3), np.nan, dtype=np.float32)
            low_valid = np.zeros((1, 1), dtype=bool)
        arrays = {
            "sr_flow": np.full((height, width, 3), np.nan, dtype=np.float32),
            "sr_valid": np.zeros((height, width), dtype=bool),
            "sr_support_count": np.int64(0),
            "sr_flow_low": flow_low,
            "sr_low_valid": low_valid,
            "sr_pair_index": np.asarray([-1, -1], dtype=np.int64),
        }
        metadata = stage_metadata(
            "sr", args.dataset, "arrow.depth_detail.superresolve_scene_flow",
            {"reason": "no pair as first frame (clip末尾)", "units": UNITS},
            {"frame": frame_index}, sys.argv, status="empty",
        )
        save_stage_frame(dataset_dir, "sr", frame_index, arrays, metadata)
        empty_frames.append(frame_index)

    atomic_write_json(
        dataset_dir / "configs" / "stage_configs" / "sr.json",
        {
            "dataset": args.dataset,
            "api": "arrow.depth_detail.superresolve_scene_flow",
            "pairs": len(entries),
            "high_depth_source": "original high-res NDR (coarse再拡大は使わない)",
            "low_valid": "solver support AND valid0_low AND valid1_in_0",
            "bilinear_vs_sr": {
                "abs_diff": _agg(bilinear_diffs),
                "relative_to_bilinear_magnitude": _agg(bilinear_rel),
                "compared_pairs": len(bilinear_diffs),
                "note": "bilinearは比較基準のみ (SRの暗黙fallbackではない)",
            },
            "low_consistency_abs_diff": _agg(low_consistency),
            "valid_fraction": _agg(sr_valid_fractions),
            "empty_frames": {str(frame): "no pair as first frame" for frame in empty_frames},
            "gt_gate": "合成step-motion GTでのbilinear比較はcore/auditの合成検証に委譲 (D8)",
            "units": UNITS,
            "metric_calibrated": False,
            "argv": list(sys.argv),
            "created_at": utc_now(),
        },
    )
    print(f"sr: {len(entries)} pairs, valid_fraction mean "
          f"{float(np.mean(sr_valid_fractions)):.4f}")
    return 0


def _grad_magnitude(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    filled = np.where(valid, depth, 0.0)
    gy, gx = np.gradient(filled)
    magnitude = np.hypot(gx, gy)
    magnitude[~valid] = np.nan
    return magnitude


def worker_temporal(args: argparse.Namespace) -> int:
    """前後候補の camera 補償済み時間融合。共通支持maskで残差を比較し、悪化はrejectする。"""
    from arrow.depth_detail import temporal_fuse

    dataset_dir = Path(args.dataset_dir)
    frames_payload = read_frame_list(Path(args.frames))
    frames = frames_payload["frames"]
    height, width = (int(v) for v in frames_payload["original_size"])
    weight = float(args.weight)
    trust_abs = float(args.trust_abs)
    trust_rel = float(args.trust_rel)
    gate_tol = float(args.gate_tolerance)
    edge_threshold = float(args.edge_threshold)

    per_frame = []
    failed_frames = []
    rejected_frames = []
    worsened_frames = []
    all_consumed = []
    denom_total = 0
    pairs_compared = 0
    for record in frames:
        frame_index = int(record["frame_index"])
        current_frame = load_frame_npz(dataset_dir, "sceneflow_post", frame_index)
        geometry = load_frame_npz(dataset_dir, "geometry", frame_index)
        prep = load_frame_npz(dataset_dir, "prep", frame_index)
        if current_frame is None or geometry is None or prep is None:
            raise SystemExit(f"temporal: stage 未生成 frame {frame_index}")
        current = np.asarray(current_frame["sceneflow_depth"], dtype=np.float64)
        edges = np.asarray(prep["prep_edges"], dtype=np.float64)
        current_in = np.array(current, copy=True)

        pair_index = np.asarray(current_frame["sceneflow_pair_index"], dtype=np.int64)
        candidates = []
        rejections = []
        consumed_pairs = []
        cur_ok = np.isfinite(current) & (current > 0)

        def _geometry_ready(payload: dict | None) -> bool:
            if payload is None:
                return False
            return bool(
                bool(payload["arrow_pose_valid"]) and bool(payload["arrow_pose_frame0_valid"])
                and np.all(np.isfinite(np.asarray(payload["arrow_K"], dtype=np.float64)))
                and np.all(np.isfinite(np.asarray(payload["arrow_camera_to_metric"], dtype=np.float64)))
            )

        K0 = np.asarray(geometry["arrow_K"], dtype=np.float64)
        metric0 = np.asarray(geometry["arrow_camera_to_metric"], dtype=np.float64)

        # fwd: 自ペア (f, f+1) の SR 3D変位 S から motion correspondence 付き候補を作る
        if pair_index[0] == frame_index:
            sr_frame = load_frame_npz(dataset_dir, "sr", frame_index)
            next_frame = load_frame_npz(dataset_dir, "sceneflow_post", frame_index + 1)
            next_geometry = load_frame_npz(dataset_dir, "geometry", frame_index + 1)
            if (
                sr_frame is None or next_frame is None or next_geometry is None
                or not _geometry_ready(next_geometry) or not _geometry_ready(geometry)
            ):
                rejections.append({"direction": "fwd", "reason": "sr_or_pose_missing"})
            else:
                K1 = np.asarray(next_geometry["arrow_K"], dtype=np.float64)
                metric1 = np.asarray(next_geometry["arrow_camera_to_metric"], dtype=np.float64)
                flow = np.asarray(sr_frame["sr_flow"], dtype=np.float64)
                flow_valid = np.asarray(sr_frame["sr_valid"], dtype=bool)
                depth1 = np.asarray(next_frame["sceneflow_depth"], dtype=np.float64)
                target_from_source = camera_transform_between(metric0, metric1)
                motion = motion_candidate_from_sr(
                    current, K0, depth1, K1, target_from_source, flow, flow_valid,
                    trust_abs, trust_rel,
                )
                candidates.append(
                    {
                        "direction": "fwd",
                        "depth": np.asarray(motion["candidate"], dtype=np.float64),
                        "support": np.asarray(motion["support"], dtype=bool),
                        "residual": "reprojection",
                        "zpred1": np.asarray(motion["zpred1"], dtype=np.float64),
                        "d1_observed": np.asarray(motion["d1_observed"], dtype=np.float64),
                        "target_from_source": target_from_source,
                        "K0": K0,
                        "flow": flow,
                        "pair": [int(frame_index), int(frame_index + 1)],
                    }
                )

        # bwd: 前ペア (f-1, f) の P1_0 (=D0*r0+S) を current camera へ forward z-buffer splat
        previous_frame = load_frame_npz(dataset_dir, "sceneflow_post", frame_index - 1) if frame_index > 0 else None
        if previous_frame is not None:
            previous_pair = np.asarray(previous_frame["sceneflow_pair_index"], dtype=np.int64)
            if previous_pair[0] == frame_index - 1 and previous_pair[1] == frame_index:
                previous_geometry = load_frame_npz(dataset_dir, "geometry", frame_index - 1)
                previous_sr = load_frame_npz(dataset_dir, "sr", frame_index - 1)
                if (
                    previous_sr is None or previous_geometry is None
                    or not _geometry_ready(previous_geometry) or not _geometry_ready(geometry)
                ):
                    rejections.append({"direction": "bwd", "reason": "sr_or_pose_missing"})
                else:
                    K_prev = np.asarray(previous_geometry["arrow_K"], dtype=np.float64)
                    metric_prev = np.asarray(previous_geometry["arrow_camera_to_metric"], dtype=np.float64)
                    flow_prev = np.asarray(previous_sr["sr_flow"], dtype=np.float64)
                    flow_valid_prev = np.asarray(previous_sr["sr_valid"], dtype=bool)
                    depth0_prev = np.asarray(previous_frame["sceneflow_depth"], dtype=np.float64)
                    target_from_source = camera_transform_between(metric_prev, metric0)
                    # P1_0 = D0*r0 + S の全XYZを forward z-buffer scatter（S_x/S_yも反映）
                    splat = splat_scene_flow(
                        depth=depth0_prev,
                        K_source=K_prev,
                        K_target=K0,
                        target_from_source=target_from_source,
                        flow=flow_prev,
                        valid=flow_valid_prev,
                    )
                    splat_depth = np.asarray(splat["depth"], dtype=np.float64)
                    splat_valid = (
                        np.asarray(splat["valid"], dtype=bool)
                        & np.isfinite(splat_depth) & (splat_depth > 0)
                    )
                    candidates.append(
                        {
                            "direction": "bwd",
                            "depth": np.where(splat_valid, splat_depth, np.nan),
                            # z-buffer 支持のみ。原座標 source_valid とは AND しない（統括指示）
                            "support": splat_valid,
                            "residual": "splat",
                            "pair": [int(frame_index - 1), int(frame_index)],
                        }
                    )

        stats = np.full(8, np.nan, dtype=np.float64)
        accepted_fwd = False
        accepted_bwd = False
        common_union = np.zeros((height, width), dtype=bool)
        for candidate in candidates:
            support = np.asarray(candidate["support"], dtype=bool)
            depth = np.asarray(candidate["depth"], dtype=np.float64)
            cand_ok = np.isfinite(depth) & (depth > 0)
            common = support & cur_ok & cand_ok
            common_union |= common
            index = 0 if candidate["direction"] == "fwd" else 4
            n = int(common.sum())
            if n == 0:
                rejections.append({"direction": candidate["direction"], "reason": "zero_support",
                                   "denominator": 0})
                continue
            pairs_compared += 1
            if candidate["residual"] == "reprojection":
                residual_map = np.abs(
                    np.asarray(candidate["zpred1"], dtype=np.float64)
                    - np.asarray(candidate["d1_observed"], dtype=np.float64)
                )
            else:
                residual_map = np.abs(depth - current)
            before_median = float(np.median(residual_map[common]))
            fused = temporal_fuse(
                current=current,
                candidate=depth,
                support=support,
                edges=edges,
                weight=weight,
                trust_abs=trust_abs,
                trust_rel=trust_rel,
            )
            if candidate["residual"] == "reprojection":
                zpred_after = reproject_zpred1(
                    fused, candidate["K0"], candidate["target_from_source"], candidate["flow"]
                )
                residual_after_map = np.abs(
                    zpred_after - np.asarray(candidate["d1_observed"], dtype=np.float64)
                )
            else:
                residual_after_map = np.abs(depth - fused)
            after_median = float(np.median(residual_after_map[common]))
            stats[index] = before_median
            stats[index + 1] = after_median
            stats[index + 2] = float(n)
            stats[index + 3] = float(n)
            consumed_pairs.append(
                {"pair": candidate["pair"], "consumed_for": frame_index,
                 "direction": candidate["direction"], "mode": candidate["residual"]}
            )
            worsened = after_median > before_median * (1.0 + gate_tol)
            if worsened:
                rejections.append(
                    {"direction": candidate["direction"], "reason": "worsened",
                     "before": before_median, "after": after_median}
                )
                continue
            current = fused
            if candidate["direction"] == "fwd":
                accepted_fwd = True
            else:
                accepted_bwd = True
        valid_out = np.isfinite(current) & (current > 0)
        denom = int(common_union.sum())
        denom_total += denom
        if denom == 0:
            failed_frames.append({"frame_index": frame_index, "reason": "temporal_denom_zero"})
        if rejections:
            rejected_frames.append({"frame_index": frame_index, "rejections": rejections})
        if any(item["reason"] == "worsened" for item in rejections):
            worsened_frames.append(frame_index)

        edge_mask = edges >= edge_threshold
        grad_before = _grad_magnitude(current_in)
        grad_after = _grad_magnitude(current)
        edge_pixels = edge_mask & np.isfinite(grad_before) & np.isfinite(grad_after)
        leakage = (
            float(np.median(grad_before[edge_pixels]) - np.median(grad_after[edge_pixels]))
            if int(edge_pixels.sum()) > 0 else np.nan
        )
        anchor = load_frame_npz(dataset_dir, "anchor", frame_index)
        if anchor is None:
            raise SystemExit(f"temporal: anchor stage 未生成 frame {frame_index}")
        anchor_depth = np.asarray(anchor["arrow_depth"], dtype=np.float64)
        anchor_valid = np.asarray(anchor["arrow_valid"], dtype=bool)
        compare = anchor_valid & valid_out
        anchor_deviation = (
            float(np.median(np.abs(current[compare] - anchor_depth[compare]) / anchor_depth[compare]))
            if int(compare.sum()) > 0 else np.nan
        )
        anchor_deviation_before = (
            float(np.median(np.abs(current_in[compare] - anchor_depth[compare]) / anchor_depth[compare]))
            if int(compare.sum()) > 0 else np.nan
        )

        arrays = {
            "temporal_depth": np.where(valid_out, current, np.nan).astype(np.float32),
            "temporal_valid": valid_out,
            "temporal_support_count": np.int64(int(valid_out.sum())),
            "temporal_common_support": common_union,
            "temporal_denom": np.int64(denom),
            "temporal_accepted_fwd": np.bool_(accepted_fwd),
            "temporal_accepted_bwd": np.bool_(accepted_bwd),
            "temporal_residual_stats": stats,
            "temporal_edge_leakage": np.float64(leakage),
            "temporal_anchor_deviation": np.float64(anchor_deviation),
            "temporal_anchor_deviation_before": np.float64(anchor_deviation_before),
            "temporal_current_in": current_in.astype(np.float32),
        }
        metadata = stage_metadata(
            "temporal", args.dataset, "arrow.depth_detail.temporal_fuse",
            {
                "frame_index": frame_index,
                "weight": weight, "trust_abs": trust_abs, "trust_rel": trust_rel,
                "gate_tolerance": gate_tol,
                "candidate_source": "sr_flow_3d (motion correspondence; 同pixel比較なし)",
                "residual_definition": {
                    "fwd": "共通maskで zpred1 対 D1obs の再投影z残差 (before/after)",
                    "bwd": "共通maskで splat候補深度 対 current/fused の残差",
                },
                "candidates": [candidate["direction"] for candidate in candidates],
                "consumed_pairs": consumed_pairs,
                "rejections": rejections,
                "units": UNITS,
            },
            {"sceneflow_post": frame_index, "sr": frame_index}, sys.argv,
        )
        save_stage_frame(dataset_dir, "temporal", frame_index, arrays, metadata)
        per_frame.append(
            {
                "frame_index": frame_index,
                "denom": denom,
                "accepted_fwd": accepted_fwd,
                "accepted_bwd": accepted_bwd,
                "rejections": rejections,
                "residual_stats": stats.tolist(),
                "edge_leakage": leakage if np.isfinite(leakage) else None,
                "anchor_deviation": anchor_deviation if np.isfinite(anchor_deviation) else None,
                "anchor_deviation_before": (
                    anchor_deviation_before if np.isfinite(anchor_deviation_before) else None
                ),
                "consumed_pairs": consumed_pairs,
            }
        )
        all_consumed.extend(consumed_pairs)

    def _median_of(values):
        finite = [v for v in values if v is not None and np.isfinite(v)]
        return float(np.median(finite)) if finite else None

    # SR pair 消費の確認: 全pairのSR値が最終depth対応付けに使われたかを記録
    sr_config_path = dataset_dir / "configs" / "stage_configs" / "sr.json"
    sr_pairs_expected = None
    if sr_config_path.is_file():
        try:
            sr_pairs_expected = int(load_json(sr_config_path).get("pairs", 0))
        except (OSError, ValueError, TypeError):
            sr_pairs_expected = None
    consumption: dict[str, list[dict]] = {}
    for item in all_consumed:
        key = f"{item['pair'][0]}_{item['pair'][1]}"
        consumption.setdefault(key, []).append(
            {"consumed_for": item["consumed_for"], "direction": item["direction"], "mode": item["mode"]}
        )
    unconsumed = []
    if sr_pairs_expected is not None:
        pairs_path = dataset_dir / "stages" / "sceneflow" / "pairs.json"
        if pairs_path.is_file():
            pair_keys = [
                f"{item['pair_index'][0]}_{item['pair_index'][1]}"
                for item in load_json(pairs_path)["pairs"]
            ]
            consumed_keys = set(consumption)
            unconsumed = [key for key in pair_keys if key not in consumed_keys]

    before_values = [item["residual_stats"][0] for item in per_frame] + [
        item["residual_stats"][4] for item in per_frame
    ]
    after_values = [item["residual_stats"][1] for item in per_frame] + [
        item["residual_stats"][5] for item in per_frame
    ]
    summary = {
        "dataset": args.dataset,
        "pairs_compared": pairs_compared,
        "denom_total": denom_total,
        "error_before": _median_of(before_values),
        "error_after": _median_of(after_values),
        "worsened_frames": len(worsened_frames),
        "rejected_frames": len(rejected_frames),
        "failed_frames": failed_frames,
        "candidate_source": "sr_flow_3d (motion correspondence; sceneflow_next_depthの同pixel比較は不使用)",
        "residual_definition": {
            "fwd": "再投影z残差 |zpred1 - D1obs| (共通mask, before/after)",
            "bwd": "forward z-buffer splat 候補 vs current/fused (共通mask)",
        },
        "sr_pair_consumption": consumption,
        "sr_pairs_expected": sr_pairs_expected,
        "sr_pairs_consumed": len(consumption),
        "sr_pairs_unconsumed": unconsumed,
        "edge_leakage_median": _median_of([item["edge_leakage"] for item in per_frame]),
        "anchor_deviation_median": _median_of([item["anchor_deviation"] for item in per_frame]),
        "anchor_deviation_before_median": _median_of(
            [item["anchor_deviation_before"] for item in per_frame]
        ),
        "weight": weight, "trust_abs": trust_abs, "trust_rel": trust_rel,
        "gate": "accepted は after_median <= before_median*(1+gate_tolerance) のみ; 悪化はreject記録",
        "zero_denominator_policy": "denom==0 は failed_frames に記録し非0終了（黙認しない）",
        "units": UNITS,
        "metric_calibrated": False,
        "per_frame": per_frame,
        "argv": list(sys.argv),
        "created_at": utc_now(),
    }
    atomic_write_json(dataset_dir / "configs" / "stage_configs" / "temporal.json", summary)
    print(f"temporal: denom_total={denom_total}, pairs={pairs_compared}, "
          f"worsened={len(worsened_frames)}, rejected={len(rejected_frames)}")
    if failed_frames:
        print(f"temporal: 0分母フレームがあります: {[item['frame_index'] for item in failed_frames]}",
              file=sys.stderr)
        return 2
    return 0


STAGE_FRAME_KEYS = {
    "anchor": ("arrow_depth", "arrow_valid", "arrow_scale", "arrow_clip_index", "arrow_support_count"),
    "geometry": (
        "arrow_K", "arrow_K_valid", "arrow_pose_valid", "arrow_pose_frame0_valid",
        "arrow_pose_reference_index", "arrow_camera_to_reference", "arrow_camera_to_reconstruct",
        "arrow_camera_to_metric",
    ),
    "prep": ("prep_edges", "prep_normal", "prep_normal_valid", "prep_normal_count"),
    "lingbot": (
        "lingbot_depth_in", "lingbot_depth_raw", "lingbot_valid", "lingbot_mask",
        "lingbot_K_norm", "lingbot_K_pixel", "lingbot_support_count",
    ),
    "lingbot_fix": (
        "lingbot_depth", "lingbot_scale_fix", "lingbot_scale_raw", "lingbot_fix_weight",
        "lingbot_fix_inliers", "lingbot_fix_residual",
    ),
    "ndr": ("ndr_depth_in", "ndr_depth", "ndr_valid", "ndr_support_count"),
    "sceneflow_post": (
        "sceneflow_depth", "sceneflow_valid", "sceneflow_support_count", "sceneflow_next_depth",
        "sceneflow_next_valid", "sceneflow_next_support_count", "sceneflow_compensation_valid",
        "sceneflow_pair_index", "sceneflow_support_frac", "sceneflow_depth_source",
        "sceneflow_correction_support", "sceneflow_correction_count",
        "sceneflow_solver_support", "sceneflow_solver_support_count",
        "sceneflow_fallback_mask", "sceneflow_fallback_count",
    ),
    "sr": ("sr_flow", "sr_valid", "sr_support_count", "sr_flow_low", "sr_low_valid", "sr_pair_index"),
    "temporal": (
        "temporal_depth", "temporal_valid", "temporal_support_count", "temporal_common_support",
        "temporal_denom", "temporal_accepted_fwd", "temporal_accepted_bwd", "temporal_residual_stats",
    ),
    "final": ("final_depth", "final_valid", "final_support_count"),
}
STAGE_CHAIN = "arrow->anchor->lingbot->ndr->sceneflow->sr->temporal"
STAGE_CHAIN_FULL = "arrow->align->anchor->geometry->lingbot->prep->ndr->sceneflow->sr->temporal->final"


def worker_final(args: argparse.Namespace) -> int:
    """per-frame NPZ pack + final.npz + 全フレーム共通 depth range を確定する。"""
    dataset_dir = Path(args.dataset_dir)
    frames_payload = read_frame_list(Path(args.frames))
    frames = frames_payload["frames"]
    height, width = (int(v) for v in frames_payload["original_size"])
    count = len(frames)

    temporal_depths = []
    sampled = []
    for record in frames:
        frame_index = int(record["frame_index"])
        temporal = load_frame_npz(dataset_dir, "temporal", frame_index)
        if temporal is None:
            raise SystemExit(f"temporal 未生成: frame {frame_index}")
        depth = np.asarray(temporal["temporal_depth"], dtype=np.float64)
        valid = np.asarray(temporal["temporal_valid"], dtype=bool) & np.isfinite(depth) & (depth > 0)
        temporal_depths.append((depth, valid))
        sampled.append(depth[::4, ::4][valid[::4, ::4]])
    all_values = np.concatenate([s for s in sampled if s.size > 0])
    if all_values.size == 0:
        raise SystemExit("final: 有効なfinal深度がありません")
    depth_range = [float(v) for v in np.percentile(all_values, [args.range_low, args.range_high])]

    configs_dir = dataset_dir / "configs" / "stage_configs"
    stage_summary = {}
    for path in sorted(configs_dir.glob("*.json")) if configs_dir.is_dir() else []:
        try:
            payload = load_json(path)
        except (OSError, ValueError):
            continue
        stage_summary[path.stem] = {
            "model": payload.get("model"),
            "created_at": payload.get("created_at"),
            "config_sha256": sha256_file(path),
        }
    invocation_path = dataset_dir / "configs" / "run_invocation.json"
    invocation = load_json(invocation_path) if invocation_path.is_file() else None
    signatures_path = dataset_dir / "signatures.json"
    if signatures_path.is_file():
        chain_hash = sha256_bytes((canonical_json(load_json(signatures_path))).encode("utf-8"))
    else:
        chain_hash = sha256_bytes(
            canonical_json({name: value["config_sha256"] for name, value in stage_summary.items()}).encode("utf-8")
        )
    model_names = sorted(
        {
            str(value["model"])
            for value in stage_summary.values()
            if isinstance(value.get("model"), str) and value.get("model")
        }
    )
    merged_metadata = {
        "format_version": FORMAT_VERSION,
        "stage": "merged",
        "dataset": args.dataset,
        "model": ", ".join(model_names) if model_names else "unknown",
        "params": {"stages": stage_summary, "depth_range_method": f"p{args.range_low}-p{args.range_high}"},
        "input_sha256": {"stage_chain": chain_hash},
        "argv": (invocation or {}).get("argv", list(sys.argv)),
        "status": "complete",
        "created_at": utc_now(),
        "units": UNITS,
        "metric_calibrated": False,
        "stages": stage_summary,
        "calibration": {
            "single_scalar": 1.0,
            "applied": False,
            "applies_to": ["depth", "camera_translation", "scene_flow_3d"],
        },
    }

    final_stack = np.full((count, height, width), np.nan, dtype=np.float32)
    valid_stack = np.zeros((count, height, width), dtype=bool)
    source_index = np.zeros(count, dtype=np.int64)
    missing = []
    for record in frames:
        frame_index = int(record["frame_index"])
        source_index[frame_index] = int(record["source_index"])
        depth, valid = temporal_depths[frame_index]
        final_depth = np.where(valid, depth, np.nan).astype(np.float32)
        final_stack[frame_index] = final_depth
        valid_stack[frame_index] = valid
        save_stage_frame(
            dataset_dir,
            "final",
            frame_index,
            {
                "final_depth": final_depth,
                "final_valid": valid,
                "final_support_count": np.int64(int(valid.sum())),
            },
            stage_metadata(
                "final", args.dataset, "arrow.depth_detail final aggregation",
                {"depth_range": depth_range, "units": UNITS}, {"temporal": frame_index}, sys.argv,
            ),
        )

        merged = {
            "frame_source_index": np.int64(int(record["source_index"])),
            "frame_rgb_height": np.int32(height),
            "frame_rgb_width": np.int32(width),
            "final_depth": final_depth,
            "final_valid": valid,
            "final_support_count": np.int64(int(valid.sum())),
        }
        for stage, keys in STAGE_FRAME_KEYS.items():
            if stage == "final":
                continue
            payload = load_frame_npz(dataset_dir, stage, frame_index)
            if payload is None:
                missing.append({"stage": stage, "frame_index": frame_index})
                continue
            for key in keys:
                if key in payload:
                    merged[key] = payload[key]
        save_npz_atomic(
            dataset_dir / "frames" / f"{frame_index:06d}.npz",
            merged,
            merged_metadata,
            compressed=True,
        )

    save_npz_atomic(
        dataset_dir / "stages" / "final.npz",
        {
            "depth": final_stack,
            "valid": valid_stack,
            "source_index": source_index,
            "depth_range": np.asarray(depth_range, dtype=np.float64),
            "stage_chain": np.asarray(STAGE_CHAIN),
            "stage_chain_full": np.asarray(STAGE_CHAIN_FULL),
            "calibration_single_scalar": np.float64(1.0),
        },
        stage_metadata(
            "final", args.dataset, "arrow.depth_detail final aggregation",
            {
                "frames": count, "depth_range": depth_range,
                "stage_chain": STAGE_CHAIN, "stage_chain_full": STAGE_CHAIN_FULL,
                "calibration": {"single_scalar": 1.0, "applied": False,
                                "applies_to": ["depth", "camera_translation", "scene_flow_3d"]},
                "units": UNITS,
            },
            {"signatures_chain": chain_hash}, sys.argv, status="complete" if not missing else "incomplete",
        ),
    )
    atomic_write_json(
        configs_dir / "final.json",
        {
            "dataset": args.dataset,
            "frames": count,
            "depth_range": depth_range,
            "range_method": f"valid final depth p{args.range_low}-p{args.range_high} (共通range)",
            "missing_keys": missing,
            "stage_chain": STAGE_CHAIN,
            "stage_chain_full": STAGE_CHAIN_FULL,
            "calibration": merged_metadata["calibration"],
            "units": UNITS,
            "metric_calibrated": False,
            "argv": list(sys.argv),
            "created_at": utc_now(),
        },
    )
    if missing:
        print(f"final: merged frame NPZ に欠損キーがあります: {len(missing)} 件", file=sys.stderr)
        return 2
    print(f"final: {count} frames, depth_range={depth_range}")
    return 0


def _composite_panels(panels: list[np.ndarray], cell_width: int) -> np.ndarray:
    """同サイズセルへ letterbox して横並び合成する。"""
    cell_height = max(panel.shape[0] for panel in panels)
    canvas = np.full((cell_height + 4, (cell_width + 2) * len(panels) + 2, 3), 12, dtype=np.uint8)
    for index, panel in enumerate(panels):
        left = 2 + index * (cell_width + 2)
        canvas[2 : 2 + panel.shape[0], left : left + panel.shape[1]] = panel
    return canvas


def _annotate_panel(panel: np.ndarray, text: str) -> np.ndarray:
    """panel 左上に小さな見出しを描く（PIL が無ければ無加工）。"""
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore

        image = Image.fromarray(np.asarray(panel, dtype=np.uint8))
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        bbox = draw.textbbox((3, 2), text, font=font)
        draw.rectangle([bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1], fill=(0, 0, 0))
        draw.text((3, 2), text, fill=(255, 230, 90), font=font)
        return np.asarray(image)
    except Exception:
        return panel


def _header_strip(width: int, text: str) -> np.ndarray:
    """合成の上に置く薄い情報帯（frame番号・共通range）。"""
    strip = np.full((16, width, 3), 12, dtype=np.uint8)
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore

        image = Image.fromarray(strip)
        draw = ImageDraw.Draw(image)
        draw.text((4, 3), text, fill=(200, 220, 255), font=ImageFont.load_default())
        return np.asarray(image)
    except Exception:
        return strip


def worker_render(args: argparse.Namespace) -> int:
    """共通rangeの比較MP4/代表PNG/final点群RRDを生成し、ffprobe/RRD検証を記録する。"""
    dataset_dir = Path(args.dataset_dir)
    frames_payload = read_frame_list(Path(args.frames))
    frames = frames_payload["frames"]
    height, width = (int(v) for v in frames_payload["original_size"])
    visuals_dir = dataset_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    with np.load(dataset_dir / "stages" / "final.npz", allow_pickle=False) as data:
        depth_range = [float(v) for v in data["depth_range"]]
    final_range = depth_range

    flow_values = []
    for record in frames:
        frame_index = int(record["frame_index"])
        sr = load_frame_npz(dataset_dir, "sr", frame_index)
        if sr is None:
            continue
        flow = np.asarray(sr["sr_flow"], dtype=np.float64)
        valid = np.asarray(sr["sr_valid"], dtype=bool)
        magnitude = np.linalg.norm(flow, axis=-1)
        flow_values.append(magnitude[::4, ::4][valid[::4, ::4]])
    flow_all = np.concatenate([value for value in flow_values if value.size > 0]) if flow_values else np.zeros(1)
    flow_range = [float(v) for v in np.percentile(flow_all, [2, 98])]

    cell_width = int(args.cell_width)
    panel_order = ["rgb", "arrow", "lingbot", "ndr", "sceneflow", "temporal", "sr_flow"]
    def build_panel(frame_index: int):
        record = frames[frame_index]
        rgb = read_rgb(Path(record["rgb_path"]))
        scale = cell_width / float(rgb.shape[1])
        panel_h = max(2, int(round(rgb.shape[0] * scale)))
        panel_h -= panel_h % 2
        if panel_h < 2:
            panel_h = 2

        def shrink(image):
            low_h, low_w = panel_h, cell_width
            image = np.asarray(image, dtype=np.uint8)
            if image.shape[0] == low_h and image.shape[1] == low_w:
                return image
            try:
                from PIL import Image  # type: ignore

                return np.asarray(
                    Image.fromarray(image).resize((low_w, low_h), Image.Resampling.BILINEAR), dtype=np.uint8
                )
            except ImportError:
                import cv2  # type: ignore

                return cv2.resize(image, (low_w, low_h), interpolation=cv2.INTER_LINEAR)

        rgb_panel = shrink(rgb)
        anchor = load_frame_npz(dataset_dir, "anchor", frame_index)
        lingbot_fix = load_frame_npz(dataset_dir, "lingbot_fix", frame_index)
        lingbot_raw = load_frame_npz(dataset_dir, "lingbot", frame_index)
        lingbot = {**(lingbot_raw or {}), **(lingbot_fix or {})}
        ndr = load_frame_npz(dataset_dir, "ndr", frame_index)
        sceneflow = load_frame_npz(dataset_dir, "sceneflow_post", frame_index)
        temporal = load_frame_npz(dataset_dir, "temporal", frame_index)
        sr = load_frame_npz(dataset_dir, "sr", frame_index)
        def depth_panel(payload, key, valid_key):
            if payload is None:
                return np.full((panel_h, cell_width, 3), 25, dtype=np.uint8)
            depth = np.asarray(payload[key], dtype=np.float64)
            valid = np.asarray(payload[valid_key], dtype=bool) & np.isfinite(depth) & (depth > 0)
            colored = colorize_scalar(depth, final_range[0], final_range[1], valid=valid)
            return shrink(colored)
        flow_panel = np.full((panel_h, cell_width, 3), 25, dtype=np.uint8)
        if sr is not None:
            flow = np.asarray(sr["sr_flow"], dtype=np.float64)
            valid = np.asarray(sr["sr_valid"], dtype=bool)
            magnitude = np.where(valid, np.linalg.norm(flow, axis=-1), np.nan)
            flow_panel = shrink(colorize_scalar(magnitude, flow_range[0], max(flow_range[1], flow_range[0] + 1e-9), valid=valid))
        panels = [
            _annotate_panel(rgb_panel, "RGB"),
            _annotate_panel(depth_panel(anchor, "arrow_depth", "arrow_valid"), "ARROW"),
            _annotate_panel(depth_panel(lingbot, "lingbot_depth", "lingbot_valid"), "LingBot"),
            _annotate_panel(depth_panel(ndr, "ndr_depth", "ndr_valid"), "NDR"),
            _annotate_panel(depth_panel(sceneflow, "sceneflow_depth", "sceneflow_valid"), "Scene flow"),
            _annotate_panel(depth_panel(temporal, "temporal_depth", "temporal_valid"), "Temporal"),
            _annotate_panel(flow_panel, "SR 3D flow"),
        ]
        composite = _composite_panels(panels, cell_width)
        strip = _header_strip(
            composite.shape[1],
            f"frame {frame_index}/{len(frames) - 1}  depth range [{final_range[0]:.2f}, "
            f"{final_range[1]:.2f}] ({UNITS}, metric_calibrated=false)  "
            f"flow range [{flow_range[0]:.4f}, {flow_range[1]:.4f}]",
        )
        return np.concatenate([strip, composite], axis=0)

    fps = float(args.fps)
    mp4_path = visuals_dir / f"{args.dataset}_compare.mp4"
    writer = None
    representative = []
    wanted = {0, len(frames) // 2, len(frames) - 1}
    for record in frames:
        frame_index = int(record["frame_index"])
        composite = build_panel(frame_index)
        if writer is None:
            writer = VideoWriter(mp4_path, composite.shape[1], composite.shape[0], fps)
        writer.append(composite)
        if frame_index in wanted:
            representative.append(composite)
        if frame_index % 20 == 0:
            print(f"render: {frame_index + 1}/{len(frames)} frames", flush=True)
    if writer is None:
        raise SystemExit("render: フレームがありません")
    writer.close()
    try:
        from PIL import Image  # type: ignore

        sheet = np.concatenate(
            [np.concatenate([panel, np.full((6, panel.shape[1], 3), 12, dtype=np.uint8)], axis=0)
             for panel in representative],
            axis=0,
        )
        png_path = visuals_dir / f"{args.dataset}_representative.png"
        Image.fromarray(sheet).save(png_path, format="PNG")
    except ImportError:
        png_path = None
        print("render: PIL が無いため代表PNGを省略（記録）", file=sys.stderr)

    rrd_path = visuals_dir / f"{args.dataset}_final_pointcloud.rrd"
    import rerun as rr

    rr.init(f"depth_detail_{args.dataset}", spawn=False)
    rr.save(str(rrd_path))
    try:
        rr.log("world", rr.ViewCoordinates.RDF, static=True)
    except Exception as error:  # 表示規約のlogに失敗しても記録は続行
        print(f"render: ViewCoordinates をlogできません: {error}", file=sys.stderr)
    stride = int(args.point_stride)
    rrd_frames = 0
    for record in frames:
        frame_index = int(record["frame_index"])
        temporal = load_frame_npz(dataset_dir, "temporal", frame_index)
        geometry = load_frame_npz(dataset_dir, "geometry", frame_index)
        if temporal is None or geometry is None:
            print(f"render: stage 未生成 frame {frame_index} をスキップ", file=sys.stderr)
            continue
        depth_full = np.asarray(temporal["temporal_depth"], dtype=np.float64)
        valid_full = (
            np.asarray(temporal["temporal_valid"], dtype=bool)
            & np.isfinite(depth_full) & (depth_full > 0)
        )
        depth = depth_full[::stride, ::stride]
        valid = valid_full[::stride, ::stride] & np.isfinite(depth) & (depth > 0)
        K = np.asarray(geometry["arrow_K"], dtype=np.float64)
        metric = np.asarray(geometry["arrow_camera_to_metric"], dtype=np.float64)
        rgb_full = read_rgb(Path(record["rgb_path"]))
        rows, cols = np.nonzero(valid)
        rr.set_time("frame", sequence=frame_index)
        sr = load_frame_npz(dataset_dir, "sr", frame_index)
        sr_logged = False
        if rows.size:
            # stride 縮小後は native = index*stride で K（元解像度）と整合させる
            native_x = cols * stride
            native_y = rows * stride
            z = depth[rows, cols]
            x = (native_x - K[0, 2]) * z / K[0, 0]
            y = (native_y - K[1, 2]) * z / K[1, 1]
            points = np.stack([x, y, z], axis=-1)
            rotation = metric[:3, :3]
            translation = metric[:3, 3]
            world = points @ rotation.T + translation
            colors = rgb_full[native_y, native_x].astype(np.uint8)
            rr.log("world/final_points", rr.Points3D(world.astype(np.float32), colors=colors))
            # SR 3D変位をサンプル矢印として同じ world へ（回転のみ適用、倍率変更なし）
            if sr is not None:
                flow = np.asarray(sr["sr_flow"], dtype=np.float64)
                sr_valid = np.asarray(sr["sr_valid"], dtype=bool)
                sample = np.zeros((height, width), dtype=bool)
                sample[:: max(8, stride * 2), :: max(8, stride * 2)] = True
                sample &= sr_valid & valid_full & np.isfinite(flow).all(axis=-1)
                nonzero = np.nonzero(sample)
                srows, scols = nonzero[0], nonzero[1]
                if srows.size:
                    sz = depth_full[srows, scols]
                    sx = (scols - K[0, 2]) * sz / K[0, 0]
                    sy = (srows - K[1, 2]) * sz / K[1, 1]
                    origins = (np.stack([sx, sy, sz], axis=-1) @ rotation.T + translation).astype(np.float32)
                    vectors = (flow[srows, scols] @ rotation.T).astype(np.float32)
                    rr.log("world/sr_arrows", rr.Arrows3D(origins=origins, vectors=vectors,
                                                          colors=np.array([[255, 120, 40]], dtype=np.uint8)))
                    sr_logged = True
            # camera pose と元RGB/深度（縮小）を frame 同期で記録
            rr.log("world/camera", rr.Transform3D(
                translation=translation.astype(np.float32), mat3x3=rotation.astype(np.float32)))
            small_rgb = resize_rgb_low(rgb_full, [max(2, height // 3), max(2, width // 3)])
            rr.log("world/camera/image", rr.Image(small_rgb))
            depth_small = colorize_scalar(
                depth_full[::3, ::3],
                final_range[0], final_range[1],
                valid=valid_full[::3, ::3],
            )
            rr.log("world/camera/depth", rr.Image(depth_small))
            rrd_frames += 1
        if not sr_logged:
            # latest-at の hold-last を防ぐ明示 empty row（rr.Clear は既存readback列を壊すため使わない）
            rr.log("world/sr_arrows", rr.Arrows3D(
                origins=np.zeros((0, 3), dtype=np.float32),
                vectors=np.zeros((0, 3), dtype=np.float32),
            ))
    rr.disconnect()
    ffprobe = shutil.which("ffprobe")
    probe = None
    if ffprobe:
        probe_run = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,codec_name,nb_frames", "-of", "json", str(mp4_path)],
            capture_output=True, text=True,
        )
        try:
            probe = {"returncode": probe_run.returncode,
                     "json": json.loads(probe_run.stdout) if probe_run.stdout else None}
        except json.JSONDecodeError:
            probe = {"returncode": probe_run.returncode, "stdout": probe_run.stdout}
    rrd_verification: dict[str, object] = {"method": "presence_and_magic"}
    rrd_head = rrd_path.read_bytes()[:4] if rrd_path.is_file() else b""
    rrd_verification.update(
        {"exists": rrd_path.is_file(), "bytes": rrd_path.stat().st_size if rrd_path.is_file() else 0,
         "head": rrd_head.decode("latin-1")}
    )
    rerun_cli = shutil.which("rerun")
    if rerun_cli:
        verify_run = subprocess.run([rerun_cli, "rrd", "verify", str(rrd_path)],
                                    capture_output=True, text=True)
        rrd_verification.update({"method": "rerun rrd verify", "returncode": verify_run.returncode,
                                 "stdout": verify_run.stdout[-2000:]})

    # SR の低/高解像度比較 PNG（境界crop）
    sr_comparison_path = None
    try:
        middle_frame = len(frames) // 2
        temporal = load_frame_npz(dataset_dir, "temporal", middle_frame)
        sr = load_frame_npz(dataset_dir, "sr", middle_frame)
        if temporal is not None and sr is not None:
            depth = np.asarray(temporal["temporal_depth"], dtype=np.float64)
            valid = np.asarray(temporal["temporal_valid"], dtype=bool) & np.isfinite(depth) & (depth > 0)
            gy, gx = np.gradient(np.where(valid, depth, 0.0))
            grad = np.hypot(gx, gy)
            grad[~valid] = np.nan
            if np.any(np.isfinite(grad)):
                row, col = np.unravel_index(np.nanargmax(grad), grad.shape)
                half = 48
                r0, r1 = max(0, row - half), min(height, row + half)
                c0, c1 = max(0, col - half), min(width, col + half)
                rgb_crop = read_rgb(Path(frames[middle_frame]["rgb_path"]))[r0:r1, c0:c1]
                flow = np.asarray(sr["sr_flow"], dtype=np.float64)
                sr_valid = np.asarray(sr["sr_valid"], dtype=bool)
                flow_low = np.asarray(sr["sr_flow_low"], dtype=np.float64)
                low_valid = np.asarray(sr["sr_low_valid"], dtype=bool)
                bilinear_flow, bilinear_valid = upsample_bilinear_masked(
                    flow_low, (height, width), valid=low_valid
                )
                sr_mag = np.where(sr_valid, np.linalg.norm(flow, axis=-1), np.nan)[r0:r1, c0:c1]
                bil_mag = np.where(
                    bilinear_valid, np.linalg.norm(bilinear_flow, axis=-1), np.nan
                )[r0:r1, c0:c1]
                vmin = flow_range[0]
                vmax = max(flow_range[1], vmin + 1e-9)
                panels = [
                    _annotate_panel(rgb_crop, "RGB (boundary crop)"),
                    _annotate_panel(
                        colorize_scalar(bil_mag, vmin, vmax, valid=np.isfinite(bil_mag)),
                        "bilinear flow |S|",
                    ),
                    _annotate_panel(
                        colorize_scalar(sr_mag, vmin, vmax, valid=np.isfinite(sr_mag)),
                        "SR flow |S|",
                    ),
                ]
                sheet = np.concatenate(
                    [np.concatenate([panel, np.full((4, panel.shape[1], 3), 12, dtype=np.uint8)], axis=0)
                     for panel in panels], axis=1,
                )
                try:
                    from PIL import Image  # type: ignore

                    sr_comparison_path = visuals_dir / f"{args.dataset}_sr_comparison.png"
                    Image.fromarray(sheet).save(sr_comparison_path, format="PNG")
                except ImportError:
                    sr_comparison_path = None
    except (OSError, ValueError, KeyError) as error:
        print(f"render: SR比較PNGを生成できません: {error}", file=sys.stderr)
        sr_comparison_path = None
    report = {
        "dataset": args.dataset,
        "mp4": {"path": str(mp4_path), "fps": fps, "codec": writer.codec,
                "frames": writer.frames, "width": writer.width, "height": writer.height,
                "ffprobe": probe},
        "png": {"path": str(png_path) if png_path else None,
                "representative_frames": sorted(wanted)},
        "rrd": {"path": str(rrd_path), "frames_logged": rrd_frames, "verification": rrd_verification},
        "sr_comparison_png": str(sr_comparison_path) if sr_comparison_path else None,
        "rrd_entities": {
            "world/final_points": "final RGB点群 (K native整合, stride=%d)" % stride,
            "world/sr_arrows": "SR 3D変位サンプル矢印 (回転のみ適用, 倍率変更なし)",
            "world/camera": "camera pose (rigid c2w + scaled translation; metric世界)",
            "world/camera/image": "元RGB (1/3縮小)",
            "world/camera/depth": "final深度 (1/3, 共通range色)",
            "coordinate_convention": "RDF (x right, y down, z forward; rerun ViewCoordinates.RDF)",
        },
        "depth_range": final_range,
        "flow_range": flow_range,
        "panel_order": panel_order,
        "cell_width": cell_width,
        "common_range_note": "全フレーム・全depthパネル共通range。SRのみ別の共通flow range",
        "units": UNITS,
        "metric_calibrated": False,
        "argv": list(sys.argv),
        "created_at": utc_now(),
    }
    atomic_write_json(visuals_dir / "render_report.json", report)
    atomic_write_json(
        dataset_dir / "configs" / "stage_configs" / "render.json",
        {
            "dataset": args.dataset,
            "mp4": report["mp4"]["path"],
            "rrd": report["rrd"]["path"],
            "png": report["png"]["path"],
            "depth_range": final_range,
            "units": UNITS,
            "metric_calibrated": False,
            "argv": list(sys.argv),
            "created_at": utc_now(),
        },
    )
    print(f"render: mp4={mp4_path.name} ({writer.frames} frames) rrd={rrd_path.name}")
    return 0


def worker_moge(args: argparse.Namespace) -> int:
    """OPTIONAL比較専用: MoGe v3 を元RGBへ推論する（主経路では実行しない）。"""
    import torch
    from moge.model.v3 import MoGeModel  # type: ignore[import-not-found]

    dataset_dir = Path(args.dataset_dir)
    frames_payload = read_frame_list(Path(args.frames))
    frames = frames_payload["frames"]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("moge: device=cuda を指定されたが CUDA が利用できません（明示エラー）")
    model = MoGeModel.from_pretrained(args.model).to(device)
    processed = 0
    for record in frames:
        frame_index = int(record["frame_index"])
        rgb = read_rgb(Path(record["rgb_path"]))
        height, width = rgb.shape[:2]
        image = torch.from_numpy(np.ascontiguousarray(rgb)).to(device).float().permute(2, 0, 1) / 255.0
        output = model.infer(
            image,
            num_tokens=int(args.num_tokens),
            resolution_level=int(args.resolution_level),
            refine_steps=int(args.refine_steps),
        )
        depth = np.asarray(output["depth"].cpu().numpy(), dtype=np.float32)
        normal = np.asarray(output["normal"].cpu().numpy(), dtype=np.float32)
        mask = np.asarray(output["mask"].cpu().numpy(), dtype=bool)
        K_norm = np.asarray(output["intrinsics"].cpu().numpy(), dtype=np.float64)
        K_pixel = moge_norm_to_pixel_k(K_norm, width, height)
        valid = np.isfinite(depth) & (depth > 0)
        arrays = {
            "moge_depth": depth,
            "moge_valid": valid,
            "moge_mask": mask,
            "moge_normal": normal,
            "moge_K_norm": K_norm.astype(np.float32),
            "moge_K": K_pixel.astype(np.float32),
            "moge_support_count": np.int64(int(valid.sum())),
        }
        save_stage_frame(
            dataset_dir, "moge", frame_index, arrays,
            stage_metadata(
                "moge", args.dataset, args.model,
                {"num_tokens": int(args.num_tokens), "resolution_level": int(args.resolution_level),
                 "refine_steps": int(args.refine_steps),
                 "K": "normalized->pixel (uv [0,1] 規約, pixel=uv*size-0.5)",
                 "role": "OPTIONAL比較専用 (主経路外)"},
                {"frame": frame_index}, sys.argv,
            ),
        )
        processed += 1
    atomic_write_json(
        dataset_dir / "configs" / "stage_configs" / "moge.json",
        {
            "dataset": args.dataset,
            "model": args.model,
            "role": "optional_comparison_only",
            "processed": processed,
            "units": UNITS,
            "metric_calibrated": False,
            "argv": list(sys.argv),
            "created_at": utc_now(),
        },
    )
    print(f"moge(optional): {processed} frames")
    return 0


# --- CLI --------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="深度詳細化パイプラインの段階worker（pipeline scope）")
    sub = parser.add_subparsers(dest="worker", required=True)

    def add_common(name, help_text):
        worker = sub.add_parser(name, help=help_text)
        worker.add_argument("--dataset-dir", required=True)
        worker.add_argument("--dataset", required=True)
        worker.add_argument("--frames", required=True)
        return worker

    ingest = sub.add_parser("ingest", help="入力列挙 + 元RGBのPNG化 + frames.json")
    ingest.add_argument("--dataset-dir", required=True)
    ingest.add_argument("--dataset", required=True)
    ingest.add_argument("--source", required=True)
    ingest.add_argument("--frames-json", required=True)

    anchor = add_common("anchor", "ARROW depth -> fullres anchor (Sim3 scale)")
    anchor.add_argument("--clips-dir", required=True)
    anchor.add_argument("--alignments", required=True)

    geometry = add_common("geometry", "ARROW pose/K -> ref0変換 + pixel K")
    geometry.add_argument("--clips-dir", required=True)
    geometry.add_argument("--alignments", required=True)

    align = add_common("align", "clip間Sim(3) (ref0 adapter)")
    align.add_argument("--clips-dir", required=True)

    lingbot = add_common("lingbot", "LingBot-Depth 実モデル")
    lingbot.add_argument("--model", default="robbyant/lingbot-depth-pretrain-vitl-14-v0.5")
    lingbot.add_argument("--device", default="cuda")
    lingbot.add_argument("--resolution-level", type=int, default=9)

    lingbot_fix = add_common("lingbot_fix", "LingBot結果 -> ARROW anchorへ再固定")
    lingbot_fix.add_argument("--strength", type=float, default=2.0)
    lingbot_fix.add_argument("--min-points", type=int, default=32)

    add_common("prep", "gray edge + depth由来normal")

    ndr = add_common("ndr", "normal_depth_refiner (rust明示, fallback禁止)")
    ndr.add_argument("--backend", default="rust")
    ndr.add_argument("--iterations", type=int, default=40)
    ndr.add_argument("--allow-numpy", action="store_true",
                     help="backend=numpy を明示的に許可する（既定はrustのみ）")
    ndr.add_argument("--trust-abs", type=float, default=0.05)
    ndr.add_argument("--trust-rel", type=float, default=0.05)
    ndr.add_argument("--min-depth", type=float, default=1e-6)
    ndr.add_argument("--max-depth", type=float, default=1000.0)
    ndr.add_argument("--max-depth-source", default="explicit_cli")

    sceneflow_prep = add_common("sceneflow_prep", "低解像度ペア + camera補償")
    sceneflow_prep.add_argument("--clips-dir", required=True)
    sceneflow_prep.add_argument("--long-side", type=int, default=160)

    sceneflow_refine = add_common("sceneflow_refine", "flow + scene_flow_refinement")
    sceneflow_refine.add_argument("--min-depth", type=float, required=True)
    sceneflow_refine.add_argument("--max-depth", type=float, required=True)
    sceneflow_refine.add_argument("--max-motion", type=float, required=True)
    sceneflow_refine.add_argument("--trust-sigma", type=float, default=4.0)
    sceneflow_refine.add_argument("--workers", type=int, default=1,
                                  help="独立pairのCPU並列数（既定1。C++ ctypesはGIL解放・errorはthread_local）")

    sceneflow_post = add_common("sceneflow_post", "upsample + gates")
    sceneflow_post.add_argument("--trust-abs", type=float, default=0.05)
    sceneflow_post.add_argument("--trust-rel", type=float, default=0.05)

    add_common("sr", "scene flow super-resolution (core)")
    temporal = add_common("temporal", "時間融合 + 残差比較")
    temporal.add_argument("--weight", type=float, default=0.25)
    temporal.add_argument("--trust-abs", type=float, default=0.05)
    temporal.add_argument("--trust-rel", type=float, default=0.05)
    temporal.add_argument("--gate-tolerance", type=float, default=0.05)
    temporal.add_argument("--edge-threshold", type=float, default=0.5)

    final = add_common("final", "per-frame pack + final.npz + 共通range")
    final.add_argument("--range-low", type=float, default=2.0)
    final.add_argument("--range-high", type=float, default=98.0)

    render = add_common("render", "比較MP4/PNG/final点群RRD")
    render.add_argument("--fps", type=float, default=5.0)
    render.add_argument("--cell-width", type=int, default=256)
    render.add_argument("--point-stride", type=int, default=4)

    moge = add_common("moge", "MoGe v3 (optional比較専用)")
    moge.add_argument("--model", default="Ruicheng/moge-3-vitl")
    moge.add_argument("--device", default="cuda")
    moge.add_argument("--num-tokens", type=int, default=3600)
    moge.add_argument("--resolution-level", type=int, default=9)
    moge.add_argument("--refine-steps", type=int, default=3)
    return parser


WORKERS = {
    "ingest": worker_ingest,
    "anchor": worker_anchor,
    "geometry": worker_geometry,
    "align": worker_align,
    "lingbot": worker_lingbot,
    "lingbot_fix": worker_lingbot_fix,
    "prep": worker_prep,
    "ndr": worker_ndr,
    "sceneflow_prep": worker_sceneflow_prep,
    "sceneflow_refine": worker_sceneflow_refine,
    "sceneflow_post": worker_sceneflow_post,
    "sr": worker_sr,
    "temporal": worker_temporal,
    "final": worker_final,
    "render": worker_render,
    "moge": worker_moge,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return WORKERS[args.worker](args)


if __name__ == "__main__":
    sys.exit(main())
