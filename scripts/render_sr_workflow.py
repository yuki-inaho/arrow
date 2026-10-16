#!/usr/bin/env python3
"""SR workflow 可視化の独立 consumer（ARROW scene.npz 非依存）。

入力は results/depth_detail_oct16/<dataset>/ の確定 stage のみ:
  stages/final/NNNNNN.npz    final_depth / final_valid（NDR+temporal 集約後の実深度）
  stages/geometry/…          arrow_K（native intrinsics）/ arrow_camera_to_metric（metric pose）
  stages/sr/…                sr_flow（camera 補償済み 3D 残差変位, source camera 座標）/ sr_valid
  stages/anchor/…            arrow_depth / arrow_valid（sparse ARROW。比較パネル専用）
  stages/ndr/…               ndr_depth / ndr_valid（比較パネル専用）
  frames.json                frame_index・rgb_path・rgb_sha256
SR/モデルの再計算はしない。値の捏造・穴埋め・メートル表記はしない（units=arrow_provisional）。

出力（<output>/<dataset>/sr_visuals/ 配下）:
  <dataset>_sr_multipanel.mp4 / .png   1920x1080 dark 4 列（18/18/40/24、既存様式の再利用）
  <dataset>_sr_workflow.rrd / .rbl     Rerun 記録（blueprint 埋め込み）と native 用 .rbl
  <dataset>_sr_compare.mp4 / .png      旧 7 パネル比較（ARROW は表示用 footprint 縮小の新方式）
  <dataset>_sr_manifest.json           frame IDs・shape・units・source SHA・config・counts・paths

使い方::

    uv run --no-sync python scripts/render_sr_workflow.py \
        --input-dir results/depth_detail_oct16 --output-dir results/sr_static_oct16 \
        --dataset all [--max-frames N]

--max-frames は明示的な preview（manifest に preview と記録し、full とは表示しない）。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

from depth_detail_models import (  # noqa: E402
    VideoWriter,
    _annotate_panel,
    _composite_panels,
    _header_strip,
    colorize_scalar,
    load_frame_npz,
    magnitude_xyz,
    read_frame_list,
    read_rgb,
)
from render_dashboard import colors_for, utc_now  # noqa: E402
from render_multipanel import (  # noqa: E402
    ACCENT,
    BG,
    BORDER,
    FRUSTUM_SCALE,
    GAP,
    MAIN_H,
    MARGIN,
    PANEL_BG,
    TEXT_COLOR,
    TIMELINE_H,
    TITLE_COLOR,
    VIEW_3D,
    VIEW_ZOOM_3D,
    _axis_indicator,
    _setup_axes,
    column_geometry,
    display,
    draw_panel,
    font,
    frustum_segments,
    log,
    render_cam3d,
    render_tracks3d,
    scene_bounds,
    stacked_rows,
)

# --- 固定 config（manifest へ記録する。表示専用で入力値は変更しない） -----------------
STAGE_NAMES = ("final", "geometry", "sr", "anchor", "ndr")
DATASET_FPS = {"grass": 5.0, "railway": 24.0}
CLOUD_STRIDE = 4        # native pixel 間隔: 点群サンプリング（K は native index で整合）
ARROW_STRIDE = 32       # native pixel 間隔: SR 変位矢印のサンプリング
TRAJECTORY_PICKS = 48   # SR 移流（real advection）軌跡の本数
BOUNDS_SAMPLE_FRAMES = 24  # 固定視点用バウンズ算出に使う間引きフレーム数
POINT_RADIUS = 0.02     # RRD 点半径（arrow_provisional 単位。メートルではない）
ARROW_RADIUS = 0.03     # RRD 矢印半径（同上）
MIN_FREE_BYTES = 35 * 2**30
DATASET_BUDGET_BYTES = 6 * 2**30
AXIS_COLORS = [[255, 107, 107], [123, 216, 143], [107, 155, 255]]
CAMERA_DIM = (85, 115, 165, 255)
CAMERA_CURRENT = (90, 160, 255, 255)
CAMERA_PATH = (120, 160, 220, 255)
ARROW_COLOR = "#FF7833"


# ---------------------------------------------------------------------------
# 基本ユーティリティ


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_ancestor(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def preflight_disk(output_dir: Path) -> float:
    free = shutil.disk_usage(existing_ancestor(output_dir)).free
    if free < MIN_FREE_BYTES:
        raise SystemExit(
            f"空きディスク不足: {free / 2**30:.2f} GiB < 必要 {MIN_FREE_BYTES / 2**30:.0f} GiB"
            f"（出力先 {output_dir}）"
        )
    return float(free)


def check_budget(dataset_dir: Path, limit: int = DATASET_BUDGET_BYTES) -> int:
    total = sum(f.stat().st_size for f in dataset_dir.rglob("*") if f.is_file())
    if total > limit:
        raise SystemExit(
            f"出力サイズ予算超過: {total / 2**30:.2f} GiB > {limit / 2**30:.0f} GiB（{dataset_dir}）"
        )
    return int(total)


# ---------------------------------------------------------------------------
# 入力の読み込みとスキーマ検証（欠落・空は fail）


def stage_path(dataset_dir: Path, stage: str, index: int) -> Path:
    return dataset_dir / "stages" / stage / f"{index:06d}.npz"


def require_frame(dataset_dir: Path, stage: str, index: int, keys: tuple[str, ...]) -> dict:
    payload = load_frame_npz(dataset_dir, stage, index)
    if payload is None:
        raise SystemExit(
            f"stage 未生成: {stage} frame {index}: {stage_path(dataset_dir, stage, index)}"
        )
    missing = [key for key in keys if key not in payload]
    if missing:
        raise SystemExit(f"stage schema 不一致: {stage} frame {index} に {missing} がありません")
    return payload


def validate_frame_arrays(index: int, h: int, w: int, final: dict, geometry: dict, sr: dict) -> None:
    depth = np.asarray(final["final_depth"])
    valid = np.asarray(final["final_valid"], dtype=bool)
    if depth.shape != (h, w) or valid.shape != (h, w):
        raise SystemExit(f"final shape 不一致 frame {index}: {depth.shape} / {valid.shape} != {(h, w)}")
    finite = valid & np.isfinite(depth) & (depth > 0)
    if not finite.any():
        raise SystemExit(f"final が空（有効画素ゼロ）: frame {index}")
    K = np.asarray(geometry["arrow_K"], dtype=np.float64)
    metric = np.asarray(geometry["arrow_camera_to_metric"], dtype=np.float64)
    if K.shape != (3, 3) or metric.shape != (4, 4):
        raise SystemExit(f"geometry shape 不一致 frame {index}: K {K.shape}, metric {metric.shape}")
    if not (np.isfinite(K).all() and np.isfinite(metric).all()):
        raise SystemExit(f"geometry に非有限値: frame {index}")
    if not bool(geometry["arrow_K_valid"]) or not bool(geometry["arrow_pose_valid"]):
        raise SystemExit(f"geometry が invalid: frame {index}")
    flow = np.asarray(sr["sr_flow"])
    sr_valid = np.asarray(sr["sr_valid"], dtype=bool)
    if flow.shape != (h, w, 3) or sr_valid.shape != (h, w):
        raise SystemExit(f"sr shape 不一致 frame {index}: {flow.shape} / {sr_valid.shape}")
    if sr_valid.any() and not np.isfinite(flow[sr_valid]).all():
        raise SystemExit(f"sr_flow に valid 画素の非有限値: frame {index}")


def depth_range_from_final_npz(dataset_dir: Path) -> list[float] | None:
    path = dataset_dir / "stages" / "final.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        if "depth_range" not in data.files:
            return None
        values = [float(v) for v in data["depth_range"]]
    return values if len(values) == 2 and values[0] < values[1] else None


# ---------------------------------------------------------------------------
# 座標・幾何（固定契約: display=(x,z,-y) Z-up、native K、rotation-only の変位）


def camera_points(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """native pixel 中心 (u,v) と depth から camera 座標 [H,W,3] を作る。"""
    height, width = depth.shape
    u = np.arange(width, dtype=np.float64)[None, :]
    v = np.arange(height, dtype=np.float64)[:, None]
    x = (u - float(K[0, 2])) * depth / float(K[0, 0])
    y = (v - float(K[1, 2])) * depth / float(K[1, 1])
    return np.stack([x, y, depth], axis=-1)


def camera_point_at(u: float, v: float, depth: float, K: np.ndarray) -> np.ndarray:
    return np.array(
        [
            (u - float(K[0, 2])) * depth / float(K[0, 0]),
            (v - float(K[1, 2])) * depth / float(K[1, 1]),
            depth,
        ],
        dtype=np.float64,
    )


def to_metric(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """camera 座標 → metric（arrow_provisional world）: p' = R p + t。"""
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def rotation_only(vectors: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """ベクトルは回転のみ適用（並進は加えない。倍率変更もしない）。"""
    return vectors @ matrix[:3, :3].T


def cloud_for_frame(
    depth: np.ndarray, valid: np.ndarray, K: np.ndarray, metric: np.ndarray,
    rgb: np.ndarray, stride: int = CLOUD_STRIDE,
) -> dict:
    """final depth + native K + metric pose の RGB-D 点群（stride は native index で整合）。"""
    d = depth[::stride, ::stride]
    v = valid[::stride, ::stride] & np.isfinite(d) & (d > 0)
    rows, cols = np.nonzero(v)
    native_x, native_y = cols * stride, rows * stride
    z = d[rows, cols]
    x = (native_x - float(K[0, 2])) * z / float(K[0, 0])
    y = (native_y - float(K[1, 2])) * z / float(K[1, 1])
    world = to_metric(np.stack([x, y, z], axis=-1), metric)
    return {"points": world.astype(np.float32), "rgb": rgb[native_y, native_x]}


def sr_arrow_samples(
    flow: np.ndarray | None, sr_valid: np.ndarray, depth: np.ndarray, valid: np.ndarray,
    K: np.ndarray, metric: np.ndarray, stride: int = ARROW_STRIDE,
) -> tuple[np.ndarray, np.ndarray]:
    """SR 実変位のサンプル矢印（rotation-only で metric へ、倍率変更なし）。

    戻り値 (origins, vectors)。ペアが無い frame は空配列（呼び出し側で必ず log する）。
    """
    empty = (np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32))
    if flow is None:
        return empty
    rows = np.arange(0, sr_valid.shape[0], stride)
    cols = np.arange(0, sr_valid.shape[1], stride)
    grid = np.ix_(rows, cols)
    d = depth[grid]
    f = flow[grid]
    ok = sr_valid[grid] & valid[grid] & np.isfinite(f).all(axis=-1) & np.isfinite(d) & (d > 0)
    if not ok.any():
        return empty
    rows_i, cols_i = np.nonzero(ok)
    native_x, native_y = cols[cols_i], rows[rows_i]
    z = d[rows_i, cols_i]
    x = (native_x - float(K[0, 2])) * z / float(K[0, 0])
    y = (native_y - float(K[1, 2])) * z / float(K[1, 1])
    origins = to_metric(np.stack([x, y, z], axis=-1), metric).astype(np.float32)
    vectors = rotation_only(f[rows_i, cols_i], metric).astype(np.float32)
    return origins, vectors


def sr_motion_2d(
    flow: np.ndarray | None, sr_valid: np.ndarray, depth: np.ndarray, valid: np.ndarray, K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """camera 補償済み SR 残差による 2D 画素移動: project(K, P+S) - [u,v]。"""
    height, width = depth.shape
    if flow is None:
        nan = np.full((height, width), np.nan, dtype=np.float32)
        return nan, nan, np.zeros((height, width), dtype=bool)
    points = camera_points(depth, K)
    moved = points + flow
    ok = (
        sr_valid & valid & np.isfinite(moved).all(axis=-1) & (moved[..., 2] > 1e-9)
        & np.isfinite(depth) & (depth > 0)
    )
    u = np.arange(width, dtype=np.float64)[None, :]
    v = np.arange(height, dtype=np.float64)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        u2 = float(K[0, 0]) * moved[..., 0] / moved[..., 2] + float(K[0, 2])
        v2 = float(K[1, 1]) * moved[..., 1] / moved[..., 2] + float(K[1, 2])
    du = np.where(ok, u2 - u, np.nan).astype(np.float32)
    dv = np.where(ok, v2 - v, np.nan).astype(np.float32)
    return du, dv, ok


def normals_from_depth(
    depth: np.ndarray, valid: np.ndarray, K: np.ndarray, stride: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """final depth から表面法線（camera 向きへ符号統一、invalid は NaN）。"""
    d = depth[::stride, ::stride]
    v = valid[::stride, ::stride] & np.isfinite(d) & (d > 0)
    height, width = d.shape
    u = np.arange(width, dtype=np.float64)[None, :] * stride
    vv = np.arange(height, dtype=np.float64)[:, None] * stride
    z = np.where(v, d, np.nan)
    x = (u - float(K[0, 2])) * z / float(K[0, 0])
    y = (vv - float(K[1, 2])) * z / float(K[1, 1])
    points = np.stack([x, y, z], axis=-1)
    gx = np.full_like(points, np.nan)
    gy = np.full_like(points, np.nan)
    gx[:, 1:-1] = (points[:, 2:] - points[:, :-2]) * 0.5
    gy[1:-1, :] = (points[2:, :] - points[:-2, :]) * 0.5
    normals = np.cross(gx, gy)
    normals[normals[..., 2] > 0] *= -1.0  # camera を向く
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    ok = np.isfinite(normals).all(axis=-1) & (norm[..., 0] > 1e-12)
    with np.errstate(invalid="ignore", divide="ignore"):
        normals = normals / norm
    normals[~ok] = np.nan
    return normals.astype(np.float32), ok


def seed_pixels(sr_valid: np.ndarray, valid: np.ndarray, picks: int = TRAJECTORY_PICKS) -> np.ndarray:
    """移流軌跡の種（frame 先頭）を決定的に選ぶ: 有効画素へ均等間隔の格子。"""
    mask = sr_valid & valid
    count = int(mask.sum())
    if count == 0:
        return np.zeros((0, 2), dtype=np.int64)
    stride = max(1, int(round(math.sqrt(count / max(1, picks)))))
    grid = np.zeros_like(mask)
    grid[::stride, ::stride] = True
    grid &= mask
    rows, cols = np.nonzero(grid)
    if rows.size > picks:
        take = np.linspace(0, rows.size - 1, picks).round().astype(np.int64)
        rows, cols = rows[take], cols[take]
    return np.stack([rows, cols], axis=-1)


def finite_strips(line: np.ndarray) -> list[np.ndarray]:
    """[T,3] の系列を連続する有限区間に分割（死亡区間を跨いで結ばない）。"""
    keep = np.isfinite(line).all(axis=-1)
    strips: list[np.ndarray] = []
    start: int | None = None
    for index, alive in enumerate(keep):
        if alive and start is None:
            start = index
        if (not alive or index == len(keep) - 1) and start is not None:
            end = index if not alive else index + 1
            if end - start >= 2:
                strips.append(line[start:end].copy())
            start = None
    return strips


def init_tracks(frame: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """frame 先頭で SR 移流軌跡の種を作る: (rows, cols, world, alive)。"""
    seeds = seed_pixels(frame["sr_valid"], frame["valid"])
    rows = seeds[:, 0].astype(np.float64)
    cols = seeds[:, 1].astype(np.float64)
    world = np.full((len(seeds), 3), np.nan, dtype=np.float64)
    for slot, (row, col) in enumerate(seeds):
        depth_value = float(frame["depth"][row, col])
        camera = camera_point_at(float(col), float(row), depth_value, frame["K"])
        world[slot] = to_metric(camera, frame["metric"])
    alive = np.ones(len(seeds), dtype=bool)
    return rows, cols, world, alive


def advance_tracks(
    rows: np.ndarray, cols: np.ndarray, world: np.ndarray, alive: np.ndarray,
    previous: dict, frame: dict, height: int, width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """1 ペアの SR 実変位で 3D 位置を進め、現フレームの観測面へ再アンカーする。

    SR 変位は camera 補償済み残差なので world 側では rotation-only (R S) を加え、
    投影 → in-bounds/positive-depth ゲート → final depth 支持が無い点は死亡させる。
    """
    rows_safe = np.where(np.isfinite(rows), rows, 0.0)
    cols_safe = np.where(np.isfinite(cols), cols, 0.0)
    rows_i = np.clip(np.round(rows_safe).astype(np.int64), 0, height - 1)
    cols_i = np.clip(np.round(cols_safe).astype(np.int64), 0, width - 1)
    step_ok = np.zeros(len(rows), dtype=bool)
    step_vector = np.zeros((len(rows), 3), dtype=np.float64)
    if previous["flow"] is not None and len(rows):
        flags = previous["sr_valid"][rows_i, cols_i] & alive
        sample = previous["flow"][rows_i, cols_i]
        flags &= np.isfinite(sample).all(axis=-1)
        step_ok = flags
        step_vector[flags] = sample[flags]
    advance = world + step_vector @ previous["metric"][:3, :3].T
    new_world = np.full_like(world, np.nan)
    new_rows = np.full(len(rows), np.nan)
    new_cols = np.full(len(rows), np.nan)
    if len(rows) and alive.any():
        camera_next = (advance - frame["metric"][:3, 3]) @ frame["metric"][:3, :3]
        with np.errstate(invalid="ignore", divide="ignore"):
            u_proj = float(frame["K"][0, 0]) * camera_next[:, 0] / camera_next[:, 2] + float(
                frame["K"][0, 2]
            )
            v_proj = float(frame["K"][1, 1]) * camera_next[:, 1] / camera_next[:, 2] + float(
                frame["K"][1, 2]
            )
        gate = (
            alive & step_ok & (camera_next[:, 2] > 1e-9)
            & (u_proj >= 0) & (u_proj <= width - 1) & (v_proj >= 0) & (v_proj <= height - 1)
        )
        u_clean = np.where(np.isfinite(u_proj), u_proj, 0.0)
        v_clean = np.where(np.isfinite(v_proj), v_proj, 0.0)
        u_i = np.clip(np.round(u_clean).astype(np.int64), 0, width - 1)
        v_i = np.clip(np.round(v_clean).astype(np.int64), 0, height - 1)
        depth_next = frame["depth"][v_i, u_i]
        measured = gate & np.isfinite(depth_next) & (depth_next > 0) & frame["valid"][v_i, u_i]
        for slot in np.nonzero(measured)[0]:
            anchor = camera_point_at(float(u_i[slot]), float(v_i[slot]), float(depth_next[slot]),
                                     frame["K"])
            new_world[slot] = to_metric(anchor, frame["metric"])
            new_rows[slot] = float(v_i[slot])
            new_cols[slot] = float(u_i[slot])
    new_alive = np.isfinite(new_world).all(axis=-1)
    return new_rows, new_cols, new_world, new_alive


# ---------------------------------------------------------------------------
# 比較パネル用: sparse 深度の表示縮小（numerator/denominator、source は不変）


def box_weights(src_len: int, out_len: int) -> np.ndarray:
    """box filter の重なり面積 [out_len, src_len]（up/down 両対応、面積比は保持）。"""
    edges_src = np.arange(src_len + 1, dtype=np.float64)
    edges_out = np.arange(out_len + 1, dtype=np.float64) * (src_len / out_len)
    lo = np.maximum(edges_out[:-1, None], edges_src[None, :-1])
    hi = np.minimum(edges_out[1:, None], edges_src[None, 1:])
    return np.clip(hi - lo, 0.0, None)


def shrink_depth_footprint(
    depth: np.ndarray, valid: np.ndarray, out_shape: tuple[int, int],
    row_weights: np.ndarray | None = None, col_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """sparse depth の表示用縮小: 有効画素の面積加重平均 (num/den)。

    invalid 画素を黒として混ぜない（旧 bilinear 縮小の暗さ対策）。返り値は
    (depth_out, coverage)。有効画素ゼロのセルは NaN / coverage 0。入力配列は変更しない。
    """
    source = np.asarray(depth, dtype=np.float64)
    ok = np.asarray(valid, dtype=bool) & np.isfinite(source) & (source > 0)
    height, width = source.shape
    out_h, out_w = int(out_shape[0]), int(out_shape[1])
    if out_h < 1 or out_w < 1:
        raise ValueError("out_shape must be positive")
    if row_weights is None:
        row_weights = box_weights(height, out_h)
    if col_weights is None:
        col_weights = box_weights(width, out_w)
    weighted = np.where(ok, source, 0.0)
    num = row_weights @ weighted @ col_weights.T
    den = row_weights @ ok.astype(np.float64) @ col_weights.T
    with np.errstate(invalid="ignore", divide="ignore"):
        depth_out = num / den
    depth_out[den == 0] = np.nan
    coverage = den / np.outer(row_weights.sum(axis=1), col_weights.sum(axis=1))
    return depth_out, coverage


def anchor_grid_axes(valid: np.ndarray, min_fill: float = 0.5) -> tuple[np.ndarray, np.ndarray, bool]:
    """sparse 有効画素が格子サンプリングなら (rows, cols, True) を返す。

    判定: 有効画素の固有行×固有列で作る格子の被覆率が min_fill 以上、かつ
    元画素数に対して十分な縮約（半分以下）であること。散在画素やほぼ密な場合は False。
    """
    ok = np.asarray(valid, dtype=bool)
    rows = np.unique(np.nonzero(ok.any(axis=1))[0])
    cols = np.unique(np.nonzero(ok.any(axis=0))[0])
    if rows.size == 0 or cols.size == 0:
        return rows, cols, False
    fill = float(ok[np.ix_(rows, cols)].mean())
    reduction = rows.size * cols.size * 2 <= ok.size
    return rows, cols, bool(fill >= min_fill and reduction)


def sparse_arrow_display(
    depth: np.ndarray, valid: np.ndarray, out_shape: tuple[int, int], vmin: float, vmax: float,
) -> tuple[np.ndarray, float, bool]:
    """ARROW sparse の表示専用展開（source 配列は変更しない）。

    格子サンプルなら native anchor 格子へ binning し、表示サイズへ NEAREST で拡大する
    （黒い格子が支配的にならず、低解像 depth が読める。invalid は背景のまま）。
    格子でない場合は面積加重の display 縮小（num/den）へフォールバックする。
    戻り値 (colored_uint8, coverage_source, used_grid)。
    """
    source = np.asarray(depth, dtype=np.float64)
    ok = np.asarray(valid, dtype=bool) & np.isfinite(source) & (source > 0)
    coverage = float(ok.mean()) if ok.size else 0.0
    out_h, out_w = int(out_shape[0]), int(out_shape[1])
    rows, cols, used_grid = anchor_grid_axes(ok)
    if used_grid:
        grid_depth = source[np.ix_(rows, cols)]
        grid_ok = ok[np.ix_(rows, cols)]
        small = colorize_scalar(grid_depth, vmin, vmax, valid=grid_ok)
        image = Image.fromarray(small).resize((out_w, out_h), Image.Resampling.NEAREST)
        return np.asarray(image), coverage, True
    depth_out, _ = shrink_depth_footprint(source, ok, (out_h, out_w))
    colored = colorize_scalar(depth_out, vmin, vmax, valid=np.isfinite(depth_out))
    return colored, coverage, False


# ---------------------------------------------------------------------------
# 2D パネル（PIL）


def colorize_depth(depth: np.ndarray, valid: np.ndarray, vmin: float, vmax: float) -> Image.Image:
    ok = valid & np.isfinite(depth) & (depth > 0)
    return Image.fromarray(colorize_scalar(depth, vmin, vmax, valid=ok))


def render_motion_panel(
    rgb: np.ndarray, du: np.ndarray, dv: np.ndarray, ok: np.ndarray, stride: int,
    flow_low: float, flow_high: float,
) -> Image.Image:
    """入力 RGB を淡く + SR 残差 2D 移動の矢印（長さ 1.5px 以上のみ、色は magnitude）。"""
    background = Image.fromarray(rgb).convert("RGB").point(lambda value: int(value * 0.45 + 120 * 0.55))
    layer = Image.new("RGBA", background.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    from matplotlib import colormaps

    cmap = colormaps["turbo"]
    height, width = du.shape
    span = max(1e-6, flow_high - flow_low)
    for row in range(0, height, stride):
        for col in range(0, width, stride):
            if not ok[row, col]:
                continue
            x0, y0 = float(col), float(row)
            x1, y1 = x0 + float(du[row, col]), y0 + float(dv[row, col])
            length = float(math.hypot(x1 - x0, y1 - y0))
            if length < 1.5:
                continue
            magnitude = float(math.hypot(du[row, col], dv[row, col]))
            value = min(1.0, max(0.0, (magnitude - flow_low) / span))
            color = tuple(int(c * 255) for c in cmap(value)[:3])
            draw.line([x0, y0, x1, y1], fill=(*color, 235), width=2)
            ux, uy = (x1 - x0) / length, (y1 - y0) / length
            px, py = -uy, ux
            head = 4.0
            draw.polygon(
                [(x1, y1),
                 (x1 - ux * head + px * head * 0.55, y1 - uy * head + py * head * 0.55),
                 (x1 - ux * head - px * head * 0.55, y1 - uy * head - py * head * 0.55)],
                fill=(*color, 235),
            )
    return Image.alpha_composite(background.convert("RGBA"), layer).convert("RGB")


def render_support_vis(sr_valid: np.ndarray, final_valid: np.ndarray) -> Image.Image:
    """支持・有効性（confidence の代替）: 左=SR ペア支持、右=final 深度支持。"""
    left = np.where(sr_valid, 255, 0).astype(np.uint8)
    right = np.where(final_valid, 255, 0).astype(np.uint8)
    canvas = np.concatenate([left, np.full_like(left, 60), right], axis=1)
    return Image.fromarray(np.stack([canvas] * 3, axis=-1))


def render_support_mask(sr_valid: np.ndarray, final_valid: np.ndarray) -> Image.Image:
    """支持マスク: 白=SR支持かつ深度支持、灰=深度のみ、黒=その他。"""
    gray = np.full(sr_valid.shape, 24, dtype=np.uint8)
    gray[final_valid] = 110
    gray[sr_valid & final_valid] = 255
    return Image.fromarray(np.stack([gray] * 3, axis=-1))


# ---------------------------------------------------------------------------
# 3D / 集計パネル


def compute_scene_bounds(sampled_clouds: list[np.ndarray], camera_positions: np.ndarray) -> dict:
    """全フレーム共通の表示バウンズ。外れ値は表示範囲だけから除き、データは保持する。"""
    arrays = [display(points) for points in sampled_clouds if len(points)]
    points = np.concatenate(arrays)
    points = points[np.isfinite(points).all(axis=1)]
    low, high = np.percentile(points, [1, 99], axis=0)
    bounds = scene_bounds(np.stack([low, high]), display(camera_positions))
    for axis in ("x", "y", "z"):
        low, high = bounds[axis]
        center = (low + high) / 2
        half = (high - low) / 2 / VIEW_ZOOM_3D
        bounds[axis] = (center - half, center + half)
    return bounds


def render_sr_scene3d(
    frame_t: int, cloud: dict, arrows: tuple, matrices: np.ndarray, valid: np.ndarray,
    bounds: dict, frustum_scale: float,
) -> Image.Image:
    """RGB-D 点群 + SR 実変位矢印（rotation-only）+ カメラ視錐台/パス。視点は固定。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = plt.figure(figsize=(7.3, 6.8), dpi=100, facecolor="#0F1116")
    ax = fig.add_subplot(projection="3d", facecolor="#0F1116")
    _setup_axes(ax, bounds)

    points = display(cloud["points"])
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],  # pyright: ignore[reportArgumentType]
               s=2, c=cloud["rgb"].astype(np.float32) / 255.0, linewidths=0, alpha=0.85)

    origins, vectors = arrows
    if len(origins):
        arrow_origin = display(origins)
        arrow_vector = display(vectors)
        ax.quiver(  # pyright: ignore[reportArgumentType]
            arrow_origin[:, 0], arrow_origin[:, 1], arrow_origin[:, 2],
            arrow_vector[:, 0], arrow_vector[:, 1], arrow_vector[:, 2],
            color=ARROW_COLOR, arrow_length_ratio=0.3, linewidth=0.9,
        )

    positions = display(matrices[:, :3, 3])
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color="#5B7BB4", lw=1.2, alpha=0.8)
    for frame in range(len(matrices)):
        if not valid[frame]:
            continue
        current = frame == frame_t
        for segment in frustum_segments(matrices[frame], frustum_scale):
            ax.plot(segment[:, 0], segment[:, 1], segment[:, 2],
                    color="#5AA0FF" if current else "#4A6A9C",
                    lw=2.0 if current else 0.9, alpha=1.0 if current else 0.3)

    _axis_indicator(ax, bounds)
    ax.view_init(elev=VIEW_3D["elev"], azim=VIEW_3D["azim"])
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    array = np.asarray(canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return Image.fromarray(array.copy())


def summary_rows(frame_t: int, stats: dict) -> list[tuple[str, str]]:
    """集計パネルの行。「現在」の値は stats["current"]（そのフレームの確定値）だけを使う。"""
    frames = int(stats["frames_rendered"])
    scope = f"{frame_t + 1}/{frames}" + (" (preview)" if stats["preview"] else "")
    current = stats["current"]
    return [
        ("フレーム", scope),
        ("SR変位支持率（現在）", f"{100 * float(current['sr_rate']):.2f} %"),
        ("深度支持率（現在）", f"{100 * float(current['final_rate']):.2f} %"),
        ("SR支持画素（現在）", f"{int(current['sr_count']):,}"),
        ("支持率の分母", f"全画素 {int(stats['pixels']):,} (H*W)"),
        ("有効カメラ姿勢", f"{int(stats['camera_valid'].sum())} / {frames}"),
        ("SR移流軌跡（生存）", f"{int(current['alive'])} / {int(stats['track_total'])}"),
        ("単位", "arrow_provisional (non-metric)"),
    ]


def render_sr_summary(frame_t: int, stats: dict) -> Image.Image:
    """SR 統計（実測値のみ。score や calibrated confidence は出さない）。"""
    width, height = 420, 195
    image = Image.new("RGB", (width, height), PANEL_BG)
    draw = ImageDraw.Draw(image)
    y = 4
    for label, value in summary_rows(frame_t, stats):
        draw.text((10, y), label, font=font(14), fill=TITLE_COLOR)
        text = draw.textlength(value, font=font(14))
        draw.text((width - 12 - text, y), value, font=font(14), fill=TEXT_COLOR)
        y += 20

    graph = (10, y + 2, width - 12, height - 6)
    draw.rectangle(graph, outline=BORDER, width=1)
    rate = np.asarray(stats["sr_rate"], dtype=np.float64)
    xs = np.linspace(graph[0] + 2, graph[2] - 2, len(rate))
    ys = graph[3] - 4 - np.clip(rate, 0.0, 1.0) * (graph[3] - graph[1] - 8)
    # ストリーミング描画のため、曲線は現在フレームまでの実測値のみ（未来値は描かない）
    draw.line(list(zip(xs[: frame_t + 1].tolist(), ys[: frame_t + 1].tolist())), fill=ACCENT, width=2)
    marker_x, marker_y = xs[frame_t], ys[frame_t]
    draw.ellipse([marker_x - 3, marker_y - 3, marker_x + 3, marker_y + 3], fill="#FFFFFF")
    return image


# ---------------------------------------------------------------------------
# 1920x1080 合成（既存 4 列 18/18/40/24 レイアウトの再利用）


def draw_timeline(
    board: Image.Image, draw: ImageDraw.ImageDraw, position: int, total: int,
    thumbs: list[Image.Image], note: str,
) -> None:
    timeline_y = MARGIN + MAIN_H + GAP
    draw.rounded_rectangle((MARGIN, timeline_y, 1920 - MARGIN, timeline_y + TIMELINE_H),
                           radius=8, fill=PANEL_BG, outline=BORDER, width=1)
    draw.polygon([(MARGIN + 22, timeline_y + 18), (MARGIN + 22, timeline_y + 44),
                  (MARGIN + 44, timeline_y + 31)], fill=TEXT_COLOR)
    draw.rectangle([MARGIN + 52, timeline_y + 20, MARGIN + 56, timeline_y + 42], fill=(110, 118, 130))
    draw.rectangle([MARGIN + 60, timeline_y + 20, MARGIN + 64, timeline_y + 42], fill=(110, 118, 130))
    draw.text((MARGIN + 78, timeline_y + 16), f"Frame {position + 1}/{total}",
              font=font(20), fill=TEXT_COLOR)
    # footer は下余白帯（timeline の外）へ置き、クリップとバーとの重なりを避ける
    note_width = draw.textlength(note, font=font(13))
    draw.text((1920 - MARGIN - 12 - note_width, timeline_y + TIMELINE_H + 3), note, font=font(13),
              fill=TITLE_COLOR)

    bar_x0, bar_x1 = 300, 1500
    bar_y = timeline_y + 24
    draw.line([bar_x0, bar_y, bar_x1, bar_y], fill=(66, 72, 84), width=5)
    span = (bar_x1 - bar_x0) * position / max(1, total - 1)
    draw.line([bar_x0, bar_y, bar_x0 + span, bar_y], fill=ACCENT, width=5)
    draw.ellipse([bar_x0 + span - 7, bar_y - 7, bar_x0 + span + 7, bar_y + 7], fill=ACCENT)

    if thumbs:
        thumb_h = 46
        thumb_w = max(6, min(74, (bar_x1 - bar_x0 - 4 * len(thumbs)) // max(1, len(thumbs))))
        x = bar_x0
        for index, image in enumerate(thumbs):
            fitted = image.resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)
            y = timeline_y + 46
            board.paste(fitted, (x, y))
            border = ACCENT if index == position else BORDER
            draw.rectangle([x - 1, y - 1, x + thumb_w, y + thumb_h], outline=border,
                           width=2 if index == position else 1)
            x += thumb_w + 4


def compose_sr_board(
    panels: dict, position: int, total: int, thumbs: list[Image.Image], note: str, depth_note: str,
) -> Image.Image:
    """4 列 18/18/40/24 の 1 フレーム合成（全パネル同期）。"""
    board = Image.new("RGB", (1920, 1080), BG)
    draw = ImageDraw.Draw(board)
    columns = column_geometry()
    top = MARGIN
    rows = stacked_rows(top)
    layout = [
        (columns[0], [("入力RGB (SR workflow)", "input", False, None),
                      ("2D移動 (SR残差)", "motion", False, None),
                      ("支持・有効性 (support, not confidence)", "supportvis", False, None)]),
        (columns[1], [("最終深度 (final)", "depth", True, depth_note),
                      ("表面法線 (final depth)", "normal", False, None),
                      ("支持マスク (分母=全画素)", "mask", False, None)]),
        (columns[2], [("3D: 点群+SR変位+カメラ (SR/generated)", "scene3d", False, None)]),
        (columns[3], [("カメラ姿勢 (provisional, SR/generated)", "cam3d", False, None),
                      ("SR移流軌跡 (real advection)", "tracks3d", False, None),
                      ("SR統計 (実測値のみ)", "summary", False, None)]),
    ]
    for index, ((x, width), entries) in enumerate(layout):
        if index == 2:
            box = (x, top, x + width, top + MAIN_H)
            title, name, has_legend, extra = entries[0]
            draw_panel(board, draw, box, title, panels.get(name), legend=has_legend, note=extra)
            continue
        for (y, height), (title, name, has_legend, extra) in zip(rows, entries):
            box = (x, y, x + width, y + height)
            frame_note = extra if extra is not None else (
                f"{position + 1}/{total}" if name == "input" else None
            )
            rect = draw_panel(board, draw, box, title, panels.get(name), legend=has_legend,
                              note=frame_note)
            if name == "supportvis" and rect is not None:
                paste_x, paste_y, fitted_w, fitted_h = rect
                third = fitted_w // 3
                for label, center_x in (("SR支持", paste_x + third // 2),
                                        ("深度支持", paste_x + third + third // 2)):
                    text_width = draw.textlength(label, font=font(13))
                    draw.text((center_x - text_width / 2, paste_y + fitted_h + 3), label,
                              font=font(13), fill=TITLE_COLOR)

    draw_timeline(board, draw, position, total, thumbs, note)
    return board


# ---------------------------------------------------------------------------
# Rerun 記録


def jpeg_bytes(image: Image.Image, quality: int = 88) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def build_blueprint(rrb, labels: dict, eye: np.ndarray, target: np.ndarray):
    """動画と同じ構成（主 3D + 6 面 2D、暗色、共通観察視点）の blueprint。"""
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/world",
                name=labels["world"],
                background=[15, 17, 22, 255],
                eye_controls=rrb.EyeControls3D(
                    kind="orbital",
                    position=[float(v) for v in eye],
                    look_target=[float(v) for v in target],
                    eye_up=[0.0, 0.0, 1.0],
                ),
            ),
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial2DView(origin="view/input", name=labels["input"],
                                      background=[15, 17, 22, 255]),
                    rrb.Spatial2DView(origin="view/motion", name=labels["motion"],
                                      background=[15, 17, 22, 255]),
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(origin="view/depth", name=labels["depth"],
                                      background=[15, 17, 22, 255]),
                    rrb.Spatial2DView(origin="view/normal", name=labels["normal"],
                                      background=[15, 17, 22, 255]),
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(origin="view/mask", name=labels["mask"],
                                      background=[15, 17, 22, 255]),
                    rrb.Spatial2DView(origin="view/support", name=labels["support"],
                                      background=[15, 17, 22, 255]),
                ),
            ),
            column_shares=[2, 1],
        ),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )


def log_static(
    rr, matrices: np.ndarray, valid: np.ndarray, strips: list, strip_colors: list,
    frustum_scale: float, bounds: dict,
) -> None:
    positions = display(matrices[:, :3, 3])
    rr.log("world/camera_path", rr.LineStrips3D([positions], colors=[CAMERA_PATH], radii=0.02),
           static=True)
    frusta = []
    for frame in range(len(matrices)):
        if valid[frame]:
            frusta.extend(frustum_segments(matrices[frame], frustum_scale))
    if frusta:
        rr.log("world/cameras",
               rr.LineStrips3D(frusta, colors=[list(CAMERA_DIM)] * len(frusta), radii=0.02),
               static=True)
    if strips:
        rr.log("world/sr_tracks",
               rr.LineStrips3D(strips, colors=np.asarray(strip_colors, dtype=np.uint8), radii=0.02),
               static=True)
    span = float(np.max([bounds["x"][1] - bounds["x"][0], bounds["y"][1] - bounds["y"][0],
                         bounds["z"][1] - bounds["z"][0]]))
    length = 0.05 * max(span, 1e-6)
    origin = np.zeros(3, dtype=np.float32)
    rr.log(
        "world/axis",
        rr.LineStrips3D(
            [np.stack([origin, [length, 0, 0]]), np.stack([origin, [0, length, 0]]),
             np.stack([origin, [0, 0, length]])],
            colors=AXIS_COLORS, radii=0.02,
        ),
        static=True,
    )


# ---------------------------------------------------------------------------
# 比較動画（旧 7 パネル + ARROW sparse の表示 footprint 方式）


def build_sr_compare(
    dataset_dir: Path, frames: list, indices: list[int], fps: float, visuals_dir: Path,
    depth_range: list[float], flow_range: list[float], cell_width: int = 256,
) -> dict:
    """旧比較（RGB|ARROW|LingBot|NDR|Scene flow|Temporal|SR 3D flow）を維持しつつ、
    ARROW のみ有効画素ベースの表示縮小にする。深度値・入力ファイルは一切変更しない。"""
    mp4_path = visuals_dir / f"{dataset_dir.name}_sr_compare.mp4"
    writer = None
    representatives = []
    wanted = [indices[0], indices[len(indices) // 2], indices[-1]]
    flow_low, flow_high = float(flow_range[0]), max(float(flow_range[1]), float(flow_range[0]) + 1e-9)
    arrow_methods = {"grid": 0, "area": 0}

    for frame_index in indices:
        record = frames[frame_index]
        rgb = read_rgb(Path(record["rgb_path"]))
        scale = cell_width / float(rgb.shape[1])
        panel_h = max(2, int(round(rgb.shape[0] * scale)))
        panel_h -= panel_h % 2

        def shrink(image: np.ndarray) -> np.ndarray:
            array = np.asarray(image, dtype=np.uint8)
            if array.shape[0] == panel_h and array.shape[1] == cell_width:
                return array
            return np.asarray(
                Image.fromarray(array).resize((cell_width, panel_h), Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )

        def depth_panel(payload: dict | None, key: str, valid_key: str) -> np.ndarray:
            if payload is None or key not in payload or valid_key not in payload:
                return np.full((panel_h, cell_width, 3), 25, dtype=np.uint8)
            depth = np.asarray(payload[key], dtype=np.float64)
            valid = np.asarray(payload[valid_key], dtype=bool) & np.isfinite(depth) & (depth > 0)
            return shrink(colorize_scalar(depth, depth_range[0], depth_range[1], valid=valid))

        anchor = load_frame_npz(dataset_dir, "anchor", frame_index)
        if anchor is None or "arrow_depth" not in anchor or "arrow_valid" not in anchor:
            raise SystemExit(f"anchor 未生成/スキーマ不一致: frame {frame_index}")
        arrow_depth = np.asarray(anchor["arrow_depth"], dtype=np.float64)
        arrow_valid = np.asarray(anchor["arrow_valid"], dtype=bool)
        arrow_colored, arrow_cov, arrow_used_grid = sparse_arrow_display(
            arrow_depth, arrow_valid, (panel_h, cell_width), depth_range[0], depth_range[1]
        )
        arrow_methods["grid" if arrow_used_grid else "area"] += 1
        arrow_text = f"ARROW sparse (display footprint) cov={100 * arrow_cov:.2f}%"
        arrow_panel = _annotate_panel(arrow_colored, arrow_text)

        lingbot_fix = load_frame_npz(dataset_dir, "lingbot_fix", frame_index) or {}
        lingbot_raw = load_frame_npz(dataset_dir, "lingbot", frame_index) or {}
        lingbot = {**lingbot_raw, **lingbot_fix}
        ndr = load_frame_npz(dataset_dir, "ndr", frame_index)
        sceneflow = load_frame_npz(dataset_dir, "sceneflow_post", frame_index)
        temporal = load_frame_npz(dataset_dir, "temporal", frame_index)
        sr = load_frame_npz(dataset_dir, "sr", frame_index)
        flow_panel = np.full((panel_h, cell_width, 3), 25, dtype=np.uint8)
        if sr is not None and "sr_flow" in sr and "sr_valid" in sr:
            flow = np.asarray(sr["sr_flow"], dtype=np.float64)
            flag = np.asarray(sr["sr_valid"], dtype=bool) & np.isfinite(flow).all(axis=-1)
            magnitude = np.where(flag, magnitude_xyz(flow), np.nan)
            flow_panel = shrink(colorize_scalar(magnitude, flow_low, flow_high, valid=flag))

        panels = [
            _annotate_panel(shrink(rgb), "RGB"),
            arrow_panel,
            _annotate_panel(depth_panel(lingbot, "lingbot_depth", "lingbot_valid"), "LingBot"),
            _annotate_panel(depth_panel(ndr, "ndr_depth", "ndr_valid"), "NDR"),
            _annotate_panel(depth_panel(sceneflow, "sceneflow_depth", "sceneflow_valid"), "Scene flow"),
            _annotate_panel(depth_panel(temporal, "temporal_depth", "temporal_valid"), "Temporal"),
            _annotate_panel(flow_panel, "SR 3D flow"),
        ]
        composite = _composite_panels(panels, cell_width)
        header = _header_strip(
            composite.shape[1],
            f"frame {frame_index}/{len(frames) - 1}  depth range [{depth_range[0]:.2f}, "
            f"{depth_range[1]:.2f}]  ARROW=display footprint only (valid-weighted; values "
            f"unchanged)  flow range [{flow_low:.4f}, {flow_high:.4f}]  "
            f"(generated, arrow_provisional)",
        )
        frame_image = np.concatenate([header, composite], axis=0)
        if writer is None:
            writer = VideoWriter(mp4_path, frame_image.shape[1], frame_image.shape[0], fps)
        writer.append(frame_image)
        if frame_index in wanted:
            representatives.append(frame_image)
        if frame_index % 20 == 0:
            log(f"  compare: {frame_index + 1}/{len(frames)} frames")
    if writer is None:
        raise SystemExit("compare: フレームがありません")
    writer.close()

    sheet = np.concatenate(
        [np.concatenate([panel, np.full((6, panel.shape[1], 3), 12, dtype=np.uint8)], axis=0)
         for panel in representatives],
        axis=0,
    )
    png_path = visuals_dir / f"{dataset_dir.name}_sr_compare.png"
    Image.fromarray(sheet).save(png_path, format="PNG")
    return {"mp4": mp4_path, "png": png_path, "arrow_display": arrow_methods}


# ---------------------------------------------------------------------------
# dataset 1 本の描画


def load_render_frame(dataset_dir: Path, record: dict, index: int, h: int, w: int) -> dict:
    final = require_frame(dataset_dir, "final", index, ("final_depth", "final_valid"))
    geometry = require_frame(dataset_dir, "geometry", index, ("arrow_K", "arrow_camera_to_metric"))
    sr = require_frame(dataset_dir, "sr", index,
                       ("sr_flow", "sr_valid", "sr_pair_index", "sr_support_count"))
    for stage in ("anchor", "ndr"):
        if not stage_path(dataset_dir, stage, index).is_file():
            raise SystemExit(f"stage 未生成: {stage} frame {index}")
    validate_frame_arrays(index, h, w, final, geometry, sr)
    rgb_path = Path(record["rgb_path"])
    if not rgb_path.is_file():
        raise SystemExit(f"RGB 未生成: {rgb_path}")
    rgb = read_rgb(rgb_path)
    if rgb.shape != (h, w, 3):
        raise SystemExit(f"RGB shape 不一致: {rgb.shape} != {(h, w, 3)} ({rgb_path})")
    pair = np.asarray(sr["sr_pair_index"], dtype=np.int64)
    has_pair = pair[0] >= 0 and int(np.asarray(sr["sr_support_count"]).reshape(-1)[0]) > 0
    return {
        "index": index,
        "depth": np.asarray(final["final_depth"], dtype=np.float64),
        "valid": np.asarray(final["final_valid"], dtype=bool),
        "K": np.asarray(geometry["arrow_K"], dtype=np.float64),
        "metric": np.asarray(geometry["arrow_camera_to_metric"], dtype=np.float64),
        "flow": np.asarray(sr["sr_flow"], dtype=np.float64) if has_pair else None,
        "sr_valid": np.asarray(sr["sr_valid"], dtype=bool),
        "pair": pair,
        "has_pair": bool(has_pair),
        "rgb": rgb,
    }


def render_dataset(args: argparse.Namespace, dataset: str) -> dict:
    started = time.perf_counter()
    input_dir = args.input_dir / dataset
    frames_json = input_dir / "frames.json"
    if not frames_json.is_file():
        raise SystemExit(f"frames.json が見つかりません: {frames_json}")
    payload = read_frame_list(frames_json)
    frames = payload["frames"]
    height, width = (int(v) for v in payload["original_size"])
    total = len(frames)
    rendered_count = total if args.max_frames is None else min(int(args.max_frames), total)
    if rendered_count < 1:
        raise SystemExit("--max-frames は 1 以上が必要")
    preview = rendered_count < total
    indices = list(range(rendered_count))
    for index in indices:
        for stage in ("final", "geometry", "sr"):
            if not stage_path(input_dir, stage, index).is_file():
                raise SystemExit(f"stage 未生成: {stage} frame {index}")
        rgb_path = Path(frames[index]["rgb_path"])
        if not rgb_path.is_file():
            raise SystemExit(f"RGB 未生成: frame {index}: {rgb_path}")

    fps = float(DATASET_FPS.get(dataset, 30.0))
    visuals_dir = args.output_dir / dataset / "sr_visuals"
    work_dir = visuals_dir / "_frames"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    visuals_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True)

    depth_range = depth_range_from_final_npz(input_dir)
    scope = "preview" if preview else "full"
    log(f"[{dataset}] frames {rendered_count}/{total} ({scope}), fps={fps}, layout 1920x1080")

    # 固定視点のため、間引いたフレームの点群と全カメラ位置からバウンズを先に決める。
    sample_step = max(1, math.ceil(rendered_count / BOUNDS_SAMPLE_FRAMES))
    sampled_clouds = []
    matrices_all = []
    valid_all = []
    for index in indices:
        geometry = require_frame(input_dir, "geometry", index,
                                 ("arrow_K", "arrow_camera_to_metric", "arrow_pose_valid"))
        matrices_all.append(np.asarray(geometry["arrow_camera_to_metric"], dtype=np.float64))
        valid_all.append(bool(geometry["arrow_pose_valid"]))
        if index % sample_step == 0:
            final = require_frame(input_dir, "final", index, ("final_depth", "final_valid"))
            cloud = cloud_for_frame(
                np.asarray(final["final_depth"], dtype=np.float64),
                np.asarray(final["final_valid"], dtype=bool),
                np.asarray(geometry["arrow_K"], dtype=np.float64),
                np.asarray(geometry["arrow_camera_to_metric"], dtype=np.float64),
                np.zeros((height, width, 3), dtype=np.uint8),
            )
            sampled_clouds.append(cloud["points"][::3])
    matrices = np.stack(matrices_all).astype(np.float32)
    valid = np.array(valid_all, dtype=bool)
    bounds = compute_scene_bounds(sampled_clouds, matrices[:, :3, 3])
    span = max(bounds["x"][1] - bounds["x"][0], bounds["y"][1] - bounds["y"][0],
               bounds["z"][1] - bounds["z"][0])
    frustum_scale = max(FRUSTUM_SCALE, span * 0.05)
    center = np.array([(bounds[axis][0] + bounds[axis][1]) / 2 for axis in ("x", "y", "z")],
                      dtype=np.float64)
    eye = center + np.array([-0.3, -2.0, 1.0], dtype=np.float64) * max(span, 1e-6)

    # SR 移流（real advection）の状態
    track_rows = np.zeros(0, dtype=np.float64)
    track_cols = np.zeros(0, dtype=np.float64)
    track_world = np.zeros((0, 3), dtype=np.float64)
    track_alive = np.zeros(0, dtype=bool)
    track_positions = np.full((rendered_count, 0, 3), np.nan, dtype=np.float32)
    track_colors: list = []

    # Rerun 初期化（Application ID は dataset ごとに固定）
    import rerun as rr
    import rerun.blueprint as rrb

    app_id = f"arrow_sr_static_{dataset}"
    rr.init(app_id, spawn=False)
    # world ヘッダ宣言: display=(x, z, -y) の Z-up 右手系。無いと core 監査で header=null。
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    labels = {
        # Rerun 0.39 headless uses Inter only and panics on missing CJK glyphs.
        "world": "RGB-D cloud + SR motion + cameras (arrow_provisional)",
        "input": "Input RGB (SR workflow)",
        "motion": "2D motion from SR (camera-compensated)",
        "depth": "Final depth (provisional)",
        "normal": "Surface normals (final depth)",
        "mask": f"Support mask (all {height * width} pixels)",
        "support": "SR support / validity (not confidence)",
    }
    blueprint = build_blueprint(rrb, labels, eye, center)
    rr.send_blueprint(blueprint)

    # 比較動画用の flow range（旧実装と同じ p2/p98、sr stage の描画対象フレームから）
    flow_values = []
    for index in indices:
        payload_sr = load_frame_npz(input_dir, "sr", index)
        if payload_sr is None:
            continue
        flag = np.asarray(payload_sr["sr_valid"], dtype=bool)
        if not flag.any():
            continue
        flow = np.asarray(payload_sr["sr_flow"], dtype=np.float64)
        magnitude = np.linalg.norm(flow, axis=-1)
        flow_values.append(magnitude[::4, ::4][flag[::4, ::4]])
    if flow_values:
        joined = np.concatenate([value for value in flow_values if value.size > 0])
        flow_range = [float(v) for v in np.percentile(joined, [2, 98])]
    else:
        flow_range = [0.0, 1.0]
    if depth_range is None:
        depth_values = []
        for index in indices:
            final = require_frame(input_dir, "final", index, ("final_depth", "final_valid"))
            depth = np.asarray(final["final_depth"], dtype=np.float64)
            flag = np.asarray(final["final_valid"], dtype=bool)
            depth_values.append(depth[::8, ::8][flag[::8, ::8]])
        joined_depth = np.concatenate(depth_values)
        depth_range = [float(v) for v in np.percentile(joined_depth, [2, 98])]
    log(f"[{dataset}] depth range {depth_range[0]:.3f}–{depth_range[1]:.3f}, "
        f"flow range {flow_range[0]:.4f}–{flow_range[1]:.4f} (arrow_provisional)")

    sr_rate = np.zeros(rendered_count, dtype=np.float64)
    final_rate = np.zeros(rendered_count, dtype=np.float64)
    sr_count = np.zeros(rendered_count, dtype=np.int64)
    alive_counts = np.zeros(rendered_count, dtype=np.int64)
    frame_reports: list[dict] = []
    thumbs: list[Image.Image] = []
    pixels = height * width
    previous_frame: dict | None = None

    for position, index in enumerate(indices):
        frame = load_render_frame(input_dir, frames[index], index, height, width)

        # --- SR 移流: 実測 SR 変位で前フレームの 3D 位置を進め、観測面へ再アンカー ----
        if position == 0:
            track_rows, track_cols, track_world, track_alive = init_tracks(frame)
            track_positions = np.full((rendered_count, len(track_rows), 3), np.nan, dtype=np.float32)
            if len(track_rows):
                track_positions[0] = track_world.astype(np.float32)
            track_colors = colors_for(len(track_rows))
        elif previous_frame is not None:
            track_rows, track_cols, track_world, track_alive = advance_tracks(
                track_rows, track_cols, track_world, track_alive, previous_frame, frame,
                height, width,
            )
            if track_positions.shape[1]:
                track_positions[position] = track_world.astype(np.float32)
        previous_frame = frame
        alive_counts[position] = int(track_alive.sum()) if len(track_alive) else 0

        # --- パネル生成（統計は現在フレーム分を先に確定させる） ---
        current_sr_count = int(frame["sr_valid"].sum())
        current_final_count = int(frame["valid"].sum())
        sr_rate[position] = current_sr_count / pixels
        final_rate[position] = current_final_count / pixels
        sr_count[position] = current_sr_count
        cloud = cloud_for_frame(frame["depth"], frame["valid"], frame["K"], frame["metric"],
                                frame["rgb"])
        arrows = sr_arrow_samples(frame["flow"], frame["sr_valid"], frame["depth"], frame["valid"],
                                  frame["K"], frame["metric"])
        du, dv, motion_ok = sr_motion_2d(frame["flow"], frame["sr_valid"], frame["depth"],
                                         frame["valid"], frame["K"])
        normals, normal_ok = normals_from_depth(frame["depth"], frame["valid"], frame["K"])
        normal_rgb = np.full((*normals.shape[:2], 3), 0.12, dtype=np.float64)
        normal_rgb[normal_ok] = normals[normal_ok] * 0.5 + 0.5
        panels = {
            "input": Image.fromarray(frame["rgb"]),
            "motion": render_motion_panel(frame["rgb"], du, dv, motion_ok, ARROW_STRIDE,
                                          flow_range[0], flow_range[1]),
            "depth": colorize_depth(frame["depth"], frame["valid"], depth_range[0], depth_range[1]),
            "normal": Image.fromarray((np.clip(normal_rgb, 0, 1) * 255).astype(np.uint8)),
            "supportvis": render_support_vis(frame["sr_valid"], frame["valid"]),
            "mask": render_support_mask(frame["sr_valid"], frame["valid"]),
            "scene3d": render_sr_scene3d(position, cloud, arrows, matrices, valid, bounds,
                                         frustum_scale),
            "cam3d": render_cam3d(position, {"camera_to_reference": matrices, "valid": valid},
                                  extent=None, frustum_scale=None),
            "tracks3d": render_tracks3d(
                position,
                {"xyz": track_positions, "colors": np.asarray(track_colors, dtype=np.uint8)},
                bounds,
            ),
            "summary": render_sr_summary(position, {
                "frames_rendered": rendered_count, "preview": preview, "pixels": pixels,
                "camera_valid": valid, "sr_rate": sr_rate,
                "current": {
                    "sr_rate": current_sr_count / pixels,
                    "final_rate": current_final_count / pixels,
                    "sr_count": current_sr_count,
                    "alive": int(alive_counts[position]),
                },
                "track_total": int(track_positions.shape[1]) if track_positions.ndim == 3 else 0,
            }),
        }
        thumb = Image.fromarray(frame["rgb"]).resize((74, 46), Image.Resampling.BILINEAR)
        thumbs.append(thumb)
        note = (
            "arrow_provisional (non-metric)・SR/generated・分母=全画素(H*W)・" + scope
            + (f" {rendered_count}/{total}" if preview else "")
        )
        board = compose_sr_board(panels, position, rendered_count, thumbs, note,
                                 f"{depth_range[0]:.2f}–{depth_range[1]:.2f} provisional")
        board.save(work_dir / f"frame_{position:04d}.png")

        # --- Rerun 記録（cameras/arrows は実 frame の timeline、空でも必ず log） ---
        # 座標は camera 慣習の metric を display=(x, z, -y) Z-up へ exactly once 変換して
        # log する（frusta/track_heads と同じ world 座標系。raw を混ぜない）。
        rr.set_time("frame", sequence=index)
        rr.set_time("time", duration=float(index) / fps)
        rr.log("world/cloud", rr.Points3D(display(cloud["points"]), colors=cloud["rgb"],
                                          radii=POINT_RADIUS))
        rr.log("world/sr_arrows", rr.Arrows3D(origins=display(arrows[0]), vectors=display(arrows[1]),
                                              colors=np.array([[255, 120, 51]], dtype=np.uint8),
                                              radii=ARROW_RADIUS))
        if valid[position]:
            segments = frustum_segments(matrices[position], frustum_scale)
            rr.log("world/current_camera",
                   rr.LineStrips3D(segments, colors=[list(CAMERA_CURRENT)], radii=0.03))
        heads = track_positions[position]
        keep = np.isfinite(heads).all(axis=-1)
        rr.log("world/track_heads", rr.Points3D(
            display(heads[keep]).astype(np.float32) if keep.any() else np.zeros((0, 3), np.float32),
            colors=np.asarray(track_colors, dtype=np.uint8)[keep] if keep.any()
            else np.zeros((0, 3), np.uint8),
            radii=0.04,
        ))
        rr.log("view/input", rr.EncodedImage(contents=jpeg_bytes(panels["input"]),
                                             media_type="image/jpeg"))
        rr.log("view/motion", rr.EncodedImage(contents=jpeg_bytes(panels["motion"]),
                                              media_type="image/jpeg"))
        rr.log("view/depth", rr.EncodedImage(contents=jpeg_bytes(panels["depth"]),
                                             media_type="image/jpeg"))
        rr.log("view/normal", rr.EncodedImage(contents=jpeg_bytes(panels["normal"]),
                                              media_type="image/jpeg"))
        rr.log("view/mask", rr.EncodedImage(contents=jpeg_bytes(panels["mask"]),
                                            media_type="image/jpeg"))
        rr.log("view/support", rr.EncodedImage(contents=jpeg_bytes(panels["supportvis"]),
                                               media_type="image/jpeg"))

        frame_reports.append({
            "frame_index": index,
            "pair_index": [int(v) for v in frame["pair"]],
            "has_pair": frame["has_pair"],
            "shape": [height, width],
            "final_valid_pixels": int(frame["valid"].sum()),
            "sr_support_pixels": int(frame["sr_valid"].sum()),
            "sr_valid_fraction": round(float(frame["sr_valid"].sum()) / pixels, 6),
            "sha256": {stage: sha256_file(stage_path(input_dir, stage, index))
                       for stage in STAGE_NAMES},
            "rgb_sha256": frames[index].get("rgb_sha256"),
        })
        log(f"[{dataset}] frame {index} ({position + 1}/{rendered_count}) "
            f"sr_support={sr_count[position]} alive_tracks={alive_counts[position]}")
        if (position + 1) % 20 == 0:
            check_budget(visuals_dir)

    # 静的記録と保存（blueprint は埋め込み、.rbl も同じ定義で出す）
    strips, strip_colors = [], []
    for slot in range(track_positions.shape[1]):
        for strip in finite_strips(track_positions[:, slot, :]):
            strips.append(display(strip))
            strip_colors.append(tuple(int(v) for v in track_colors[slot]))
    log_static(rr, matrices, valid, strips, strip_colors, frustum_scale, bounds)
    rrd_path = visuals_dir / f"{dataset}_sr_workflow.rrd"
    rbl_path = visuals_dir / f"{dataset}_sr_workflow.rbl"
    rr.save(str(rrd_path))
    blueprint.save(app_id, str(rbl_path))
    rr.disconnect()  # rrd を flush（テスト/連続実行で即座に読めるようにする）

    # 代表 PNG（mp4 中央フレームの board を複製）→ mp4 化 → 作業フォルダ削除
    png_path = visuals_dir / f"{dataset}_sr_multipanel.png"
    board_index = len(indices) // 2
    shutil.copyfile(work_dir / f"frame_{board_index:04d}.png", png_path)
    mp4_path = visuals_dir / f"{dataset}_sr_multipanel.mp4"
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg が見つかりません")
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-framerate", str(fps),
            "-i", str(work_dir / "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
            str(mp4_path),
        ],
        check=True,
    )
    shutil.rmtree(work_dir, ignore_errors=True)
    log(f"[{dataset}] multipanel: {mp4_path.name}（{mp4_path.stat().st_size / 2**20:.1f} MB）")

    # 比較動画（旧 7 パネル + footprint ARROW）
    compare = build_sr_compare(input_dir, frames, indices, fps, visuals_dir, depth_range,
                               flow_range)
    log(f"[{dataset}] ARROW compare 表示: grid binning {compare['arrow_display']['grid']} フレーム / "
        f"area fallback {compare['arrow_display']['area']} フレーム")

    total_bytes = check_budget(visuals_dir)
    report = {
        "dataset": dataset,
        "input_dir": str(input_dir),
        "outputs": {
            "sr_multipanel_mp4": str(mp4_path),
            "sr_multipanel_png": str(png_path),
            "sr_workflow_rrd": str(rrd_path),
            "sr_workflow_rbl": str(rbl_path),
            "sr_compare_mp4": str(compare["mp4"]),
            "sr_compare_png": str(compare["png"]),
        },
        "frames_rendered": rendered_count,
        "frames_total": total,
        "frame_ids": indices,
        "frame_shape": [height, width],
        "preview": preview,
        "fps": fps,
        "units": "arrow_provisional",
        "metric_calibrated": False,
        "depth_range": [float(v) for v in depth_range],
        "flow_range": [float(v) for v in flow_range],
        "config": {
            "cloud_stride": CLOUD_STRIDE,
            "arrow_stride": ARROW_STRIDE,
            "point_radius_rrd": POINT_RADIUS,
            "arrow_radius_rrd": ARROW_RADIUS,
            "trajectory_picks": TRAJECTORY_PICKS,
            "display_transform": "(x, z, -y) Z-up",
            "viewport": VIEW_3D,
            "layout": "1920x1080 / columns 18-18-40-24",
            "view_bounds": "global 1st-99th percentile cloud extent plus all cameras; display only",
        },
        "counts": {
            "sr_support_pixels_total": int(sr_count.sum()),
            "alive_trajs_final": int(alive_counts[-1]) if rendered_count else 0,
            "trajs_total": int(track_positions.shape[1]) if track_positions.ndim == 3 else 0,
        },
        "compare_arrow_display": compare["arrow_display"],
        "frames": frame_reports,
        "bytes": total_bytes,
        "elapsed_s": round(time.perf_counter() - started, 1),
        "created_at": utc_now(),
    }
    return report


# ---------------------------------------------------------------------------
# CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SR workflow 可視化（depth_detail_oct16 の独立 consumer）")
    parser.add_argument("--input-dir", type=Path, required=True, help="depth_detail 出力ルート")
    parser.add_argument("--output-dir", type=Path, required=True, help="SR 可視化の出力ルート")
    parser.add_argument("--dataset", choices=("all", "grass", "railway"), required=True)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="明示的な preview（先頭 N フレームのみ。full とは呼称しない）")
    args = parser.parse_args(argv)
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    datasets = ("grass", "railway") if args.dataset == "all" else (args.dataset,)
    free = preflight_disk(args.output_dir)
    log(f"preflight: 空きディスク {free / 2**30:.1f} GiB (必要 {MIN_FREE_BYTES / 2**30:.0f} GiB)")
    for dataset in datasets:
        report = render_dataset(args, dataset)
        manifest_path = args.output_dir / dataset / "sr_visuals" / f"{dataset}_sr_manifest.json"
        manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                                 encoding="utf-8")
        log(f"[{dataset}] manifest: {manifest_path}")
    log(f"完了: {len(datasets)} dataset(s) -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
