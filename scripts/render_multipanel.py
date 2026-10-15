#!/usr/bin/env python3
"""ARROW マルチパネル可視化（LAYOUT_SPEC 実装）の素材生成・合成器。

仕様: results/video_concept/LAYOUT_SPEC.md
（ヘッダー / メイン4列18-18-40-24%（第1/2/4列は縦3段） / タイムライン / 処理フロー帯、
全パネル同一フレーム同期、追跡点の識別色は全関連パネルで共通）

このモジュールは「素材生成（materials）」から実装を進める:
- 既存キャッシュ（render_dashboard.py: overlay/depth/canvas/series）を再利用し、
  新パネル素材をキャッシュへ追加する:
    panels/NNNN_motion.png    2D移動（元画像を淡く + 識別色の矢印。フレームk→k+1 の差分）
    panels/NNNN_confvis.png   信頼度・可視性（左=信頼度 右=可視性、明暗、共通の画像座標）
    panels/NNNN_normal.png    表面法線（成分→RGB、全フレーム共通の変換規則）
    panels/NNNN_mask.png      解析対象領域（白=解析対象、黒=空領域。前景抽出ではない）
    thumbs/NNNN_thumb.png     タイムライン用サムネイル
    series/cameras.npz        カメラ姿勢（camera_to_reference / valid / intrinsics）
    series/summary.npz        集計（全追跡点数・可視追跡率・有効3D点数・カメラ有効）

使い方（素材生成）::

    uv run python scripts/render_multipanel.py --scene results/demo_seq/railway_seq.npz
"""

import argparse
from dataclasses import fields
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

from render_dashboard import (  # noqa: E402
    default_cache_dir, ensure_cache, group_slice, track_picks, utc_now,
)

REPO = Path(__file__).resolve().parents[1]

# 信頼度の正規化（明暗表現の定義）: クリップ全体の有限値の 5–95 パーセンタイルで [0,1] に写す。
# 0〜1 を正解確率として解釈させないための定義。
CONFIDENCE_PERCENTILES = (5, 95)
VISIBILITY_THRESHOLD = 0.0  # sigmoid(logit) > 0.5 ⇔ logit > 0 を「可視判定」とする


def log(message: str) -> None:
    print(message, flush=True)


def group_numpy(result, task: str):
    entry, points, uv, xy = group_slice(result, task)
    return entry, points.numpy(), uv.numpy(), xy.numpy()


def render_motion(result, frame_t: int, seeds: int, picks: np.ndarray, colors, size) -> Image.Image:
    """2D移動: 元画像を淡く残し、フレームk→k+1 の差分を識別色の矢印で描く。"""
    width, height = size
    frame = (result.images[0, frame_t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    background = Image.fromarray(frame).convert("RGB").point(lambda v: int(v * 0.45 + 120 * 0.55))
    _, _, uv, _ = group_numpy(result, "track")
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    if frame_t + 1 < len(result.metadata["frame_info"]):
        for k, seed in enumerate(picks):
            start = uv[frame_t * seeds + seed]
            end = uv[(frame_t + 1) * seeds + seed]
            if not (np.isfinite(start).all() and np.isfinite(end).all()):
                continue
            x0, y0 = start[0] * width - 0.5, start[1] * height - 0.5
            x1, y1 = end[0] * width - 0.5, end[1] * height - 0.5
            length = float(np.hypot(x1 - x0, y1 - y0))
            if length < 1.5:
                continue
            color = (*colors[k], 230)
            draw.line([x0, y0, x1, y1], fill=color, width=2)
            ux, uy = (x1 - x0) / length, (y1 - y0) / length
            px, py = -uy, ux
            head = 5.0
            draw.polygon(
                [(x1, y1), (x1 - ux * head + px * head * 0.55, y1 - uy * head + py * head * 0.55),
                 (x1 - ux * head - px * head * 0.55, y1 - uy * head - py * head * 0.55)],
                fill=color,
            )
    return Image.alpha_composite(background.convert("RGBA"), layer).convert("RGB")


def render_confvis(result, frame_t: int, seeds: int, pick: int, size,
                   confidence_range, width_px: int, height_px: int) -> Image.Image:
    """信頼度・可視性: 左=信頼度、右=可視性（明暗）。全シードを現在位置へスタンプする。"""
    width, height = size
    entry, _, uv, _ = group_numpy(result, "track")
    confidence = result.predictions.point_confidence[0].numpy()
    visibility = result.predictions.visibility_logits[0].numpy()
    start, count = entry["start"], entry["count"]
    confidence = confidence[start : start + count]
    visibility = visibility[start : start + count]

    low, high = confidence_range
    half_w = width_px // 2
    canvas = np.zeros((height_px, width_px), dtype=np.float32)
    for seed in range(seeds):
        index = frame_t * seeds + seed
        value = uv[index]
        if not np.isfinite(value).all():
            continue
        x = int(np.clip(round(value[0] * width - 0.5), 0, width - 1))
        y = int(np.clip(round(value[1] * height - 0.5), 0, height - 1))
        px = int(round(x * (half_w - 1) / max(1, width - 1)))
        py = int(round(y * (height_px - 1) / max(1, height - 1)))
        conf = confidence[index]
        vis = visibility[index]
        if np.isfinite(conf):
            value = float(np.clip((conf - low) / max(1e-6, high - low), 0, 1))
            canvas[max(0, py - 1) : py + 2, max(0, px - 1) : px + 2] = value
        if np.isfinite(vis):
            value = float(1.0 / (1.0 + np.exp(-vis)))
            canvas[max(0, py - 1) : py + 2, px + half_w - 1 : px + half_w + 2] = value
    canvas = np.clip(canvas, 0, 1)
    rgb = np.stack([canvas] * 3, axis=-1)
    rgb[:, half_w - 2 : half_w + 2] = [0.45, 0.5, 0.6]  # 中央の区切り線
    image = Image.fromarray((rgb * 255).astype(np.uint8))
    return image.resize((width_px, height_px), Image.Resampling.NEAREST)


def render_normal(result, frame_t: int, size) -> Image.Image:
    """表面法線: 成分→RGB（n*0.5+0.5）。全フレーム共通の変換規則。"""
    width, height = size
    entry, _, _, xy = group_numpy(result, "depth")
    normals = result.predictions.normals[0].numpy()
    start, count = entry["start"], entry["count"]
    frames = len(result.metadata["frame_info"])
    per_frame = count // frames
    first = xy[:per_frame]
    cols, rows = len(np.unique(first[:, 0])), len(np.unique(first[:, 1]))
    grid = normals[start : start + count].reshape(frames, rows, cols, 3)[frame_t]
    rgb = grid * 0.5 + 0.5
    rgb[~np.isfinite(grid)] = 0.12
    image = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    return image.resize((width, height), Image.Resampling.BICUBIC)


def render_mask(sky_frame: np.ndarray, size) -> Image.Image:
    """解析対象領域: 白=解析対象（非空）、黒=空領域（除外）。前景抽出ではない。"""
    width, height = size
    gray = np.where(sky_frame, 0, 255).astype(np.uint8)
    image = Image.fromarray(gray).resize((width, height), Image.Resampling.NEAREST)
    return image.convert("RGB")


def build_materials(result, cache_dir: Path, pick: int, rebuild: bool = False,
                    alignment: dict | None = None) -> dict:
    """新パネル素材と系列をキャッシュへ追加する（既存分は再利用）。

    alignment 指定時は深度パネルを「世界共通の色範囲・スケール」で上書き再生成し、
    返り値の depth_range も世界範囲に置き換える（クリップ結合時の色の一貫性）。
    """
    # 信頼度の正規化範囲（クリップ全体・有限値のみ）
    confidence = result.predictions.point_confidence[0].numpy()
    finite = confidence[np.isfinite(confidence)]
    low, high = np.percentile(finite, list(CONFIDENCE_PERCENTILES))

    frames = len(result.metadata["frame_info"])
    height, width = result.images.shape[-2:]
    entry, _, _, _ = group_numpy(result, "track")
    seeds = entry["points_per_frame"]
    picks = track_picks(seeds, pick)
    from render_dashboard import colors_for

    colors = colors_for(len(picks))

    panels = cache_dir / "panels"
    panels.mkdir(parents=True, exist_ok=True)
    thumbs = cache_dir / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    series = cache_dir / "series"
    series.mkdir(parents=True, exist_ok=True)

    # 空マスク（既存機能 segment_sky を使用。キャッシュに保存）
    sky_path = series / "sky_mask.npz"
    if rebuild or not sky_path.is_file():
        from arrow.sky import segment_sky

        started = time.perf_counter()
        sky = segment_sky(result.images)
        np.savez_compressed(sky_path, sky=sky.numpy().astype(bool))
        log(f"  空マスクを生成（{time.perf_counter() - started:.1f} 秒）")
    sky = np.load(sky_path)["sky"][0]  # [F, H, W]

    started = time.perf_counter()
    for frame_t in range(frames):
        targets = [
            ("motion", lambda t=frame_t: render_motion(result, t, seeds, picks, colors, (width, height))),
            ("confvis", lambda t=frame_t: render_confvis(result, t, seeds, pick, (width, height), (low, high),
                                                         width * 2, height)),
            ("normal", lambda t=frame_t: render_normal(result, t, (width, height))),
            ("mask", lambda t=frame_t: render_mask(sky[t], (width, height))),
        ]
        for name, render in targets:
            target = panels / f"{frame_t:04d}_{name}.png"
            if rebuild or not target.is_file():
                render().save(target)
        thumb = thumbs / f"{frame_t:04d}_thumb.png"
        if rebuild or not thumb.is_file():
            frame = (result.images[0, frame_t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(frame).resize(
                (max(1, round(width * 54 / height)), 54), Image.Resampling.BILINEAR
            ).save(thumb)
    log(f"  パネル素材 {frames} フレーム分を生成（{time.perf_counter() - started:.1f} 秒）")

    # 深度パネル: alignment 指定時は世界共通の色範囲・スケールで上書き再生成する
    # （クリップごとの相対値を世界スケールに補正し、クリップを結合しても色がちぐはぐにならないように）
    world_depth_range = None
    if alignment and alignment.get("depth_range_world"):
        from render_dashboard import depth_grid, render_depth

        low_world, high_world = (float(value) for value in alignment["depth_range_world"])
        depth_scale = float(alignment["scale"])
        z, _, _, _ = depth_grid(result)
        for frame_t in range(frames):
            render_depth(z * depth_scale, frame_t, low_world, high_world, width, height).save(
                panels / f"{frame_t:04d}_depth.png"
            )
        world_depth_range = {"low": low_world, "high": high_world}
        log(f"  深度パネルを世界範囲で再生成（{low_world:.2f}–{high_world:.2f}、"
            f"スケール {depth_scale:.3f}）")

    # カメラ姿勢（pose グループから推定して保存）
    cameras_path = series / "cameras.npz"
    if rebuild or not cameras_path.is_file():
        from arrow.tasks.pose import estimate_poses

        pose_entry = next(g for g in result.metadata["groups"] if g["task"] == "pose")
        pose_start, pose_count = pose_entry["start"], pose_entry["count"]
        pose_predictions = type(result.predictions)(
            **{
                f.name: getattr(result.predictions, f.name)[:, pose_start : pose_start + pose_count]
                for f in fields(type(result.predictions))
            }
        )
        xy = result.queries.xy[:, pose_start : pose_start + pose_count]
        poses = estimate_poses(
            pose_predictions, xy, num_frames=frames,
            reference_frame=pose_entry["reference_frame"], **pose_entry["pose"],
        )
        np.savez_compressed(
            cameras_path,
            camera_to_reference=poses.camera_to_reference[0].numpy().astype(np.float32),
            valid=poses.valid[0].numpy().astype(bool),
            intrinsics=poses.intrinsics[0].numpy().astype(np.float32),
        )
        log(f"  カメラ姿勢を保存（有効 {int(poses.valid.sum())}/{frames}）")

    # 集計系列（可視追跡率などの実測値）
    summary_path = series / "summary.npz"
    if rebuild or not summary_path.is_file():
        visibility = result.predictions.visibility_logits[0].numpy()
        track_entry = next(g for g in result.metadata["groups"] if g["task"] == "track")
        t_start, t_count = track_entry["start"], track_entry["count"]
        logits = visibility[t_start : t_start + t_count].reshape(frames, seeds)
        visible = ((logits > VISIBILITY_THRESHOLD) & np.isfinite(logits)).sum(axis=1)
        reconstruct_entry = next(g for g in result.metadata["groups"] if g["task"] == "reconstruct")
        r_start, r_count = reconstruct_entry["start"], reconstruct_entry["count"]
        reconstruct_points = result.predictions.points[0, r_start : r_start + r_count].numpy()
        per_frame = r_count // frames
        valid_points = [
            int(np.isfinite(reconstruct_points[f * per_frame : (f + 1) * per_frame]).all(-1).sum())
            for f in range(frames)
        ]
        np.savez_compressed(
            summary_path,
            frames=np.int64(frames),
            total_tracks=np.int64(seeds),
            visible_count=visible.astype(np.int64),
            visible_rate=(visible / max(1, seeds)).astype(np.float32),
            valid_points=np.array(valid_points, dtype=np.int64),
        )
        log(f"  集計系列を保存（全追跡点数 {seeds}）")

    payload: dict = {"materials": {
        "created_at": utc_now(),
        "panels": ["overlay", "depth", "canvas", "motion", "confvis", "normal", "mask"],
        "thumbs": "thumbs/{index:04d}_thumb.png",
        "series": {"cameras": "series/cameras.npz", "summary": "series/summary.npz",
                   "sky_mask": "series/sky_mask.npz"},
        "confidence_normalization": {"percentiles": list(CONFIDENCE_PERCENTILES), "low": float(low),
                                     "high": float(high)},
        "visibility_threshold": VISIBILITY_THRESHOLD,
    }}
    if world_depth_range is not None:
        payload["depth_range"] = world_depth_range
    return payload


# --- 3D 系パネル（series からオフライン描画・視点固定・キャッシュ化） ---------------------

MARGIN = 20
GAP = 12
MAIN_H = 920
TIMELINE_H = 108
PANEL_TITLE_H = 26
COLUMN_SHARES = (0.18, 0.18, 0.40, 0.24)

BG = (15, 17, 22)
PANEL_BG = (27, 30, 37)
BORDER = (58, 64, 74)
TITLE_COLOR = (176, 184, 196)
TEXT_COLOR = (232, 236, 242)
ACCENT = (90, 160, 255)

# 観察視点（固定。再生中は動かさない）
VIEW_3D = {"elev": 24.0, "azim": -68.0}
FRUSTUM_SCALE = 0.45
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def font(size: int):
    from PIL import ImageFont

    return ImageFont.truetype(FONT_BOLD, size)


def display(points: np.ndarray) -> np.ndarray:
    """参照カメラ座標 (x 右, y 下, z 前) → 表示用 (x, z, -y)（Z 上・右手系）。"""
    source = np.asarray(points, dtype=np.float32)
    out = np.empty_like(source)
    out[..., 0] = source[..., 0]
    out[..., 1] = source[..., 2]
    out[..., 2] = -source[..., 1]
    return out


def frustum_segments(matrix: np.ndarray, scale: float) -> list[np.ndarray]:
    """camera_to_reference [4,4] から視錐台の線分（表示座標）を作る。"""
    rotation = matrix[:3, :3]
    center = matrix[:3, 3]
    right, down, forward = rotation[:, 0], rotation[:, 1], rotation[:, 2]
    depth = scale
    half_w, half_h = scale * 0.35, scale * 0.28
    base = center + forward * depth
    corners = [
        display(corner)
        for corner in (
            base - right * half_w - down * half_h,
            base + right * half_w - down * half_h,
            base + right * half_w + down * half_h,
            base - right * half_w + down * half_h,
        )
    ]
    apex = display(center)
    segments = [np.stack([apex, corner]) for corner in corners]
    segments += [np.stack([corners[i], corners[(i + 1) % 4]]) for i in range(4)]
    return segments


def scene_bounds(*arrays: np.ndarray) -> dict:
    both = np.concatenate([a[~np.isnan(a).any(-1)] for a in arrays])
    low, high = both.min(0), both.max(0)
    pad = 0.08 * float(np.max(high - low))
    return {
        "x": (float(low[0] - pad), float(high[0] + pad)),
        "y": (float(low[1] - pad), float(high[1] + pad)),
        "z": (float(low[2] - pad), float(high[2] + pad)),
    }


def _setup_axes(ax, bounds, *, grid=True) -> None:
    ax.set_xlim(*bounds["x"])
    ax.set_ylim(*bounds["y"])
    ax.set_zlim(*bounds["z"])
    size = (bounds["x"][1] - bounds["x"][0], bounds["y"][1] - bounds["y"][0],
            bounds["z"][1] - bounds["z"][0])
    ax.set_box_aspect(size)
    ax.set_axis_off()
    if grid:
        z = bounds["z"][0]
        for gx in np.linspace(bounds["x"][0], bounds["x"][1], 7):
            ax.plot([gx, gx], list(bounds["y"]), [z, z], color="#262B35", lw=0.6, alpha=0.6)
        for gy in np.linspace(bounds["y"][0], bounds["y"][1], 7):
            ax.plot(list(bounds["x"]), [gy, gy], [z, z], color="#262B35", lw=0.6, alpha=0.6)


def _axis_indicator(ax, bounds) -> None:
    """画面隅の小さな XYZ 軸（x=赤, y=緑, z=青）。"""
    x0 = bounds["x"][1] - 0.12 * (bounds["x"][1] - bounds["x"][0])
    y0 = bounds["y"][1] - 0.12 * (bounds["y"][1] - bounds["y"][0])
    z0 = bounds["z"][0]
    scale = 0.08 * float(np.mean([bounds["x"][1] - bounds["x"][0],
                                  bounds["y"][1] - bounds["y"][0],
                                  bounds["z"][1] - bounds["z"][0]]))
    for vector, color in [((scale, 0, 0), "#FF6B6B"), ((0, scale, 0), "#7BD88F"), ((0, 0, scale), "#6B9BFF")]:
        ax.plot([x0, x0 + vector[0]], [y0, y0 + vector[1]], [z0, z0 + vector[2]], color=color, lw=2.0)


def render_scene3d(frame_t: int, tracks, cloud, cameras, bounds,
                   frustum_scale: float = FRUSTUM_SCALE) -> Image.Image:
    """3D再構成・追跡: 点群 + 軌跡（現在=明るいマーカー/過去=半透明）+ カメラ視錐台（現在を強調）。

    frustum_scale は視錐台の大きさ（表示範囲に比例させると世界座標でも視認できる）。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    xyz, colors = tracks["xyz"], tracks["colors"].astype(np.float32) / 255.0
    valid = cameras["valid"]

    fig = plt.figure(figsize=(7.3, 6.8), dpi=100, facecolor="#0F1116")
    ax = fig.add_subplot(projection="3d", facecolor="#0F1116")
    _setup_axes(ax, bounds)

    points = display(cloud["points"])
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],  # pyright: ignore[reportArgumentType]
               s=2, c=cloud["rgb"].astype(np.float32) / 255.0, linewidths=0, alpha=0.85)

    for seed in range(xyz.shape[1]):
        line = display(xyz[:, seed, :])
        keep = ~np.isnan(line).any(-1)
        if keep.sum() >= 2:
            ax.plot(line[keep, 0], line[keep, 1], line[keep, 2],
                    color=colors[seed], lw=1.5, alpha=0.35)
    heads = display(xyz[frame_t])
    keep = ~np.isnan(heads).any(-1)
    ax.scatter(heads[keep, 0], heads[keep, 1], heads[keep, 2],  # pyright: ignore[reportArgumentType]
               s=34, c=colors[keep], depthshade=False, edgecolors="white", linewidths=0.4)

    cameras_to_ref = cameras["camera_to_reference"]
    for frame in range(len(cameras_to_ref)):
        if not valid[frame]:
            continue
        current = frame == frame_t
        for segment in frustum_segments(cameras_to_ref[frame], frustum_scale):
            ax.plot(segment[:, 0], segment[:, 1], segment[:, 2],
                    color="#5AA0FF" if current else "#4A6A9C",
                    lw=2.0 if current else 0.9, alpha=1.0 if current else 0.5)

    _axis_indicator(ax, bounds)
    ax.view_init(elev=VIEW_3D["elev"], azim=VIEW_3D["azim"])
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    array = np.asarray(canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return Image.fromarray(array.copy())


def render_cam3d(frame_t: int, cameras, extent=None) -> Image.Image:
    """カメラ姿勢: 上方からの視点。位置（x, 前）と撮影方向。現在フレームを強調。

    extent を渡すと表示範囲を固定できる（クリップ間で共通の見え方にする）。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    matrices = cameras["camera_to_reference"]
    valid = cameras["valid"]
    positions = np.array([display(matrix[:3, 3]) for matrix in matrices], dtype=np.float32)

    # 表示範囲を先に決め、視錐台の大きさを範囲に比例させる
    finite = positions[valid]
    if extent is not None:
        span = max(float(extent[0][1]) - float(extent[0][0]),
                   float(extent[1][1]) - float(extent[1][0]))
    elif len(finite):
        span = float(np.max(finite.max(0) - finite.min(0)))
    else:
        span = 1.0
    span = max(span, 1e-6)
    frustum_scale = span * 0.06

    fig = plt.figure(figsize=(4.42, 1.98), dpi=100, facecolor="#0F1116")
    ax = fig.add_subplot(facecolor="#0F1116")
    for frame in range(len(matrices)):
        if not valid[frame]:
            continue
        current = frame == frame_t
        color = "#5AA0FF" if current else "#3A5A8C"
        alpha = 1.0 if current else 0.35
        linewidth = 2.2 if current else 0.8
        for segment in frustum_segments(matrices[frame], frustum_scale):
            ax.plot(segment[:, 0], segment[:, 1], color=color, lw=linewidth, alpha=alpha,
                    solid_capstyle="round")
        x, y = positions[frame, 0], positions[frame, 1]
        ax.plot(x, y, "o", color=color, markersize=4.5 if current else 2.0,
                alpha=1.0 if current else 0.45)
    if extent is not None:
        ax.set_xlim(float(extent[0][0]), float(extent[0][1]))
        ax.set_ylim(float(extent[1][0]), float(extent[1][1]))
    elif len(finite):
        pad = 0.15 * span
        ax.set_xlim(float(finite[:, 0].min()) - pad, float(finite[:, 0].max()) + pad)
        ax.set_ylim(float(finite[:, 1].min()) - pad, float(finite[:, 1].max()) + pad)
    ax.set_aspect("equal")
    ax.grid(True, color="#262B35", linewidth=0.6, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    array = np.asarray(canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return Image.fromarray(array.copy())


def render_tracks3d(frame_t: int, tracks, bounds) -> Image.Image:
    """3D軌跡: 背景形状（点群）を除いた軌跡のみ。識別色は元画像・中央と共通。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    xyz, colors = tracks["xyz"], tracks["colors"].astype(np.float32) / 255.0
    fig = plt.figure(figsize=(4.42, 1.98), dpi=100, facecolor="#0F1116")
    ax = fig.add_subplot(projection="3d", facecolor="#0F1116")
    _setup_axes(ax, bounds, grid=False)
    for seed in range(xyz.shape[1]):
        line = display(xyz[:, seed, :])
        keep = ~np.isnan(line).any(-1)
        if keep.sum() >= 2:
            ax.plot(line[keep, 0], line[keep, 1], line[keep, 2],
                    color=colors[seed], lw=1.3, alpha=0.4)
    heads = display(xyz[frame_t])
    keep = ~np.isnan(heads).any(-1)
    ax.scatter(heads[keep, 0], heads[keep, 1], heads[keep, 2],  # pyright: ignore[reportArgumentType]
               s=20, c=colors[keep], depthshade=False, edgecolors="white", linewidths=0.3)
    ax.view_init(elev=VIEW_3D["elev"], azim=VIEW_3D["azim"])
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    array = np.asarray(canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return Image.fromarray(array.copy())


def render_summary(frame_t: int, summary: dict) -> Image.Image:
    """集計: 実測値（フレーム数・追跡点数・有効3D点数・カメラ・可視追跡率）+ 小さなグラフ。"""
    width, height = 420, 195
    image = Image.new("RGB", (width, height), PANEL_BG)
    draw = ImageDraw.Draw(image)
    frames = int(summary["frames"])
    total = int(summary["total_tracks"])
    rate = summary["visible_rate"]
    points = summary["valid_points"]
    cameras_valid = summary["camera_valid"]
    rows = [
        ("入力フレーム数", f"{frames}"),
        ("全追跡点数", f"{total:,}"),
        ("有効3D点数（現在）", f"{int(points[frame_t]):,}"),
        ("有効カメラ姿勢", f"{int(cameras_valid.sum())} / {frames}"),
        ("可視追跡率（現在）", f"{100 * float(rate[frame_t]):.1f} %"),
    ]
    y = 6
    for label, value in rows:
        draw.text((10, y), label, font=font(15), fill=TITLE_COLOR)
        text = draw.textlength(value, font=font(15))
        draw.text((width - 12 - text, y), value, font=font(15), fill=TEXT_COLOR)
        y += 25

    # 可視追跡率の時間変化（実測）と現在位置マーカー
    graph = (10, y + 4, width - 12, height - 8)
    draw.rectangle(graph, outline=BORDER, width=1)
    xs = np.linspace(graph[0] + 2, graph[2] - 2, frames)
    ys = graph[3] - 4 - rate * (graph[3] - graph[1] - 8)
    points_xy = list(zip(xs.tolist(), ys.tolist()))
    draw.line(points_xy, fill=ACCENT, width=2)
    marker_x, marker_y = points_xy[frame_t]
    draw.ellipse([marker_x - 3, marker_y - 3, marker_x + 3, marker_y + 3], fill="#FFFFFF")
    return image


def build_3d_materials(cache_dir: Path, rebuild: bool = False,
                       alignment: dict | None = None) -> None:
    """3D 系パネル（scene3d/cam3d/tracks3d/summary）をキャッシュへ追加する。

    alignment 指定時は点群・軌跡・カメラを世界座標へ変換し、視点も世界共通の
    バウンズ（クリップ間で同一の見え方）にする。
    """
    series = cache_dir / "series"
    tracks = dict(np.load(series / "tracks.npz"))
    cloud = dict(np.load(series / "cloud.npz"))
    cameras = dict(np.load(series / "cameras.npz"))
    summary_npz = np.load(series / "summary.npz")
    summary = {key: summary_npz[key] for key in summary_npz.files}
    summary["camera_valid"] = cameras["valid"]

    if alignment is not None:
        scale = float(alignment["scale"])
        rotation = np.asarray(alignment["rotation"], dtype=np.float32)
        translation = np.asarray(alignment["translation"], dtype=np.float32)
        cloud["points"] = scale * (cloud["points"] @ rotation.T) + translation
        tracks["xyz"] = scale * (tracks["xyz"] @ rotation.T) + translation
        matrices = cameras["camera_to_reference"].astype(np.float32)
        new_rotation = np.einsum("ij,fjk->fik", rotation, matrices[:, :3, :3])
        new_translation = scale * (matrices[:, :3, 3] @ rotation.T) + translation
        cameras["camera_to_reference"] = np.concatenate(
            [np.concatenate([new_rotation, new_translation[..., None]], axis=2),
             np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (len(matrices), 1, 1))],
            axis=1,
        )

    frames = int(summary["frames"])
    panels = cache_dir / "panels"
    world = alignment.get("world_bounds") if alignment else None
    if world:
        low = np.asarray(world["low"], dtype=np.float32)
        high = np.asarray(world["high"], dtype=np.float32)
        # 参照座標 → 表示座標 (x, z, -y) に合わせて各軸の min/max を組み替える
        low_display = np.array([low[0], low[2], -high[1]], dtype=np.float32)
        high_display = np.array([high[0], high[2], -low[1]], dtype=np.float32)
        pad = 0.06 * float(np.max(high_display - low_display))
        bounds = {
            "x": (float(low_display[0] - pad), float(high_display[0] + pad)),
            "y": (float(low_display[1] - pad), float(high_display[1] + pad)),
            "z": (float(low_display[2] - pad), float(high_display[2] + pad)),
        }
    else:
        bounds = scene_bounds(display(cloud["points"]), display(tracks["xyz"]))
    track_bounds = scene_bounds(display(tracks["xyz"]))

    started = time.perf_counter()
    camera_extent = (bounds["x"], bounds["y"])
    # 視錐台の大きさは表示範囲に比例させる（世界座標では固定値だと見えないほど小さくなる）
    span = max(bounds["x"][1] - bounds["x"][0], bounds["y"][1] - bounds["y"][0],
               bounds["z"][1] - bounds["z"][0])
    frustum_scale = span * 0.05
    for frame_t in range(frames):
        targets = [
            ("scene3d", lambda t=frame_t: render_scene3d(t, tracks, cloud, cameras, bounds,
                                                         frustum_scale)),
            ("cam3d", lambda t=frame_t: render_cam3d(t, cameras, camera_extent)),
            ("tracks3d", lambda t=frame_t: render_tracks3d(t, tracks, track_bounds)),
            ("summary", lambda t=frame_t: render_summary(t, summary)),
        ]
        for name, render in targets:
            target = panels / f"{frame_t:04d}_{name}.png"
            if rebuild or not target.is_file():
                render().save(target)
    log(f"  3D系パネル {frames} フレーム分を生成（{time.perf_counter() - started:.1f} 秒）")


# --- レイアウト合成（4 領域・4 列） ------------------------------------------------------


def column_geometry() -> list[tuple[int, int]]:
    """(x, width) の一覧を仕様の割合で返す。"""
    effective = 1920 - 2 * MARGIN - 3 * GAP
    widths = [round(effective * share) for share in COLUMN_SHARES]
    widths[-1] += effective - sum(widths)
    columns, x = [], MARGIN
    for width in widths:
        columns.append((x, width))
        x += width + GAP
    return columns


def stacked_rows(top: int) -> list[tuple[int, int]]:
    """第1/2/4列の縦3段の (y, height) を返す。"""
    height = (MAIN_H - 2 * GAP) // 3
    rows = []
    y = top
    for index in range(3):
        rows.append((y, height if index < 2 else MAIN_H - 2 * (height + GAP)))
        y += height + GAP
    return rows


def load_panel(cache_dir: Path, frame_t: int, name: str) -> Image.Image | None:
    path = cache_dir / "panels" / f"{frame_t:04d}_{name}.png"
    return Image.open(path).convert("RGB") if path.is_file() else None


def fit_image(image: Image.Image, box_w: int, box_h: int) -> Image.Image:
    scale = min(box_w / image.width, box_h / image.height)
    return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                        Image.Resampling.LANCZOS)


def draw_panel(board: Image.Image, draw: ImageDraw.ImageDraw, box, title: str,
               image: Image.Image | None, *, legend: bool = False,
               note: str | None = None) -> tuple[int, int, int, int] | None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=8, fill=PANEL_BG, outline=BORDER, width=1)
    draw.text((x0 + 10, y0 + 4), title, font=font(16), fill=TITLE_COLOR)
    if note:
        note_width = draw.textlength(note, font=font(13))
        draw.text((x1 - 12 - note_width, y0 + 6), note, font=font(13), fill=TITLE_COLOR)
    content = (x0 + 6, y0 + PANEL_TITLE_H + 4, x1 - 6, y1 - 6)
    if image is None:
        draw.text(((x0 + x1) / 2 - 20, (y0 + y1) / 2 - 10), "未生成", font=font(18), fill=TITLE_COLOR)
        return None
    fitted = fit_image(image, content[2] - content[0], content[3] - content[1])
    paste_x = content[0] + (content[2] - content[0] - fitted.width) // 2
    paste_y = content[1] + (content[3] - content[1] - fitted.height) // 2
    board.paste(fitted, (paste_x, paste_y))
    if legend:
        from matplotlib import colormaps

        bar_h = min(fitted.height, content[3] - content[1])
        bar_x = content[2] - 12
        bar_y = paste_y
        cmap = colormaps["turbo"]
        for step in range(bar_h):
            value = 1.0 - step / max(1, bar_h - 1)
            rgb = tuple(int(c * 255) for c in cmap(value)[:3])
            draw.line([bar_x, bar_y + step, bar_x + 5, bar_y + step], fill=rgb)
        draw.rectangle([bar_x, bar_y, bar_x + 5, bar_y + bar_h], outline=BORDER, width=1)
        draw.text((bar_x - 26, bar_y - 2), "近", font=font(13), fill=TITLE_COLOR)
        draw.text((bar_x - 26, bar_y + bar_h - 14), "遠", font=font(13), fill=TITLE_COLOR)
    return (paste_x, paste_y, fitted.width, fitted.height)


def compose_frame(cache_dir: Path, frame_t: int) -> Image.Image:
    """仕様レイアウトの 1 フレームを合成する（全パネルが同一フレーム k に同期）。"""
    frames = len(list((cache_dir / "thumbs").glob("*.png")))
    board = Image.new("RGB", (1920, 1080), BG)
    draw = ImageDraw.Draw(board)

    columns = column_geometry()
    top = MARGIN
    rows = stacked_rows(top)

    panels = [
        (columns[0], [("元画像", "overlay", False), ("2D移動", "motion", False),
                      ("信頼度・可視性", "confvis", False)]),
        (columns[1], [("奥行き", "depth", True), ("表面法線", "normal", False),
                      ("解析対象領域", "mask", False)]),
        (columns[2], [("3D再構成・追跡", "scene3d", False)]),
        (columns[3], [("カメラ姿勢", "cam3d", False), ("3D軌跡", "tracks3d", False),
                      ("集計", "summary", False)]),
    ]
    for index, ((x, width), entries) in enumerate(panels):
        if index == 2:  # 第3列は分割しない
            box = (x, top, x + width, top + MAIN_H)
            title, name, has_legend = entries[0]
            draw_panel(board, draw, box, title, load_panel(cache_dir, frame_t, name),
                       legend=has_legend)
            continue
        for (y, height), (title, name, has_legend) in zip(rows, entries):
            box = (x, y, x + width, y + height)
            note = f"{frame_t + 1}/{frames}" if name == "overlay" else None
            rect = draw_panel(board, draw, box, title, load_panel(cache_dir, frame_t, name),
                              legend=has_legend, note=note)
            if name == "confvis" and rect is not None:
                paste_x, paste_y, fitted_w, fitted_h = rect
                half = fitted_w // 2
                for label, center_x in (("信頼度", paste_x + half // 2),
                                        ("可視性", paste_x + half + half // 2)):
                    text_width = draw.textlength(label, font=font(13))
                    draw.text((center_x - text_width / 2, paste_y + fitted_h + 3), label,
                              font=font(13), fill=TITLE_COLOR)

    # タイムライン領域
    timeline_y = MARGIN + MAIN_H + GAP
    draw.rounded_rectangle((MARGIN, timeline_y, 1920 - MARGIN, timeline_y + TIMELINE_H),
                           radius=8, fill=PANEL_BG, outline=BORDER, width=1)
    draw.polygon([(MARGIN + 22, timeline_y + 18), (MARGIN + 22, timeline_y + 44),
                  (MARGIN + 44, timeline_y + 31)], fill=TEXT_COLOR)  # 再生アイコン
    draw.rectangle([MARGIN + 52, timeline_y + 20, MARGIN + 56, timeline_y + 42],
                   fill=(110, 118, 130))  # 一時停止アイコン
    draw.rectangle([MARGIN + 60, timeline_y + 20, MARGIN + 64, timeline_y + 42],
                   fill=(110, 118, 130))
    draw.text((MARGIN + 78, timeline_y + 16), f"Frame {frame_t + 1}/{frames}",
              font=font(20), fill=TEXT_COLOR)
    bar_x0, bar_x1 = 300, 1900 - 20
    bar_y = timeline_y + 24
    draw.line([bar_x0, bar_y, bar_x1, bar_y], fill=(66, 72, 84), width=5)
    span = (bar_x1 - bar_x0) * frame_t / max(1, frames - 1)
    draw.line([bar_x0, bar_y, bar_x0 + span, bar_y], fill=ACCENT, width=5)
    draw.ellipse([bar_x0 + span - 7, bar_y - 7, bar_x0 + span + 7, bar_y + 7], fill=ACCENT)

    thumbs = sorted((cache_dir / "thumbs").glob("*.png"))
    if thumbs:
        thumb_h = 46
        thumb_w = min(74, (bar_x1 - bar_x0 - 4 * len(thumbs)) // max(1, len(thumbs)))
        x = bar_x0
        for index, path in enumerate(thumbs):
            image = Image.open(path).convert("RGB")
            image = image.resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)
            y = timeline_y + 46
            board.paste(image, (x, y))
            border = ACCENT if index == frame_t else BORDER
            draw.rectangle([x - 1, y - 1, x + thumb_w, y + thumb_h], outline=border,
                           width=2 if index == frame_t else 1)
            x += thumb_w + 4

    return board


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="マルチパネル可視化（LAYOUT_SPEC 実装）の素材生成と合成")
    parser.add_argument("--scene", type=Path, required=True, help="保存済みシーン (.npz)")
    parser.add_argument("--canvas-frames", type=Path, help="3D キャンバス画像フォルダ（初回キャッシュ構築時）")
    parser.add_argument("--cache-dir", type=Path, help="キャッシュ先（既定: results/dashboard_cache/…）")
    parser.add_argument("--pick", type=int, default=56, help="識別色を付ける追跡点の数（既定 56）")
    parser.add_argument("--rebuild-materials", action="store_true", help="素材を再生成する")
    parser.add_argument("--alignment", type=Path,
                        help="align_clip_scenes.py の alignments.json（クリップ結合の座標系接続を適用）")
    parser.add_argument("--output", type=Path, help="統合動画 mp4 を出力する（全フレーム合成 → エンコード）")
    parser.add_argument("--fps", type=int, default=30, help="動画 fps（既定 30。仕様の原則）")
    parser.add_argument("--hold", type=int, default=1,
                        help="各フレームを動画内で保持する枚数（30fps のまま尺を伸ばせる。既定 1）")
    parser.add_argument("--summary-frame", type=int, default=None, help="静止画サマリーのフレーム（既定: 中央）")
    parser.add_argument("--summary-output", type=Path, help="静止画サマリー PNG の出力先")
    parser.add_argument("--frame-output", type=Path, help="単一フレーム PNG の出力先（確認用）")
    parser.add_argument("--config-output", type=Path, help="可視化設定 JSON の出力先")
    parser.add_argument("--frames-dir", type=Path,
                        help="合成連番の出力先（既定: results/video_concept/build/multipanel_frames）")
    args = parser.parse_args(argv)
    for name in ("scene", "canvas_frames", "cache_dir", "output", "summary_output",
                 "frame_output", "config_output", "frames_dir", "alignment"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if not args.scene.is_file():
        parser.error(f"シーンが見つかりません: {args.scene}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cache_dir = args.cache_dir or default_cache_dir(args.scene)
    canvas_frames = args.canvas_frames if args.canvas_frames and args.canvas_frames.is_dir() else None

    started = time.perf_counter()
    result, manifest, hit = ensure_cache(args.scene, cache_dir, args.pick, canvas_frames, False)
    log(f"キャッシュ: {'ヒット' if hit else '構築'}（{time.perf_counter() - started:.1f} 秒）— {cache_dir}")

    alignment = None
    if args.alignment:
        data = json.loads(args.alignment.read_text(encoding="utf-8"))
        entry = next((item for item in data["clips"] if item["name"] == args.scene.name), None)
        if entry is None:
            raise SystemExit(f"alignment に {args.scene.name} がありません: {args.alignment}")
        alignment = {
            "scale": float(entry["scale"]),
            "rotation": np.asarray(entry["rotation"], dtype=np.float32),
            "translation": np.asarray(entry["translation"], dtype=np.float32),
            "depth_range_world": data.get("depth_range_world"),
            "world_bounds": data.get("world_bounds"),
        }
        log(f"alignment を適用: scale {alignment['scale']:.4f}（{entry['name']}）")

    started = time.perf_counter()
    manifest.update(build_materials(result, cache_dir, args.pick, rebuild=args.rebuild_materials,
                                    alignment=alignment))
    log(f"2D 素材生成（{time.perf_counter() - started:.1f} 秒）")

    started = time.perf_counter()
    build_3d_materials(cache_dir, rebuild=args.rebuild_materials, alignment=alignment)
    log(f"3D 素材生成（{time.perf_counter() - started:.1f} 秒）")

    materials = manifest["materials"]
    materials["panels"] = ["overlay", "depth", "motion", "confvis", "normal", "mask",
                           "scene3d", "cam3d", "tracks3d", "summary"]
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")

    frames = int(manifest["scene"]["frames"])
    representative = args.summary_frame if args.summary_frame is not None else frames // 2
    representative = max(0, min(frames - 1, representative))

    if args.config_output:
        settings = {
            "layout": "LAYOUT_SPEC (1920x1080 / ヘッダー62・メイン726・タイムライン108・フロー108)",
            "view3d": VIEW_3D,
            "layers": materials["panels"],
            "trail": {"length": "全履歴", "alpha_past": 0.35, "current_marker": "white-edge"},
            "density": {"pick": args.pick, "cloud_points": len(np.load(cache_dir / "series" / "cloud.npz")["points"])},
            "color_range": {"depth": manifest["depth_range"],
                            "confidence": materials["confidence_normalization"]},
            "sync": {"frame_based": True},
        }
        args.config_output.parent.mkdir(parents=True, exist_ok=True)
        args.config_output.write_text(json.dumps(settings, ensure_ascii=False, indent=1) + "\n")
        log(f"可視化設定を保存: {args.config_output}")

    if args.frame_output or args.summary_output:
        board = compose_frame(cache_dir, representative)
        for target in (args.frame_output, args.summary_output):
            if target:
                target.parent.mkdir(parents=True, exist_ok=True)
                board.save(target)
                log(f"静止画を保存: {target}（frame {representative}）")

    if args.output:
        frames_dir = args.frames_dir or (args.output.parent / "build" / "multipanel_frames")
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True)
        started = time.perf_counter()
        index = 0
        for frame_t in range(frames):
            board = compose_frame(cache_dir, frame_t)
            for _ in range(max(1, args.hold)):
                board.save(frames_dir / f"frame_{index:04d}.png")
                index += 1
            if (frame_t + 1) % 6 == 0:
                log(f"  … {frame_t + 1}/{frames} 枚")
        compose_seconds = time.perf_counter() - started
        if shutil.which("ffmpeg") is None:
            raise SystemExit("ffmpeg が見つかりません")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-framerate", str(args.fps),
                "-i", str(frames_dir / "frame_%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
                str(args.output),
            ],
            check=True,
        )
        log(f"合成 {compose_seconds:.1f} 秒 / 動画を保存: {args.output}"
            f"（{index} 枚 / {args.fps} fps ≒ {index / args.fps:.1f} 秒、"
            f"{args.output.stat().st_size / 2**20:.1f} MB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
