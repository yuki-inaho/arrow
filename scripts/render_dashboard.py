#!/usr/bin/env python3
"""固定レイアウト（複数パネル同時表示）の可視化動画を合成する（中間出力物キャッシュ対応）。

レイアウト（正: results/video_concept/layout_dashboard.png）:
- 上部: 短いタイトル（1 行のみ）
- 左上: 元画像 + 2D 追跡（色分けした点と軌跡）
- 左下: 相対深度のカラーマップ（色範囲はクリップ全体で固定）
- 右  : 3D 再構成 + 追跡（viewer_capture.py --scrub が保存したキャンバス画像）
- 下部: 全パネル共通のタイムライン + 統計バッジ

## 中間出力物キャッシュ（--cache-dir、既定: results/dashboard_cache/<シーン名>-<hash8>）

    manifest.json              来歴・params・統計・パネル/系列の一覧
    panels/NNNN_overlay.png    元画像 + 2D 追跡の描画（pick 点・色分け軌跡）
    panels/NNNN_depth.png      相対深度カラーマップ（色範囲はクリップで固定）
    panels/NNNN_canvas.png     3D キャンバス（viewer_capture.py --scrub の出力を取り込み）
    series/tracks.npz          seed_idx / colors / xyz[T,P,3]（rerun 出力と共用）
    series/cloud.npz           points[N,3] / rgb[N,3]（rerun 用の再構成点群）

- **無効化条件**: 入力 npz の bytes・mtime 不一致 / `params.pick` 不一致。
- **レイアウトや出力形式の変更はキャッシュに影響しない**（合成・エンコードだけをやり直すため高速）。
- 2 回目以降の実行はキャッシュヒットとなり、パネル描画とキャンバス取り込みを省略できる。

使い方::

    # 1 回目（キャッシュ構築 + 合成）
    uv run python scripts/render_dashboard.py --scene results/demo_seq/railway_seq.npz \
        --canvas-frames results/video_concept/build/frames_scrub \
        --output results/video_concept/draft_dashboard.mp4

    # 2 回目以降（キャッシュヒット。--canvas-frames は不要）
    uv run python scripts/render_dashboard.py --scene results/demo_seq/railway_seq.npz \
        --output results/video_concept/draft_dashboard.mp4
"""

import argparse
import colorsys
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]

BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
BG = (18, 20, 26)
PANEL_BORDER = (52, 58, 68)
LABEL = (154, 163, 173)
WHITE = (235, 238, 242)
ACCENT = (90, 160, 255)
BADGE_TEXT = (10, 12, 16)

TITLE = "ARROW｜連番画像から 2D・3D を同時に再構成"
PANELS = {
    "left_top": (28, 64, 714, 530),
    "left_bottom": (28, 546, 714, 1012),
    "right": (730, 64, 1892, 1012),
}
CANVAS = (1920, 1080)
DEPTH_PERCENTILES = (2, 98)


def font(size: int):
    return ImageFont.truetype(BOLD, size)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- シーンからの描画（キャッシュ構築で使用） ---------------------------------------


def group_slice(result, task: str):
    entry = next(g for g in result.metadata["groups"] if g["task"] == task)
    start, count = entry["start"], entry["count"]
    return (
        entry,
        result.predictions.points[0, start : start + count],
        result.predictions.uv[0, start : start + count],
        result.queries.xy[0, start : start + count],
    )


def colors_for(count: int):
    return [
        tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / count, 0.8, 1.0)) for i in range(count)
    ]


def track_picks(seeds: int, pick: int) -> np.ndarray:
    return np.linspace(0, seeds - 1, min(pick, seeds)).astype(int)


def render_2d_overlay(result, frame_t: int, width: int, height: int, pick: int) -> Image.Image:
    """元画像 + 色分けされた追跡点と軌跡（現在フレームまで）。"""
    frame = (result.images[0, frame_t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)

    entry, _, uv, _ = group_slice(result, "track")
    seeds = entry["points_per_frame"]
    picks = track_picks(seeds, pick)
    colors = colors_for(len(picks))
    for k, seed in enumerate(picks):
        trail = []
        for target in range(0, frame_t + 1):
            value = uv[target * seeds + seed]
            if np.isfinite(np.asarray(value)).all():
                trail.append((float(value[0]) * width - 0.5, float(value[1]) * height - 0.5))
        if len(trail) >= 2:
            draw.line(trail, fill=colors[k], width=2)
        if trail:
            x, y = trail[-1]
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=colors[k])
    return image


def depth_grid(result) -> tuple[np.ndarray, float, float, int]:
    """深度グループを (frames, rows, cols) に整形し、色範囲（クリップ固定）を返す。"""
    entry, points, _, xy = group_slice(result, "depth")
    frames = len(result.metadata["frame_info"])
    per_frame = entry["count"] // frames
    first = xy[:per_frame].numpy()
    cols, rows = len(np.unique(first[:, 0])), len(np.unique(first[:, 1]))
    z = points[..., 2].numpy().reshape(frames, rows, cols)
    finite = z[np.isfinite(z)]
    low, high = np.percentile(finite, list(DEPTH_PERCENTILES))
    return z, float(low), float(high), frames


def render_depth(z: np.ndarray, frame_t: int, low: float, high: float,
                 width: int, height: int) -> Image.Image:
    from matplotlib import colormaps

    normalized = np.clip((z[frame_t] - low) / max(1e-6, high - low), 0, 1)
    normalized[~np.isfinite(z[frame_t])] = 1.0
    rgb = (colormaps["turbo"](normalized)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb).resize((width, height), Image.Resampling.BICUBIC)


# --- 中間出力物キャッシュ（スキーマ v1） ---------------------------------------------


def scene_key(scene_path: Path) -> str:
    digest = hashlib.sha1(str(scene_path).encode()).hexdigest()[:8]
    return f"{scene_path.stem}-{digest}"


def default_cache_dir(scene_path: Path) -> Path:
    return REPO / "results" / "dashboard_cache" / scene_key(scene_path)


def source_fingerprint(scene_path: Path) -> dict:
    stat = scene_path.stat()
    try:
        display = str(scene_path.relative_to(REPO))
    except ValueError:
        display = str(scene_path)
    return {"npz": display, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def cache_is_valid(manifest: dict, scene_path: Path, pick: int) -> bool:
    return (
        manifest.get("source") == source_fingerprint(scene_path)
        and manifest.get("params", {}).get("pick") == pick
    )


def build_series(result, series_dir: Path, pick: int) -> dict:
    """rerun 出力と共用する数値系列（3D 軌跡と再構成点群）を保存する。"""
    series_dir.mkdir(parents=True, exist_ok=True)
    entry, points, _, _ = group_slice(result, "track")
    seeds = entry["points_per_frame"]
    targets = len(entry["target_frames"])
    picks = track_picks(seeds, pick)
    xyz = points[: targets * seeds].numpy().reshape(targets, seeds, 3)[:, picks, :]
    colors = np.array(colors_for(len(picks)), dtype=np.uint8)
    np.savez_compressed(
        series_dir / "tracks.npz",
        seed_idx=picks.astype(np.int64), colors=colors, xyz=xyz.astype(np.float32),
    )

    frames = len(result.metadata["frame_info"])
    entry, points, _, xy = group_slice(result, "reconstruct")
    per_frame = entry["count"] // frames
    xy = xy.numpy().reshape(frames, per_frame, 2)
    pts = points.numpy().reshape(frames, per_frame, 3)
    frame_rgb = (result.images[0].permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    take = slice(0, per_frame, max(1, per_frame // 9000))
    cloud, rgb = [], []
    for f in range(frames):
        select_xy = xy[f][take]
        xs = np.clip(np.round(select_xy[:, 0]).astype(int), 0, result.images.shape[-1] - 1)
        ys = np.clip(np.round(select_xy[:, 1]).astype(int), 0, result.images.shape[-2] - 1)
        cloud.append(pts[f][take])
        rgb.append(frame_rgb[f][ys, xs])
    cloud, rgb = np.concatenate(cloud), np.concatenate(rgb)
    keep = np.isfinite(cloud).all(-1)
    np.savez_compressed(
        series_dir / "cloud.npz", points=cloud[keep].astype(np.float32), rgb=rgb[keep]
    )
    return {"tracks": "series/tracks.npz", "cloud": "series/cloud.npz"}


def ensure_cache(scene_path: Path, cache_dir: Path, pick: int,
                 canvas_frames: Path | None, rebuild: bool) -> tuple[object, dict, bool]:
    """キャッシュを検証し、必要なら構築する。(result, manifest, ヒットしたか) を返す。"""
    import json

    from arrow.utils.results_io import load_result

    result = load_result(scene_path)
    frames = len(result.metadata["frame_info"])
    height, width = result.images.shape[-2:]
    manifest_path = cache_dir / "manifest.json"

    cached = None
    if manifest_path.is_file() and not rebuild:
        try:
            candidate = json.loads(manifest_path.read_text())
        except ValueError:
            candidate = None
        if candidate and cache_is_valid(candidate, scene_path, pick):
            cached = candidate

    # キャンバス（3D パネル）の取り込み。既存キャッシュでは不足分のみ補う。
    if canvas_frames is not None:
        canvas_files = sorted(canvas_frames.glob("frame_*.png"))
        if len(canvas_files) < frames:
            raise SystemExit(
                f"3D キャンバスが足りません（{len(canvas_files)}/{frames} 枚）: {canvas_frames}"
            )
        (cache_dir / "panels").mkdir(parents=True, exist_ok=True)
        for index in range(frames):
            target = cache_dir / "panels" / f"{index:04d}_canvas.png"
            source = canvas_files[index]
            if rebuild or not target.is_file() or target.stat().st_mtime_ns < source.stat().st_mtime_ns:
                shutil.copy2(source, target)

    if cached is not None:
        # 系列ファイルが欠けていれば再構築（パネルはそのまま）
        series_ok = all((cache_dir / path).is_file() for path in cached["series"].values())
        if not series_ok:
            cached["series"] = build_series(result, cache_dir / "series", pick)
            manifest_path.write_text(json.dumps(cached, ensure_ascii=False, indent=1) + "\n")
        return result, cached, True

    # --- 構築 ---
    panels_dir = cache_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    z, low, high, _ = depth_grid(result)
    for frame_t in range(frames):
        overlay = render_2d_overlay(result, frame_t, width, height, pick)
        overlay.save(panels_dir / f"{frame_t:04d}_overlay.png")
        depth = render_depth(z, frame_t, low, high, width, height)
        depth.save(panels_dir / f"{frame_t:04d}_depth.png")

    series = build_series(result, cache_dir / "series", pick)
    track_entry, _, _, _ = group_slice(result, "track")
    _, reconstruct_points, _, _ = group_slice(result, "reconstruct")
    finite_points = int(np.isfinite(reconstruct_points.numpy()).all(-1).sum())

    manifest = {
        "created_at": utc_now(),
        "source": source_fingerprint(scene_path),
        "params": {"pick": pick, "depth_percentiles": list(DEPTH_PERCENTILES)},
        "scene": {"frames": frames, "width": width, "height": height},
        "stats": {
            "tracks": track_entry["points_per_frame"],
            "points": finite_points,
            "npz_bytes": scene_path.stat().st_size,
        },
        "depth_range": {"low": low, "high": high},
        "panels_pattern": "panels/{index:04d}_{panel}.png",
        "panels": ["overlay", "depth", "canvas"],
        "series": series,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    return result, manifest, False


# --- 合成（レイアウト。キャッシュのパネル画像だけを使う） -------------------------------


def fit(image: Image.Image, box_w: int, box_h: int) -> Image.Image:
    scale = min(box_w / image.width, box_h / image.height)
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)


def compose(canvas_3d: Image.Image, overlay: Image.Image, depth: Image.Image,
            frame_t: int, frames: int, badges: list[str]) -> Image.Image:
    """1 フレームぶんの複合画面を組み立てる。"""
    board = Image.new("RGB", CANVAS, BG)
    draw = ImageDraw.Draw(board)

    draw.text((28, 16), TITLE, font=font(26), fill=WHITE)
    for box in PANELS.values():
        draw.rounded_rectangle(box, radius=10, outline=PANEL_BORDER, width=2)

    top, bottom, right = PANELS["left_top"], PANELS["left_bottom"], PANELS["right"]
    fitted = fit(overlay, 662, 404)
    board.paste(fitted, (top[0] + (662 + 12 - fitted.width) // 2,
                         top[1] + (404 + 42 - fitted.height) // 2 + 6))
    draw.text((top[0] + 14, top[1] + 10), "入力 + 2D 追跡", font=font(20), fill=LABEL)

    fitted = fit(depth, 662, 404)
    board.paste(fitted, (bottom[0] + (662 + 12 - fitted.width) // 2,
                         bottom[1] + (404 + 42 - fitted.height) // 2 + 6))
    draw.text((bottom[0] + 14, bottom[1] + 10), "相対深度（色範囲はクリップで固定）",
              font=font(20), fill=LABEL)

    fitted = fit(canvas_3d, right[2] - right[0] - 12, right[3] - right[1] - 42)
    board.paste(fitted, (right[0] + 6 + (right[2] - right[0] - 12 - fitted.width) // 2,
                         right[1] + 36 + (right[3] - right[1] - 42 - fitted.height) // 2))
    draw.text((right[0] + 14, right[1] + 10), "3D 再構成 + 追跡", font=font(20), fill=LABEL)

    line_y = 1044
    draw.text((28, 1016), f"Frame {frame_t + 1}/{frames}", font=font(22), fill=LABEL)
    x0, x1 = 210, 1400
    draw.line([x0, line_y, x1, line_y], fill=(70, 76, 86), width=3)
    step = (x1 - x0) / max(1, frames - 1)
    for f in range(frames):
        x = x0 + step * f
        draw.line([x, line_y - 7, x, line_y + 7], fill=ACCENT if f <= frame_t else (70, 76, 86), width=3)
    current = x0 + step * frame_t
    draw.ellipse([current - 9, line_y - 9, current + 9, line_y + 9], fill=ACCENT)

    x = 1888
    for text in reversed(badges):
        width_px = draw.textlength(text, font=font(20)) + 24
        draw.rounded_rectangle([x - width_px, 1018, x, 1052], radius=8, fill=ACCENT)
        draw.text((x - width_px + 12, 1023), text, font=font(20), fill=BADGE_TEXT)
        x -= width_px + 14
    return board


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="固定レイアウトの可視化動画を合成する（キャッシュ対応）")
    parser.add_argument("--scene", type=Path, required=True, help="保存済みシーン (.npz)")
    parser.add_argument("--canvas-frames", type=Path,
                        help="3D キャンバス画像フォルダ（viewer_capture.py --scrub の出力。"
                             "キャッシュ未構築時に必要）")
    parser.add_argument("--cache-dir", type=Path, help="中間出力物のキャッシュ先（既定: results/dashboard_cache/…）")
    parser.add_argument("--rebuild-cache", action="store_true", help="キャッシュを作り直す")
    parser.add_argument("--output", type=Path, required=True, help="出力 mp4")
    parser.add_argument("--fps", type=int, default=12, help="出力 fps（既定 12）")
    parser.add_argument("--pick", type=int, default=56, help="2D に重ねる追跡点の数")
    parser.add_argument("--clean-frames", action="store_true", help="合成後、連番 PNG を削除する")
    args = parser.parse_args(argv)
    args.scene = args.scene.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.cache_dir is not None:
        args.cache_dir = args.cache_dir.expanduser().resolve()
    if args.canvas_frames is not None:
        args.canvas_frames = args.canvas_frames.expanduser().resolve()
    if not args.scene.is_file():
        parser.error(f"シーンが見つかりません: {args.scene}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cache_dir = args.cache_dir or default_cache_dir(args.scene)
    canvas_frames = args.canvas_frames if args.canvas_frames and args.canvas_frames.is_dir() else None

    cache_started = time.perf_counter()
    result, manifest, hit = ensure_cache(args.scene, cache_dir, args.pick, canvas_frames,
                                         args.rebuild_cache)
    cache_seconds = time.perf_counter() - cache_started
    frames = manifest["scene"]["frames"]

    pattern = manifest["panels_pattern"]
    badges = [
        f"{manifest['stats']['tracks']:,} トラック",
        f"{manifest['stats']['points']:,} 点",
        f"NPZ {manifest['stats']['npz_bytes'] / 2**20:.0f} MB",
    ]
    print(f"キャッシュ: {'ヒット' if hit else '構築'}（{cache_seconds:.1f} 秒）— {cache_dir}")
    print(f"合成開始: {frames} フレーム / {manifest['scene']['width']}x{manifest['scene']['height']}")

    frames_dir = args.output.parent / "build" / "dashboard_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)

    compose_started = time.perf_counter()
    for frame_t in range(frames):
        overlay = Image.open(cache_dir / pattern.format(index=frame_t, panel="overlay"))
        depth = Image.open(cache_dir / pattern.format(index=frame_t, panel="depth"))
        canvas_path = cache_dir / pattern.format(index=frame_t, panel="canvas")
        if not canvas_path.is_file():
            raise SystemExit(
                f"3D キャンバスがキャッシュにありません: {canvas_path}\n"
                "--canvas-frames を指定して 1 度キャッシュを構築してください。"
            )
        canvas_3d = Image.open(canvas_path)
        board = compose(canvas_3d, overlay, depth, frame_t, frames, badges)
        board.save(frames_dir / f"frame_{frame_t:04d}.png")
        if (frame_t + 1) % 6 == 0:
            print(f"  … {frame_t + 1}/{frames} 枚")
    compose_seconds = time.perf_counter() - compose_started

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
    duration = frames / args.fps
    print(f"合成 {compose_seconds:.1f} 秒 / 合計 {cache_seconds + compose_seconds:.1f} 秒"
          f"（エンコード除く）")
    print(f"動画を保存: {args.output}（{frames} 枚 / {args.fps} fps ≒ {duration:.1f} 秒、"
          f"{args.output.stat().st_size / 2**20:.1f} MB）")
    print(f"合成連番: {frames_dir}")
    if args.clean_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
