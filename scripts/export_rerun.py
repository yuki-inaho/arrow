#!/usr/bin/env python3
"""中間出力物キャッシュ（render_dashboard.py・render_multipanel.py と共用）から
Rerun 記録（.rrd）を書き出す。

動画（マルチパネル）と同じ情報・同じ向き（上が本当の上）・近い観察視点で、
インタラクティブに確認できる記録を作る:

- 3D ビュー（主画面）: 点群（RGB・細かい点）、識別色の全軌跡、カメラ視錐台
  （全フレームは薄青、現在フレームは明青で強調）、カメラ位置の軌跡、XYZ 軸
- 2D ビュー（6 面）: 元画像+2D追跡 / 2D移動 / 奥行き / 表面法線 / 解析対象領域 /
  信頼度・可視性
- 表示座標は (x, z, -y)（Z 上）に変換。動画と同じ「上が本当の上」の向き。
- 2D 画像は JPEG に再エンコードして記録（ファイルサイズ削減）。

使い方::

    uv run python scripts/export_rerun.py --scene results/demo_seq/railway_seq.npz \
        --output results/video_concept/arrow_railway.rrd

    # ローカルで開く場合
    rerun results/video_concept/arrow_railway.rrd
"""

import argparse
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from render_dashboard import default_cache_dir, ensure_cache  # noqa: E402
from render_multipanel import FRUSTUM_SCALE, frustum_segments  # noqa: E402

# 2D ビュー（表示パス → キャッシュ素材名）。動画の第1・第2列と同じ 6 面。
PANEL_VIEWS = {
    "view/input_2d_tracks": "overlay",
    "view/motion": "motion",
    "view/depth": "depth",
    "view/normal": "normal",
    "view/mask": "mask",
    "view/confvis": "confvis",
}

BACKGROUND = [15, 17, 22, 255]          # 動画と同じ暗色基調
CAMERA_DIM = (85, 115, 165, 255)        # 過去フレームの視錐台（薄青）
CAMERA_CURRENT = (90, 160, 255, 255)    # 現在フレームの視錐台（強調）
CAMERA_PATH = (120, 160, 220, 255)      # カメラ位置の軌跡
AXIS_COLORS = [[255, 107, 107], [123, 216, 143], [107, 155, 255]]  # X=赤, Y=緑, Z=青


def to_display(points: np.ndarray) -> np.ndarray:
    """参照カメラ座標 (x 右, y 下, z 前) を表示用 (x, z, -y)（Z 上）に変換する。"""
    source = np.asarray(points, dtype=np.float32)
    out = np.empty_like(source)
    out[..., 0] = source[..., 0]
    out[..., 1] = source[..., 2]
    out[..., 2] = -source[..., 1]
    return out


def _jpeg_bytes(path: Path) -> bytes:
    buffer = BytesIO()
    Image.open(path).convert("RGB").save(buffer, format="JPEG", quality=88)
    return buffer.getvalue()


def estimate_view(cache_dir: Path, alignment: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """観察視点（eye 位置, 注視点）を推定する（表示座標・Z 上）。

    alignment に world_bounds があれば全クリップ共通の範囲から求める
    （クリップ結合時に視点が変わらないように）。
    """
    world = alignment.get("world_bounds") if alignment else None
    if world:
        # 新形式（points/cameras/overall）と旧形式（low/high）の両対応
        points = world.get("points", world)
        low = np.asarray(points["low"], dtype=np.float32)
        high = np.asarray(points["high"], dtype=np.float32)
        low_display = np.array([low[0], low[2], -high[1]], dtype=np.float32)
        high_display = np.array([high[0], high[2], -low[1]], dtype=np.float32)
        center = (low_display + high_display) / 2
        span = float(np.max(high_display - low_display))
    else:
        cloud = np.load(cache_dir / "series" / "cloud.npz")["points"]
        sample = to_display(cloud[:: max(1, len(cloud) // 20000)])
        low, high = sample.min(0), sample.max(0)
        center = (low + high) / 2
        span = float(np.max(high - low))
    eye = center + np.array([-0.3, -2.0, 1.0], dtype=np.float32) * span
    return eye, center


def load_alignment_entry(data: dict | None, name: str) -> dict | None:
    """alignments.json からクリップ名に対応する変換（スケール・回転・並進）を取り出す。"""
    if data is None:
        return None
    entry = next((item for item in data["clips"] if item["name"] == name), None)
    if entry is None:
        return None
    return {
        "scale": float(entry["scale"]),
        "rotation": np.asarray(entry["rotation"], dtype=np.float32),
        "translation": np.asarray(entry["translation"], dtype=np.float32),
        "world_bounds": data.get("world_bounds"),
    }


def send_blueprint(eye_position: np.ndarray, look_target: np.ndarray) -> None:
    """ビューアの初期レイアウト（3D 主画面 + 2D 6 面 + 暗色スタイル + 観察視点）。"""
    import rerun as rr
    import rerun.blueprint as rrb

    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin="/world",
                    name="3D 再構成・追跡",
                    background=BACKGROUND,
                    eye_controls=rrb.EyeControls3D(
                        kind="orbital",
                        position=[float(v) for v in eye_position],
                        look_target=[float(v) for v in look_target],
                        eye_up=[0.0, 0.0, 1.0],
                    ),
                ),
                rrb.Vertical(
                    rrb.Horizontal(
                        rrb.Spatial2DView(origin="view/input_2d_tracks", name="元画像 + 2D追跡",
                                          background=BACKGROUND),
                        rrb.Spatial2DView(origin="view/motion", name="2D移動",
                                          background=BACKGROUND),
                    ),
                    rrb.Horizontal(
                        rrb.Spatial2DView(origin="view/depth", name="奥行き",
                                          background=BACKGROUND),
                        rrb.Spatial2DView(origin="view/normal", name="表面法線",
                                          background=BACKGROUND),
                    ),
                    rrb.Horizontal(
                        rrb.Spatial2DView(origin="view/mask", name="解析対象領域",
                                          background=BACKGROUND),
                        rrb.Spatial2DView(origin="view/confvis", name="信頼度・可視性",
                                          background=BACKGROUND),
                    ),
                ),
                column_shares=[2, 1],
            ),
        )
    )


def log_scene(cache_dir: Path, manifest: dict, *, frame_offset: int = 0, log_static: bool = True,
              alignment: dict | None = None, skip_frames: int = 0) -> int:
    """キャッシュ済みシーンを現在の rerun セッションへログする（保存はしない）。

    log_static は点群・全軌跡・全カメラ視錐台（static=True）を送るかどうか。
    クリップ結合では最初のクリップだけ True にすると、全体で 1 回の静的記録になる。
    alignment 指定時は点群・軌跡・カメラを世界座標へ変換する（クリップ間の連続性）。
    skip_frames は先頭からログしないフレーム数（クリップ重複区間の除去用）。
    フレーム軸は `frame_offset` を起点にした連番。戻り値はログしたフレーム数。
    """
    import rerun as rr

    cached_tracks = np.load(cache_dir / manifest["series"]["tracks"])
    cached_cloud = np.load(cache_dir / manifest["series"]["cloud"])
    xyz = cached_tracks["xyz"]           # [T, P, 3]
    colors = cached_tracks["colors"]     # [P, 3] uint8
    cloud_points = cached_cloud["points"]
    cloud_rgb = cached_cloud["rgb"]

    cameras_path = cache_dir / "series" / "cameras.npz"
    cameras = dict(np.load(cameras_path)) if cameras_path.is_file() else None

    # 世界座標への変換（クリップ間の位置・向き・スケールの連続性）
    frustum_scale = FRUSTUM_SCALE
    if alignment is not None:
        scale = float(alignment["scale"])
        rotation = np.asarray(alignment["rotation"], dtype=np.float32)
        translation = np.asarray(alignment["translation"], dtype=np.float32)
        cloud_points = scale * (cloud_points @ rotation.T) + translation
        xyz = scale * (xyz @ rotation.T) + translation
        if cameras is not None:
            matrices = cameras["camera_to_reference"].astype(np.float32)
            new_rotation = np.einsum("ij,fjk->fik", rotation, matrices[:, :3, :3])
            new_translation = scale * (matrices[:, :3, 3] @ rotation.T) + translation
            cameras["camera_to_reference"] = np.concatenate(
                [np.concatenate([new_rotation, new_translation[..., None]], axis=2),
                 np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (len(matrices), 1, 1))],
                axis=1,
            )
        world = alignment.get("world_bounds")
        if world:
            points = world.get("points", world)
            low = np.asarray(points["low"], dtype=np.float32)
            high = np.asarray(points["high"], dtype=np.float32)
            frustum_scale = float(np.max(high - low)) * 0.05

    cloud_display = to_display(cloud_points)

    # 静的記録（点群・全軌跡・全カメラ・軸）。static=True を付けないと
    # log_time 軸のみに属し、"frame" 軸表示時に隠れる。
    if log_static:
        rr.log(
            "world/reconstruction",
            rr.Points3D(cloud_display, colors=cloud_rgb, radii=0.025),
            static=True,
        )
        strips = []
        strip_colors = []
        for pick_index in range(xyz.shape[1]):
            line = xyz[:, pick_index, :]
            keep = np.isfinite(line).all(-1)
            if keep.sum() >= 2:
                strips.append(to_display(line[keep]))
                strip_colors.append(colors[pick_index])
        rr.log(
            "world/tracks",
            rr.LineStrips3D(strips, colors=np.array(strip_colors, dtype=np.uint8), radii=0.02),
            static=True,
        )
        if cameras is not None:
            matrices = cameras["camera_to_reference"]
            frusta = []
            for frame in range(len(matrices)):
                if cameras["valid"][frame]:
                    frusta.extend(frustum_segments(matrices[frame], frustum_scale))
            if frusta:
                rr.log(
                    "world/cameras",
                    rr.LineStrips3D(frusta, colors=[CAMERA_DIM] * len(frusta), radii=0.02),
                    static=True,
                )
            centers = to_display(matrices[:, :3, 3])
            rr.log(
                "world/camera_path",
                rr.LineStrips3D([centers], colors=[CAMERA_PATH], radii=0.02),
                static=True,
            )
        span = float(np.max(np.ptp(cloud_display, axis=0)))
        length = 0.05 * span
        origin = np.zeros(3, dtype=np.float32)
        rr.log(
            "world/axis",
            rr.LineStrips3D(
                [np.stack([origin, [length, 0, 0]]),
                 np.stack([origin, [0, length, 0]]),
                 np.stack([origin, [0, 0, length]])],
                colors=AXIS_COLORS,
                radii=0.02,
            ),
            static=True,
        )

    # 時間軸に沿った記録: 2D 6 面 + 追跡点マーカー + 現在カメラの視錐台
    pattern = manifest["panels_pattern"]
    frames = manifest["scene"]["frames"]
    for frame_t in range(skip_frames, frames):
        rr.set_time("frame", sequence=frame_offset + frame_t)
        for view_path, panel in PANEL_VIEWS.items():
            image_path = cache_dir / pattern.format(index=frame_t, panel=panel)
            if image_path.is_file():
                rr.log(view_path, rr.EncodedImage(contents=_jpeg_bytes(image_path),
                                                  media_type="image/jpeg"))
        heads = xyz[frame_t]
        keep = np.isfinite(heads).all(-1)
        if keep.any():
            rr.log(
                "world/track_heads",
                rr.Points3D(to_display(heads[keep]), colors=colors[keep], radii=0.045),
            )
        if cameras is not None and cameras["valid"][frame_t]:
            segments = frustum_segments(cameras["camera_to_reference"][frame_t], frustum_scale)
            rr.log(
                "world/current_camera",
                rr.LineStrips3D(segments, colors=[CAMERA_CURRENT] * len(segments), radii=0.03),
            )
    return frames


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="キャッシュから Rerun 記録を出力する")
    parser.add_argument("--scene", type=Path, required=True, help="保存済みシーン (.npz)")
    parser.add_argument("--output", type=Path, required=True, help="出力 .rrd")
    parser.add_argument("--cache-dir", type=Path, help="キャッシュ先（既定: results/dashboard_cache/…）")
    parser.add_argument("--pick", type=int, default=56, help="軌跡の本数（既定 56。キャッシュと一致させる）")
    parser.add_argument("--frame-offset", type=int, default=0,
                        help="フレームタイムラインの開始番号（クリップ結合時に連番化する。既定 0）")
    parser.add_argument("--recording-id",
                        help="recording ID（同一 recording へ複数シーンを書き足すときに指定する）")
    parser.add_argument("--alignment", type=Path,
                        help="align_clip_scenes.py の alignments.json（世界座標接続を適用）")
    args = parser.parse_args(argv)
    args.scene = args.scene.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.cache_dir is not None:
        args.cache_dir = args.cache_dir.expanduser().resolve()
    if args.alignment is not None:
        args.alignment = args.alignment.expanduser().resolve()
    if not args.scene.is_file():
        parser.error(f"シーンが見つかりません: {args.scene}")
    return args


def main(argv: list[str] | None = None) -> int:
    import rerun as rr

    args = parse_args(argv)
    cache_dir = args.cache_dir or default_cache_dir(args.scene)
    _, manifest, hit = ensure_cache(args.scene, cache_dir, args.pick, None, rebuild=False)
    print(f"キャッシュ: {'ヒット' if hit else '構築'} — {cache_dir}")

    rr.init(f"arrow_{args.scene.stem}", recording_id=args.recording_id, spawn=False)
    if args.recording_id:
        print(f"recording id を共通化: {args.recording_id}")
    alignment = None
    if args.alignment:
        data = json.loads(args.alignment.read_text(encoding="utf-8"))
        alignment = load_alignment_entry(data, args.scene.name)
        if alignment is None:
            raise SystemExit(f"alignment に {args.scene.name} がありません: {args.alignment}")
        print(f"alignment を適用: scale {alignment['scale']:.4f}")
    eye, target = estimate_view(cache_dir, alignment)
    send_blueprint(eye, target)
    frames = log_scene(cache_dir, manifest, frame_offset=args.frame_offset, log_static=True,
                       alignment=alignment)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(args.output))
    size_mb = args.output.stat().st_size / 2**20
    offset_note = f"、frame {args.frame_offset}–{args.frame_offset + frames - 1}" if args.frame_offset else ""
    print(f"rerun 記録を保存: {args.output}（{size_mb:.1f} MB、{frames} フレーム分{offset_note}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
