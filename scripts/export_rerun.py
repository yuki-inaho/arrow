#!/usr/bin/env python3
"""中間出力物キャッシュ（render_dashboard.py と共用）から Rerun 記録（.rrd）を書き出す。

- 入力: 保存済みシーン (.npz)。キャッシュが無ければ render_dashboard.py と同じ層で構築する
  （3D キャンバス PNG は rerun 出力では不要）。
- 出力: .rrd（rerun-sdk 0.39）。記録内容:
  - 3D 再構成点群（RGB 色付け・静的）
  - 3D 軌跡（2D パネルと同じ色のラインストリップ・静的）
  - フレームごと: 入力+2D追跡の画像、深度カラーマップ、トラック先頭点（時間軸 "frame"）
- 表示座標は (x, -y, z) に変換（見上げ方向をそろえる）。
- クリップ結合（merge_clip_outputs.py）からも `send_blueprint` / `log_scene` を再利用し、
  1 セッションへ連番フレームでログして単一レコーディングの .rrd を作れる。

使い方::

    uv run python scripts/export_rerun.py --scene results/demo_seq/railway_seq.npz \
        --output results/video_concept/arrow_railway.rrd

    # ローカルで開く場合
    rerun results/video_concept/arrow_railway.rrd
"""

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from render_dashboard import default_cache_dir, ensure_cache  # noqa: E402


def to_display(points: np.ndarray) -> np.ndarray:
    """参照カメラ座標 (x 右, y 下, z 前) を表示用 (x, -y, z) に変換する。"""
    out = np.array(points, dtype=np.float32, copy=True)
    out[..., 1] *= -1
    return out


def send_blueprint() -> None:
    """ビューアの初期レイアウト（3D を主、右に 2D 2 面）を送る。"""
    import rerun as rr
    import rerun.blueprint as rrb

    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="/world", name="3D 再構成 + 追跡"),
                rrb.Vertical(
                    rrb.Spatial2DView(origin="view/input_2d_tracks", name="入力 + 2D 追跡"),
                    rrb.Spatial2DView(origin="view/depth", name="相対深度"),
                ),
                column_shares=[2, 1],
            ),
        )
    )


def log_scene(cache_dir: Path, manifest: dict, *, frame_offset: int = 0, log_static: bool = True) -> int:
    """キャッシュ済みシーンを現在の rerun セッションへログする（保存はしない）。

    log_static は点群・全軌跡（static=True）を送るかどうか。クリップ結合では
    最初のクリップだけ True にすると、全体で 1 回の静的記録になる。
    フレーム軸は `frame_offset` を起点にした連番。戻り値はログしたフレーム数。
    """
    import rerun as rr

    cached_tracks = np.load(cache_dir / manifest["series"]["tracks"])
    cached_cloud = np.load(cache_dir / manifest["series"]["cloud"])
    xyz = cached_tracks["xyz"]           # [T, P, 3]
    colors = cached_tracks["colors"]     # [P, 3] uint8
    cloud_points = cached_cloud["points"]
    cloud_rgb = cached_cloud["rgb"]

    # 静的記録: 点群と全フレームの軌跡（static=True で全タイムラインに表示。
    # static を付けないと log_time 軸のみに属し、"frame" 軸表示時に隠れる）
    if log_static:
        rr.log(
            "world/reconstruction",
            rr.Points3D(to_display(cloud_points), colors=cloud_rgb, radii=0.06),
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
            rr.LineStrips3D(strips, colors=np.array(strip_colors, dtype=np.uint8), radii=0.05),
            static=True,
        )

    # 時間軸に沿った記録
    pattern = manifest["panels_pattern"]
    frames = manifest["scene"]["frames"]
    for frame_t in range(frames):
        rr.set_time("frame", sequence=frame_offset + frame_t)
        overlay = np.asarray(Image.open(cache_dir / pattern.format(index=frame_t, panel="overlay")))
        depth = np.asarray(Image.open(cache_dir / pattern.format(index=frame_t, panel="depth")))
        rr.log("view/input_2d_tracks", rr.Image(overlay))
        rr.log("view/depth", rr.Image(depth))
        heads = xyz[frame_t]
        keep = np.isfinite(heads).all(-1)
        if keep.any():
            rr.log(
                "world/track_heads",
                rr.Points3D(to_display(heads[keep]), colors=colors[keep], radii=0.05),
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
    args = parser.parse_args(argv)
    args.scene = args.scene.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.cache_dir is not None:
        args.cache_dir = args.cache_dir.expanduser().resolve()
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
    send_blueprint()
    frames = log_scene(cache_dir, manifest, frame_offset=args.frame_offset, log_static=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(args.output))
    size_mb = args.output.stat().st_size / 2**20
    offset_note = f"、frame {args.frame_offset}–{args.frame_offset + frames - 1}" if args.frame_offset else ""
    print(f"rerun 記録を保存: {args.output}（{size_mb:.1f} MB、{frames} フレーム分{offset_note}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
