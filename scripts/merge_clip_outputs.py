#!/usr/bin/env python3
"""クリップ単位のダッシュボード成果物（動画・rerun）を単一ファイルへ結合する後加工ツール。

- 動画: クリップ動画（*_dashboard.mp4）を開始フレーム順に ffmpeg concat（無再エンコード）で結合する。
- rerun: 各クリップを開始フレームのオフセット付きで 1 セッションへ連番ログし、
  単一レコーディングの 1 つの .rrd として保存する（フレーム軸は全クリップ通しの連番）。
- クリップ順と開始番号・フレーム数は manifest.json（process_sequence.py の出力）から取得する。
  無い場合はファイル名の `clip_<start>-` から読み取る。

前提: 結合対象のクリップ動画は本ツールの元になったマルチパネル動画
（render_multipanel.py の出力）であること。rerun はキャッシュ（series/panels）から再描画する。

使い方::

    uv run python scripts/merge_clip_outputs.py --clips-dir results/grass_clips \
        --videos-dir results/grass_video \
        --output-video results/grass_video/grass_all_dashboard.mp4 \
        --output-rrd results/grass_video/grass_all.rrd
"""

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

import export_rerun  # noqa: E402


def log(message: str) -> None:
    print(message, flush=True)


def discover_clips(clips_dir: Path) -> list[tuple[Path, int, int]]:
    """(NPZ パス, 開始フレーム, フレーム数) を開始順に返す。manifest.json を優先する。"""
    manifest_path = clips_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        clips = [
            (clips_dir / entry["name"], int(entry["start"]), int(entry["frames"]))
            for entry in manifest["clips"]
            if entry.get("status") == "done"
        ]
    else:
        pattern = re.compile(r"clip_(\d+)-")
        clips = []
        for path in sorted(clips_dir.glob("*.npz")):
            match = pattern.search(path.name)
            if match:
                clips.append((path, int(match.group(1)), 0))
    clips.sort(key=lambda item: item[1])
    return clips


def merge_videos(videos_dir: Path, clips: list[tuple[Path, int, int]], output: Path) -> None:
    """クリップ動画を ffmpeg concat（無再エンコード）で単一 MP4 に結合する。"""
    paths = []
    for npz_path, _, _ in clips:
        candidates = [videos_dir / f"{npz_path.stem}_dashboard.mp4"]
        stripped = re.sub(r"_s\d+$", "", npz_path.stem)  # 例: clip_0000-0024_s1 → clip_0000-0024
        if stripped != npz_path.stem:
            candidates.append(videos_dir / f"{stripped}_dashboard.mp4")
        video = next((path for path in candidates if path.is_file()), None)
        if video is None:
            raise SystemExit(f"クリップ動画が見つかりません: {' または '.join(map(str, candidates))}")
        paths.append(video)
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg が見つかりません")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        for path in paths:
            handle.write(f"file '{path.resolve()}'\n")
        list_path = Path(handle.name)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
             "-i", str(list_path), "-c", "copy", "-movflags", "+faststart", str(output)],
            check=True,
        )
    finally:
        list_path.unlink()

    probe_note = ""
    if shutil.which("ffprobe") is not None:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(output)],
            capture_output=True, text=True, check=True,
        )
        values = [value for value in probe.stdout.split() if value]
        if len(values) >= 2:
            probe_note = f" / {values[0]} フレーム / {float(values[1]):.1f} 秒"
    log(f"動画を結合: {output}（{len(paths)} クリップ{probe_note}、"
        f"{output.stat().st_size / 2**20:.1f} MB）")


def merge_reruns(clips: list[tuple[Path, int, int]], output: Path, pick: int,
                 alignment_path: Path | None = None) -> None:
    """全クリップを 1 セッションへ連番フレームでログし、単一の .rrd として保存する。

    rrd の単純マージでは recording が分割されるため、同一プロセス内で連続ログする。
    フレーム軸は全クリップ通しの連番（クリップの開始フレーム番号をオフセットに使う）。
    alignment_path 指定時は各クリップを世界座標へ変換してログする（クリップ間の連続性）。
    """
    import rerun as rr

    from render_dashboard import default_cache_dir, ensure_cache

    alignment_data = None
    if alignment_path is not None:
        alignment_data = json.loads(alignment_path.read_text(encoding="utf-8"))
        log(f"alignment を適用: {alignment_path}")

    output.parent.mkdir(parents=True, exist_ok=True)
    rr.init("arrow_all_clips", spawn=False)
    first_cache = default_cache_dir(clips[0][0])
    first_alignment = export_rerun.load_alignment_entry(alignment_data, clips[0][0].name)
    eye, target = export_rerun.estimate_view(first_cache, first_alignment)
    export_rerun.send_blueprint(eye, target)
    frame_end = 0
    log_static = True
    for npz_path, start, _ in clips:
        cache_dir = default_cache_dir(npz_path)
        _, manifest, hit = ensure_cache(npz_path, cache_dir, pick, None, rebuild=False)
        part_alignment = export_rerun.load_alignment_entry(alignment_data, npz_path.name)
        log(f"  rerun 部分記録: {npz_path.name}（frame {start} から、"
            f"キャッシュ{'ヒット' if hit else '構築'}）")
        frames = export_rerun.log_scene(cache_dir, manifest, frame_offset=start,
                                        log_static=log_static, alignment=part_alignment)
        frame_end = start + frames
        log_static = False
    rr.save(str(output))
    world_note = "・世界座標" if alignment_data is not None else ""
    log(f"rerun を結合: {output}（{len(clips)} クリップ / 全 {frame_end} フレーム、"
        f"{output.stat().st_size / 2**20:.1f} MB、単一レコーディング・フレーム軸連番{world_note}）")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="クリップ成果物（動画・rerun）を単一ファイルへ結合する")
    parser.add_argument("--clips-dir", type=Path, required=True,
                        help="クリップ NPZ と manifest.json の場所")
    parser.add_argument("--videos-dir", type=Path,
                        help="クリップ動画（*_dashboard.mp4）の場所（既定: clips-dir）")
    parser.add_argument("--output-video", type=Path,
                        help="結合動画の出力パス（既定: clips-dir/merged_all_dashboard.mp4）")
    parser.add_argument("--output-rrd", type=Path,
                        help="結合 rerun の出力パス（既定: clips-dir/merged_all.rrd）")
    parser.add_argument("--pick", type=int, default=56, help="軌跡の本数（既定 56。キャッシュと一致させる）")
    parser.add_argument("--alignment", type=Path,
                        help="align_clip_scenes.py の alignments.json（世界座標接続を適用）")
    parser.add_argument("--skip-video", action="store_true", help="動画の結合を省略する")
    parser.add_argument("--skip-rrd", action="store_true", help="rerun の結合を省略する")
    args = parser.parse_args(argv)
    args.clips_dir = args.clips_dir.expanduser().resolve()
    args.videos_dir = (args.videos_dir or args.clips_dir).expanduser().resolve()
    args.output_video = (args.output_video or args.clips_dir / "merged_all_dashboard.mp4").expanduser().resolve()
    args.output_rrd = (args.output_rrd or args.clips_dir / "merged_all.rrd").expanduser().resolve()
    if args.alignment is not None:
        args.alignment = args.alignment.expanduser().resolve()
    if not args.clips_dir.is_dir():
        parser.error(f"クリップディレクトリが見つかりません: {args.clips_dir}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    clips = discover_clips(args.clips_dir)
    if not clips:
        raise SystemExit(f"クリップが見つかりません: {args.clips_dir}")
    total_frames = sum(frames for _, _, frames in clips)
    log(f"クリップ {len(clips)} 個（全 {total_frames} フレーム）を結合する — {args.clips_dir}")
    if not args.skip_video:
        merge_videos(args.videos_dir, clips, args.output_video)
    if not args.skip_rrd:
        merge_reruns(clips, args.output_rrd, args.pick, args.alignment)
    return 0


if __name__ == "__main__":
    sys.exit(main())
