#!/usr/bin/env python3
"""連番画像から ARROW のクリップ成果物（NPZ）を量産するバッチスクリプト。

- 入力: 連番画像フォルダ（自然順）または動画。`--start` で開始位置を指定できる。
- 出力: `<output-dir>/clip_<start>-<end>_s<stride>.npz` と `manifest.json`。
- 再開可能: 完了済みクリップ（manifest 上 done かつ NPZ の軽い検査に合格）は
  `--force` なしでスキップする。壊れた出力は自動で実行し直す。
- 検査: 各クリップの metadata を読み、format_version と選択フレーム数を確認してから
  manifest に記録する（実行時間・ピークメモリ・クエリ数も残る）。

使い方::

    # 実行計画の確認（推論はしない）
    uv run python scripts/process_sequence.py results/demo_seq/frames --dry-run

    # 12 フレームずつに分割して track と pose を実行
    uv run python scripts/process_sequence.py results/demo_seq/frames \
        --output-dir results/demo_seq/clips --clip-frames 12 \
        --tasks track pose --track-stride 8

    # 中断後は同じコマンドを再実行するだけでよい（完了済みはスキップされる）

モデルの重みを使うため、初回は Hugging Face からのダウンロードが発生する。
このマシンでは実行前に LD_LIBRARY_PATH の前置が必要（docs/ONBOARDING.md 参照）。
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]

MANIFEST_NAME = "manifest.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def total_frames(path: Path) -> int:
    """入力の総フレーム数を数える（フォルダは自然順の枚数、動画はストリーム情報）。"""
    from arrow.data.io import media_image_paths

    paths = media_image_paths(path)
    if paths is not None:
        return len(paths)
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = int(stream.frames or 0)
        if not frames and stream.duration and stream.time_base and stream.average_rate:
            frames = round(float(stream.duration * stream.time_base) * float(stream.average_rate))
        if not frames:
            raise SystemExit(
                f"動画のフレーム数を取得できません: {path}\n"
                "連番画像に展開してから実行してください。"
            )
        return frames


def plan_clips(start: int, total: int, clip_frames: int, stride: int) -> list[tuple[int, int]]:
    """(start, end) の一覧を返す。end は排他（CLI の --end と同じ意味）。"""
    clips = []
    cursor = start
    span = clip_frames * stride
    while cursor < total:
        end = min(cursor + span, total)
        if end <= cursor:
            break
        clips.append((cursor, end))
        cursor = end
    return clips


def clip_name(start: int, end: int, stride: int) -> str:
    return f"clip_{start:04d}-{end:04d}_s{stride}.npz"


def expected_selected(start: int, end: int, stride: int) -> int:
    return len(range(start, end, stride))


def inspect_npz(path: Path) -> dict | None:
    """NPZ の軽い検査。正しければ metadata の要約、そうでなければ None を返す。"""
    import numpy as np

    try:
        with np.load(path, allow_pickle=False) as arrays:
            metadata = json.loads(str(arrays["metadata"]))
            if metadata.get("format_version") != 1:
                return None
            return {
                "format_version": metadata["format_version"],
                "frames": len(metadata.get("frame_info", [])),
                "seconds": metadata.get("runtime", {}).get("seconds"),
                "peak_allocated_bytes": metadata.get("runtime", {}).get("peak_allocated_bytes"),
                "queries": sum(group.get("count", 0) for group in metadata.get("groups", [])),
            }
    except Exception:
        # 壊れた・途中で切れた NPZ は zipfile.BadZipFile / ValueError / KeyError / OSError など
        # さまざまな例外になる。ここは「再実行すべき出力」として扱う。
        return None


def inference_command(args, start: int, end: int, output: Path) -> list[str]:
    command = [
        sys.executable, "-m", "arrow.cli", "infer", str(args.input),
        "--start", str(start), "--end", str(end), "--stride", str(args.frame_stride),
        "--tasks", *args.tasks, "--output", str(output),
    ]
    for flag, value in [
        ("--max-side", args.max_side),
        ("--device", args.device),
        ("--track-stride", args.track_stride),
        ("--reconstruction-stride", args.reconstruction_stride),
        ("--depth-stride", args.depth_stride),
        ("--decode-chunk-size", args.decode_chunk_size),
    ]:
        if value is not None:
            command += [flag, str(value)]
    return command


def load_manifest(path: Path, signature: dict) -> dict:
    if path.is_file():
        try:
            manifest = json.loads(path.read_text())
        except ValueError:
            manifest = None
        if manifest and manifest.get("signature") == signature:
            return manifest
        if manifest is not None:
            backup = path.with_suffix(".json.bak")
            path.rename(backup)
            print(f"設定が前回と異なるため、旧 manifest を退避しました: {backup}")
    return {"signature": signature, "clips": []}


def save_manifest(path: Path, manifest: dict) -> None:
    manifest["updated_at"] = utc_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="連番画像から ARROW のクリップ成果物（NPZ）を量産する",
    )
    parser.add_argument("input", type=Path, help="連番画像フォルダまたは動画")
    parser.add_argument("--output-dir", type=Path,
                        help="成果物の出力先（既定: results/<入力名>_clips）")
    parser.add_argument("--clip-frames", type=int, default=24,
                        help="1 クリップあたりのフレーム数（既定: 24）")
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="クリップ内で何フレームおきに使うか（既定: 1）")
    parser.add_argument("--start", type=int, default=0, help="開始フレーム番号（既定: 0）")
    parser.add_argument("--tasks", nargs="+", default=["track", "pose"],
                        help="実行するタスク（既定: track pose）")
    parser.add_argument("--max-side", type=int, help="リサイズ後の長辺（既定: 504）")
    parser.add_argument("--device", help="cuda / cuda:1 / mps / cpu（既定: 自動）")
    parser.add_argument("--track-stride", type=int, help="トラック格子の間隔（既定: 2）")
    parser.add_argument("--reconstruction-stride", type=int, help="再構成格子の間隔（既定: 2）")
    parser.add_argument("--depth-stride", type=int, help="深度格子の間隔（既定: 2）")
    parser.add_argument("--decode-chunk-size", type=int, help="デコードのチャンク幅（既定: 100000）")
    parser.add_argument("--force", action="store_true", help="完了済みクリップも再実行する")
    parser.add_argument("--dry-run", action="store_true", help="実行計画だけ表示する")
    args = parser.parse_args(argv)
    args.input = args.input.expanduser().resolve()
    if not args.input.exists():
        parser.error(f"入力が見つかりません: {args.input}")
    if args.clip_frames < 1 or args.frame_stride < 1 or args.start < 0:
        parser.error("clip-frames と frame-stride は 1 以上、start は 0 以上が必要です")
    if args.output_dir is None:
        args.output_dir = REPO / "results" / f"{args.input.stem}_clips"
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    total = total_frames(args.input)
    clips = plan_clips(args.start, total, args.clip_frames, args.frame_stride)
    if not clips:
        raise SystemExit(f"選択範囲にフレームがありません（総数 {total}、開始 {args.start}）")

    signature = {
        "input": str(args.input),
        "clip_frames": args.clip_frames,
        "frame_stride": args.frame_stride,
        "start": args.start,
        "tasks": args.tasks,
        "max_side": args.max_side,
        "device": args.device,
        "track_stride": args.track_stride,
        "reconstruction_stride": args.reconstruction_stride,
        "depth_stride": args.depth_stride,
        "decode_chunk_size": args.decode_chunk_size,
    }
    manifest_path = args.output_dir / MANIFEST_NAME

    print(f"入力: {args.input}（総フレーム数 {total}）")
    print(f"出力: {args.output_dir}")
    print(f"クリップ: {len(clips)} 個 × 最大 {args.clip_frames} フレーム"
          f"（{args.frame_stride} フレームおき、タスク: {' '.join(args.tasks)}）")
    if args.dry_run:
        for index, (start, end) in enumerate(clips, 1):
            print(f"  [{index}/{len(clips)}] {clip_name(start, end, args.frame_stride)}"
                  f"（フレーム {start}–{end - 1}、{expected_selected(start, end, args.frame_stride)} 枚）")
        return 0

    manifest = load_manifest(manifest_path, signature)
    records = {record["name"]: record for record in manifest["clips"]}
    failed = 0
    started = time.perf_counter()

    for index, (start, end) in enumerate(clips, 1):
        name = clip_name(start, end, args.frame_stride)
        output = args.output_dir / name
        expected = expected_selected(start, end, args.frame_stride)
        prefix = f"[{index}/{len(clips)}] {name}"
        record = records.get(name)

        if not args.force and record and record.get("status") == "done" and output.is_file():
            info = inspect_npz(output)
            if info is not None and info["frames"] == expected:
                print(f"{prefix}: スキップ（完了済み、{info['queries']:,} クエリ）")
                continue
            print(f"{prefix}: 出力が不完全なため再実行する")

        output.parent.mkdir(parents=True, exist_ok=True)
        command = inference_command(args, start, end, output)
        print(f"{prefix}: 実行（フレーム {start}–{end - 1}、{expected} 枚）")
        clip_started = time.perf_counter()
        result = subprocess.run(command, cwd=REPO)
        elapsed = time.perf_counter() - clip_started

        entry = {
            "name": name,
            "start": start,
            "end": end,
            "stride": args.frame_stride,
            "frames": expected,
            "finished_at": utc_now(),
        }
        info = inspect_npz(output) if result.returncode == 0 else None
        if info is not None and info["frames"] == expected:
            entry.update(status="done", seconds=info["seconds"], wall_seconds=round(elapsed, 2),
                         queries=info["queries"], peak_allocated_bytes=info["peak_allocated_bytes"],
                         size_bytes=output.stat().st_size)
            print(f"{prefix}: 完了（{entry['wall_seconds']} 秒、{entry['queries']:,} クエリ、"
                  f"{entry['size_bytes'] / 2**20:.1f} MB）")
        else:
            failed += 1
            entry.update(status="failed", returncode=result.returncode, wall_seconds=round(elapsed, 2))
            print(f"{prefix}: 失敗（終了コード {result.returncode}）")
        records[name] = entry
        manifest["clips"] = [records[key] for key in sorted(records)]
        save_manifest(manifest_path, manifest)

    done = sum(1 for record in records.values() if record.get("status") == "done")
    print(f"完了 {done} / {len(clips)} クリップ（{time.perf_counter() - started:.1f} 秒）")
    print(f"manifest: {manifest_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
