#!/usr/bin/env python3
"""ビューアを Playwright で操作しながら連番キャプチャし、ffmpeg で動画化する（可視化動画の PoC）。

できること:
- 保存済みシーン (.npz) をビューアで開く
- 表示ビューを切り替える（3D / Depth。Depth レイヤーが無いシーンでは 3D のみ）
- 3D キャンバスを少しずつドラッグして**オービットしながら連番 PNG を保存**
- `--scrub N` で**時間スライダーを N 段階動かしながらキャンバスのみを収録**（複合レイアウト動画用）
- ffmpeg（libx264）で mp4 に変換

使い方::

    # オービット収録（可視化動画の PoC）
    uv run python scripts/viewer_capture.py --scene results/demo_seq/railway_seq.npz \
        --view 3d --orbit-degrees 30 --frames 45 \
        --output results/video_concept/orbit_poc.mp4

    # 時間ステップ収録（render_dashboard.py に渡す 3D パネル素材）
    uv run python scripts/viewer_capture.py --scene results/demo_seq/railway_seq.npz \
        --view 3d --scrub 24 --output results/video_concept/build/scrub.mp4

注意:
- ビューアは 127.0.0.1 の空きポートで起動する。`ffmpeg` が必要。
- 取り込みレートはスクリーンショット速度（本機で 2〜4 fps 程度）に依存する。
  出力 fps は `--fps`（既定 24）で指定する。
"""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from viewer_screenshot import CHROMIUM_ARGS, viewer_server, wait_for_scene_ready  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# ドラッグ量（px）と見た目の回転量（度）の目安。キャンバスの幅に比例させて使う。
PIXELS_PER_DEGREE = 6.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ビューア操作つき連番キャプチャ（PoC）")
    parser.add_argument("--scene", type=Path, required=True, help="開く保存済みシーン (.npz)")
    parser.add_argument("--view", choices=["3d", "depth"], help="表示ビューの切り替え（任意）")
    parser.add_argument("--orbit-degrees", type=float, default=24.0,
                        help="キャプチャ中の回転量の目安（度。0 で静止。--scrub 指定時は無視）")
    parser.add_argument("--scrub", type=int, default=0,
                        help="時間スライダーを N 段階動かしながらキャンバスのみを収録する（0 で無効）")
    parser.add_argument("--frames", type=int, default=36, help="キャプチャするフレーム数")
    parser.add_argument("--fps", type=int, default=24, help="出力動画のフレームレート")
    parser.add_argument("--width", type=int, default=1920, help="ビューポート幅")
    parser.add_argument("--height", type=int, default=1080, help="ビューポート高さ")
    parser.add_argument("--render-wait", type=float, default=1.5, help="開始前の描画待ち（秒）")
    parser.add_argument("--output", type=Path, required=True, help="出力 mp4 パス")
    parser.add_argument("--clean-frames", action="store_true",
                        help="動画化後に連番 PNG を削除する")
    parser.add_argument("--url", help="起動済みビューアの URL（省略時は起動する）")
    args = parser.parse_args(argv)
    args.scene = args.scene.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.scene.is_file():
        parser.error(f"シーンが見つかりません: {args.scene}")
    if args.frames < 2:
        parser.error("frames は 2 以上が必要です")
    return args


def capture_frames(url: str, frames_dir: Path, args: argparse.Namespace) -> int:
    """ビューアを開いて操作しながら連番 PNG を保存し、保存枚数を返す。"""
    from playwright.sync_api import sync_playwright
    from viewer_screenshot import _status_probe

    frames_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=CHROMIUM_ARGS)
        context = browser.new_context(
            viewport={"width": args.width, "height": args.height}, device_scale_factor=1
        )
        try:
            page = context.new_page()
            page.goto(url, wait_until="load")
            wait_for_scene_ready(page, timeout_s=120.0, probe=_status_probe(url))
            if args.view:
                label = "3D" if args.view == "3d" else "Depth"
                try:
                    # 表示ビューは role="radio" のセグメントボタン（3D / Depth）
                    page.get_by_role("radio", name=label, exact=True).click()
                    print(f"表示ビューを {label} に切り替え")
                except Exception:
                    print(f"（表示ビューの切り替えボタンが見つからないためスキップ: {label}）")
            page.wait_for_timeout(int(args.render_wait * 1000))

            canvas = page.locator("#scene-canvas")
            box = canvas.bounding_box()
            if box is None:
                raise RuntimeError("キャンバスが見つかりません")

            if args.scrub:
                return scrub_frames(page, canvas, frames_dir, args)

            cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
            step_px = (args.orbit_degrees * PIXELS_PER_DEGREE) / max(1, args.frames - 1)

            for index in range(args.frames):
                if args.orbit_degrees and index:
                    # 小さなドラッグを繰り返して少しずつ回す（相対回転）
                    page.mouse.move(cx, cy)
                    page.mouse.down()
                    page.mouse.move(cx + step_px, cy, steps=3)
                    page.mouse.up()
                    page.wait_for_timeout(60)
                page.screenshot(path=str(frames_dir / f"frame_{index:04d}.png"))
                saved += 1
                if (index + 1) % 10 == 0:
                    print(f"  … {index + 1}/{args.frames} 枚")
        finally:
            context.close()
            browser.close()
    return saved


def scrub_frames(page, canvas, frames_dir: Path, args: argparse.Namespace) -> int:
    """時間スライダーを N 段階動かしながら、キャンバス要素のみを保存する。

    render_dashboard.py 用に、UI を含まない 3D パネル素材を作るモード。
    """
    slider = page.evaluate(
        "() => { const el = document.querySelector('#time');"
        " if (!el || el.disabled) return null;"
        " return { min: Number(el.min), max: Number(el.max) }; }"
    )
    if slider is None or not (slider["max"] > slider["min"]):
        print("（時間スライダーが使えないため、静止のまま収録します）")
        slider = None
    for index in range(args.scrub):
        if slider is not None:
            value = slider["min"] + (slider["max"] - slider["min"]) * index / max(1, args.scrub - 1)
            # React の制御 input は value 直接代入では onChange が発火しないため、
            # プロトタイプのネイティブ setter で値を入れ直してから input イベントを送る。
            page.evaluate(
                "(value) => {"
                " const el = document.querySelector('#time');"
                " const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;"
                " setter.call(el, value);"
                " el.dispatchEvent(new Event('input', { bubbles: true }));"
                " }",
                value,
            )
            page.wait_for_timeout(150)
        canvas.screenshot(path=str(frames_dir / f"frame_{index:04d}.png"))
        if (index + 1) % 6 == 0:
            print(f"  … {index + 1}/{args.scrub} 枚")
    return args.scrub


def encode_video(frames_dir: Path, output: Path, fps: int) -> None:
    """連番 PNG を H.264 の mp4 に変換する。"""
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg が見つかりません")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-v", "error",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        "-movflags", "+faststart",
        str(output),
    ]
    subprocess.run(command, check=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    frames_dir = args.output.parent / f"frames_{args.output.stem}"

    if args.url:
        saved = capture_frames(args.url, frames_dir, args)
    else:
        args.scene = args.scene.resolve()
        with viewer_server(args.scene, wait=120.0) as url:
            saved = capture_frames(url, frames_dir, args)

    encode_video(frames_dir, args.output, args.fps)
    duration = saved / args.fps
    size_mb = args.output.stat().st_size / 2**20
    print(f"動画を保存: {args.output}（{saved} 枚 / {args.fps} fps ≒ {duration:.1f} 秒、{size_mb:.1f} MB）")
    print(f"連番 PNG: {frames_dir}")
    if args.clean_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)
        print("連番 PNG を削除しました")
    return 0


if __name__ == "__main__":
    sys.exit(main())
