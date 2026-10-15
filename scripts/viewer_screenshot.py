#!/usr/bin/env python3
"""ARROW ビューアをヘッドレスブラウザ（Playwright）で開き、操作と可視化を PNG に保存する。

参考実装と同じ「web ビューアを起動 → Playwright で開く → 描画を待つ → スクリーンショット」方式:

- yuki-inaho/vggt-omega ``scripts/visualize.py``（``rerun --serve-web`` + ``sync_playwright``）
- yuki-inaho/Open-d4rt ``scripts/visualize_rerun.py``（``playwright screenshot`` CLI）

2 つのモード:

- ``--run-sample``: サンプルデータ（``assets/videos/railway.mp4``）を **UI 上で**実行する。
  「From server」→「Add selected」→「Run」を実際にクリックし、''Scene ready'' を待って撮影。
- ``--scene 保存済み.npz``: 保存済み結果をビューアで開いて撮影（推論なし、高速）。

使い方::

    uv run python scripts/viewer_screenshot.py --run-sample --output results/ui/railway_sample.png
    uv run python scripts/viewer_screenshot.py --scene results/railway.npz --output results/ui/railway_scene.png

サーバを別途起動済みなら ``--url http://127.0.0.1:7860`` で接続できる。
スマホやデスクトップと同じ見た目になるよう、既定のビューポートは 1600x1000 ピクセル。

注意: このコンテナでは GPU が無い設定でも WebGL が動くよう SwiftShader を指定している。
"""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time

import httpx

REPO = Path(__file__).resolve().parents[1]

# SwiftShader で WebGL を有効化する（ヘッドレスでの既定の組み合わせ）。
CHROMIUM_ARGS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
]

VIEWPORT: dict[str, int] = {"width": 1600, "height": 1000}


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def wait_for_http(url: str, *, process: subprocess.Popen | None = None, timeout: float = 120.0):
    """サーバが HTTP に応答するまで待つ。プロセスが先に落ちたらその場で失敗させる。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"ビューアが起動前に終了した（コード {process.returncode}）")
        try:
            if httpx.get(f"{url}/api/status", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    raise TimeoutError(f"ビューア {url} が {timeout:.0f} 秒以内に起動しなかった")


@contextmanager
def viewer_server(scene: Path | None = None, *, port: int | None = None, wait: float = 120.0):
    """``arrow-viewer`` を起動し、``http://127.0.0.1:<port>`` を yield する。"""
    port = free_port() if port is None else port
    command = [sys.executable, "-m", "arrow.viewer.server", "--host", "127.0.0.1", "--port", str(port)]
    if scene is not None:
        command += ["--scene", str(scene)]
    with tempfile.TemporaryFile("w+", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        url = f"http://127.0.0.1:{port}"
        try:
            wait_for_http(url, process=process, timeout=wait)
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            if process.returncode not in (0, -15, 143):
                log.seek(0)
                print("ビューアのログ（末尾）:\n" + log.read()[-2000:], file=sys.stderr)


def _status_probe(url: str):
    """サーバの /api/status を読む関数を返す（読めないときは None）。"""
    def probe() -> dict | None:
        try:
            return httpx.get(f"{url}/api/status", timeout=10).json()
        except httpx.HTTPError:
            return None
    return probe


def click_through_sample_run(page, *, timeout_s: float = 900.0, probe=None) -> None:
    """UI でサンプル動画を読み込み、Run を押して ''Scene ready'' まで待つ。"""
    # 起動直後は「From device」タブ。サーバ側の例（assets/videos）へ切り替える。
    page.get_by_role("tab", name="From server").click()
    add = page.get_by_role("button", name=re.compile(r"^Add selected"))
    add.wait_for(state="visible", timeout=30_000)
    if add.is_disabled():
        # 事前選択されていなければ railway.mp4 の行をクリックして選択する。
        page.locator("#clip-browser").get_by_role("button", name=re.compile(r"railway\.mp4")).click()
    add.click()

    run = page.locator("#run-inference")
    run.wait_for(state="visible", timeout=30_000)
    page.wait_for_function(
        "() => { const b = document.querySelector('#run-inference'); return b && !b.disabled; }",
        timeout=60_000,
    )
    print("UI 計画:", (page.text_content("#readout") or "").splitlines()[0])
    run.click()
    wait_for_scene_ready(page, timeout_s=timeout_s, probe=probe)


def wait_for_scene_ready(page, *, timeout_s: float = 900.0, probe=None) -> None:
    """``#status`` が ''Scene ready'' になるまで待つ。

    サーバ側が error / cancelled になったら、タイムアウトを待たずに即座に失敗させる
    （``probe`` は /api/status の JSON を返す関数。省略時は DOM のみを見る）。
    """
    deadline = time.monotonic() + timeout_s
    last_note = 0.0
    while time.monotonic() < deadline:
        status = page.text_content("#status") or ""
        if "Scene ready" in status:
            return
        alert = page.query_selector("div.error[role='alert']")
        if alert is not None:
            raise RuntimeError(f"UI がエラーを表示した: {alert.inner_text().strip()}")
        info = probe() if probe is not None else None
        if info is not None:
            state = info.get("state")
            if state == "error":
                raise RuntimeError(
                    f"サーバ側エラー: {info.get('stage', '')}: {info.get('detail', '')}".rstrip(": ")
                )
            if state == "cancelled":
                raise RuntimeError("実行がキャンセルされた")
            if state == "running" and time.monotonic() - last_note >= 20:
                print(
                    f"… {info.get('stage', '')} {float(info.get('progress', 0.0)):.0%} "
                    f"{info.get('detail', '')}".rstrip(),
                    flush=True,
                )
                last_note = time.monotonic()
        page.wait_for_timeout(1000)
    raise TimeoutError(f"推論が {timeout_s:.0f} 秒以内に完了しなかった")


@dataclass
class SceneSummary:
    points: int
    tracks: int
    cameras: int

    def text(self) -> str:
        parts = []
        if self.points:
            parts.append(f"{self.points:,} points")
        if self.tracks:
            parts.append(f"{self.tracks:,} tracks")
        if self.cameras:
            parts.append(f"{self.cameras} camera poses")
        return "、".join(parts) or "（内容なし）"


def scene_summary(page) -> SceneSummary:
    """``.scene-info`` の表示（`48,221 tracks` など）を数値に戻す。"""
    info = page.query_selector(".scene-info")
    text = info.inner_text() if info is not None else ""
    def number(label: str) -> int:
        match = re.search(rf"([\d,]+)\s+{label}", text)
        return int(match.group(1).replace(",", "")) if match else 0
    return SceneSummary(number("points"), number("tracks"), number("camera poses"))


def capture(
    url: str,
    output: Path,
    *,
    run_sample: bool,
    canvas_output: Path | None = None,
    timeout_s: float = 900.0,
    render_wait: float = 2.5,
    viewport: dict[str, int] = VIEWPORT,
) -> SceneSummary:
    """ブラウザを開いて（必要ならサンプル実行して）描画後に PNG を保存する。"""
    from playwright.sync_api import sync_playwright

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=CHROMIUM_ARGS)
        context = browser.new_context(
            viewport={"width": viewport["width"], "height": viewport["height"]},
            device_scale_factor=1,
        )
        try:
            page = context.new_page()
            page.goto(url, wait_until="load")
            probe = _status_probe(url)
            if run_sample:
                click_through_sample_run(page, timeout_s=timeout_s, probe=probe)
            else:
                wait_for_scene_ready(page, timeout_s=120.0, probe=probe)
            page.wait_for_timeout(int(render_wait * 1000))
            summary = scene_summary(page)
            page.screenshot(path=str(output))
            print(f"スクリーンショットを保存: {output}（{summary.text()}）")
            if canvas_output is not None:
                canvas_output = Path(canvas_output)
                canvas_output.parent.mkdir(parents=True, exist_ok=True)
                page.locator("#scene-canvas").screenshot(path=str(canvas_output))
                print(f"キャンバスのみを保存: {canvas_output}")
            return summary
        finally:
            context.close()
            browser.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--run-sample", action="store_true", help="UI 上でサンプル動画を実行する")
    parser.add_argument("--scene", type=Path, help="開く保存済み結果 (.npz)")
    parser.add_argument("--url", help="起動済みビューアの URL（省略時はこのスクリプトが起動する）")
    parser.add_argument("--output", type=Path, default=Path("results/ui/viewer.png"))
    parser.add_argument("--canvas-output", type=Path, help="3D キャンバス領域だけを別途保存する")
    parser.add_argument("--width", type=int, default=VIEWPORT["width"])
    parser.add_argument("--height", type=int, default=VIEWPORT["height"])
    parser.add_argument("--render-wait", type=float, default=2.5, help="描画を待つ秒数")
    parser.add_argument("--timeout", type=float, default=900.0, help="推論の制限時間（秒）")
    args = parser.parse_args(argv)
    if not args.run_sample and args.scene is None and args.url is None:
        parser.error("--run-sample、--scene、--url のいずれかを指定してください")
    viewport = {"width": args.width, "height": args.height}

    if args.url:
        summary = capture(
            args.url, args.output, run_sample=args.run_sample, canvas_output=args.canvas_output,
            timeout_s=args.timeout, render_wait=args.render_wait, viewport=viewport,
        )
    else:
        with viewer_server(args.scene) as url:
            summary = capture(
                url, args.output, run_sample=args.run_sample, canvas_output=args.canvas_output,
                timeout_s=args.timeout, render_wait=args.render_wait, viewport=viewport,
            )
    return 0 if (summary.tracks or summary.points or summary.cameras) else 1


if __name__ == "__main__":
    sys.exit(main())
