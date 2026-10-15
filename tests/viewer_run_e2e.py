"""ビューア経由の推論実行 E2E テスト。

arrow-viewer を起動し、バンドル済みアセットに対して POST /api/run で推論を開始し、
セッションが complete になるまで待ってから、配信されるシーンを確認する。

実行方法: uv run python tests/viewer_run_e2e.py
"""

from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx

REPO = Path(__file__).resolve().parents[1]
ASSET = REPO / "assets" / "videos" / "railway.mp4"

# 空きポートを OS に選ばせる
with socket.socket() as probe:
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]

process = subprocess.Popen(
    [
        sys.executable, "-m", "arrow.viewer.server",
        "--host", "127.0.0.1", "--port", str(port),
    ],
    cwd=REPO,
)
url = f"http://127.0.0.1:{port}"
try:
    # ビューアがリクエストを受け付けるまで待つ
    deadline = time.monotonic() + 120
    while True:
        if process.poll() is not None:
            raise SystemExit(f"ビューアが早期終了した: {process.returncode}")
        try:
            if httpx.get(f"{url}/api/status", timeout=2).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        if time.monotonic() > deadline:
            raise SystemExit("ビューアが起動しなかった")
        time.sleep(0.5)
    print("ビューア起動:", url)

    # アセットの先頭 6 フレーム（3 フレームおき）を track タスクで推論する
    request = {
        "path": str(ASSET),
        "tasks": [{"task": "track", "stride": 16}],
        "start": 0,
        "stride": 3,
        "frames": 6,
        "source_frame": 0,
    }
    plan = httpx.post(f"{url}/api/plan", json=request, timeout=30)
    print("POST /api/plan ->", plan.status_code, str(plan.json())[:120])
    started = httpx.post(f"{url}/api/run", json=request, timeout=30)
    print("POST /api/run ->", started.status_code, started.json())

    # セッションが running から抜けるまで待つ
    state, deadline = "running", time.monotonic() + 600
    while state == "running":
        status = httpx.get(f"{url}/api/status", timeout=10).json()
        state = status["state"]
        if state == "running":
            if time.monotonic() > deadline:
                raise SystemExit("推論が 10 分以内に終わらなかった")
            time.sleep(1.0)
    print("最終状態:", state, "| clip:", status.get("clip"))
    assert state == "complete", status
    assert status["has_result"] is True

    # 生成されたシーンを取得する
    scene = httpx.get(f"{url}/api/scene", timeout=60)
    print("GET /api/scene ->", scene.status_code, "scene key:", scene.headers.get("X-Scene-Key"))
    payload = scene.json()
    print("シーンのキー:", sorted(payload)[:12])
    print("ビューア経由の推論 E2E: OK")
finally:
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
