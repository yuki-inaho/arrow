"""Playwright でビューア UI をブラウザ上で動かすエンドツーエンドテスト。

- test_ui_runs_sample_clip: サンプル動画（assets/videos/railway.mp4）を **UI 上で**
  実行する（From server → Add selected → Run）。''Scene ready'' になった後、
  トラックとカメラ姿勢が描画されることを確認し、PNG を保存する。
- test_ui_opens_saved_result: 保存済み結果（results/railway.npz）を開いて描画を確認する。

実行方法（``uv sync`` で環境作成済み、Chromium は導入済みとする）::

    uv run playwright install chromium
    uv run pytest tests/test_viewer_ui.py -v -s

実モデルを動かす sample テストは、このマシンでは LD_LIBRARY_PATH の前置が必要
（tests/test_e2e_arrow.py の docstring を参照）。
スクリーンショットは results/ui/ に保存される。
"""

from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.viewer_screenshot import capture, viewer_server  # noqa: E402

pytestmark = pytest.mark.ui

ARTIFACTS = REPO / "results" / "ui"


@pytest.fixture(autouse=True)
def free_pytest_gpu_memory():
    """UI テストの前に、テストプロセスが保持する GPU キャッシュ（モデル重み等）を返却する。

    ビューアは別プロセスで 24 フレーム推論（ピーク約 15GB）を行うため、先行する e2e
    モジュールが読み込んだ重み（約 5.3GB）のキャッシュを残したままにすると、
    24GB GPU では CUDA OOM になる（本スイートで実際に発生した）。
    """
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    yield


def canvas_stats(png: Path) -> dict:
    """キャンバスの PNG が「空でない」ことを示す統計（輝度のばらつきと色数）。"""
    import numpy as np
    from PIL import Image

    array = np.asarray(Image.open(png).convert("RGB")).astype(float)
    quantized = (array // 16).reshape(-1, 3)
    return {"std": float(array.std()), "colors": int(len(np.unique(quantized, axis=0)))}


def assert_canvas_drawn(canvas: Path) -> None:
    assert canvas.is_file() and canvas.stat().st_size > 0
    stats = canvas_stats(canvas)
    # 背景だけなら輝度のばらつきも色数も小さい。点群・トラックが描かれていれば十分大きい。
    assert stats["std"] > 5, f"キャンバスが単色の可能性がある: {stats}"
    assert stats["colors"] > 50, f"キャンバスの色数が少なすぎる: {stats}"


def test_ui_runs_sample_clip():
    """サンプル動画を UI 上で実行し、シーンが可視化されること。"""
    full = ARTIFACTS / "railway_sample_ui.png"
    canvas = ARTIFACTS / "railway_sample_ui_canvas.png"
    with viewer_server() as url:
        summary = capture(url, full, run_sample=True, canvas_output=canvas, timeout_s=900)
    assert summary.tracks >= 1000, f"トラックが少なすぎる: {summary}"
    assert summary.cameras >= 2, f"カメラ姿勢が少なすぎる: {summary}"
    assert_canvas_drawn(canvas)


def test_ui_opens_saved_result():
    """保存済み結果をビューアで開き、シーンが可視化されること。"""
    scene = REPO / "results" / "railway.npz"
    if not scene.is_file():
        pytest.skip("results/railway.npz がない（先に CLI 推論を実行する）")
    full = ARTIFACTS / "railway_scene_ui.png"
    canvas = ARTIFACTS / "railway_scene_ui_canvas.png"
    with viewer_server(scene) as url:
        summary = capture(url, full, run_sample=False, canvas_output=canvas)
    assert summary.tracks >= 1000, f"トラックが少なすぎる: {summary}"
    assert_canvas_drawn(canvas)
