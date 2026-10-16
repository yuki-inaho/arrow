"""scripts/depth_detail_models.py の実行経路テスト（scene flow CPU並列）。

- 実 sceneflow venv の worker を subprocess で起動する（GPU/モデル推論は使わない）。
- 並列(workers=2)と逐次(workers=1)で pair 出力が一致し、pair一意・順序固定であること。
- pair内例外が nonzero で伝搬し、範囲外入力は1pair目開始前に拒否されること。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO = Path("/home/kasm-user/Desktop/arrow")
SCENEFLOW_PY = Path("/home/kasm-user/Desktop/rgbd_sceneflow_sandbox/.venv/bin/python")
SCENEFLOW_PYTHONPATH = "/home/kasm-user/Desktop/rgbd_sceneflow_sandbox/scene_flow_primitives/src"

H, W = 24, 32
COMPARE_KEYS = ["depth0_refined", "depth1_refined", "scene_flow", "match_flow", "support"]


def build_inputs(root: Path, pairs=((0, 1), (1, 2)), corrupt=None, low: float = 1.0) -> None:
    inputs = root / "stages" / "sceneflow" / "pairs" / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    items = []
    for (a, b) in pairs:
        name = f"pair_{a:06d}_{b:06d}.npz"
        if corrupt == (a, b):
            # preflight(depthキー)は通るが pair 処理内で shape 不一致で失敗する入力
            np.savez_compressed(
                inputs / name,
                rgb0_low=np.zeros((H - 2, W, 3), dtype=np.uint8),  # shape不一致
                rgb1_in_0=np.zeros((H, W, 3), dtype=np.uint8),
                depth0_low=np.ones((H, W), dtype=np.float32),
                depth1_in_0=np.ones((H, W), dtype=np.float32),
                valid0_low=np.ones((H, W), dtype=bool),
                valid1_in_0=np.ones((H, W), dtype=bool),
                K_low=np.eye(3, dtype=np.float64),
                low_size=np.asarray([H, W], dtype=np.int32),
            )
        else:
            rgb0 = (rng.random((H, W, 3)) * 255).astype(np.uint8)
            rgb1 = (rng.random((H, W, 3)) * 255).astype(np.uint8)
            depth = np.linspace(low, low + 1.0, H * W, dtype=np.float64).reshape(H, W)
            np.savez_compressed(
                inputs / name,
                rgb0_low=rgb0, rgb1_low=rgb1, rgb1_in_0=rgb1,
                depth0_low=depth.astype(np.float32),
                depth1_low=depth.astype(np.float32),
                depth1_in_0=depth.astype(np.float32),
                valid0_low=np.ones((H, W), dtype=bool),
                valid1_low=np.ones((H, W), dtype=bool),
                valid1_in_0=np.ones((H, W), dtype=bool),
                K_low=np.array([[16.0, 0.0, 15.5], [0.0, 16.0, 11.5], [0.0, 0.0, 1.0]]),
                K_low_source=np.array([[16.0, 0.0, 15.5], [0.0, 16.0, 11.5], [0.0, 0.0, 1.0]]),
                pair_index=np.asarray([a, b], dtype=np.int64),
                clip_index=np.int32(0),
                low_size=np.asarray([H, W], dtype=np.int32),
            )
        items.append({"pair_index": [a, b], "clip_index": 0, "file": str(inputs / name)})
    (root / "stages" / "sceneflow").mkdir(parents=True, exist_ok=True)
    (root / "stages" / "sceneflow" / "pairs.json").write_text(
        json.dumps({"dataset": "syn", "long_side": 32, "low_size": [H, W], "pairs": items})
    )


def run_refine(root: Path, workers: int, *, min_depth: float = 0.5, max_depth: float = 100.0,
               max_motion: float = 0.2) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{SCENEFLOW_PYTHONPATH}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else SCENEFLOW_PYTHONPATH
    )
    command = [
        str(SCENEFLOW_PY), str(REPO / "scripts" / "depth_detail_models.py"), "sceneflow_refine",
        "--dataset-dir", str(root), "--dataset", "syn",
        "--frames", str(root / "frames.json"),
        "--min-depth", repr(min_depth), "--max-depth", repr(max_depth),
        "--max-motion", repr(max_motion), "--trust-sigma", "4.0", "--workers", str(workers),
    ]
    return subprocess.run(command, cwd=str(REPO), env=env, capture_output=True, text=True)


def load_outputs(root: Path) -> dict[tuple[int, int], dict]:
    outputs = {}
    for path in sorted((root / "stages" / "sceneflow" / "pairs" / "outputs").glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            pair = tuple(int(v) for v in data["pair_index"])
            outputs[pair] = {key: np.asarray(data[key]) for key in COMPARE_KEYS}
    return outputs


pytestmark = pytest.mark.skipif(
    not SCENEFLOW_PY.is_file(), reason="sceneflow venv が無い環境では skip"
)


def test_parallel_workers_match_serial_outputs(tmp_path):
    root1 = tmp_path / "w1"
    build_inputs(root1)
    root2 = tmp_path / "w2"
    shutil.copytree(root1, root2)

    serial = run_refine(root1, 1)
    assert serial.returncode == 0, f"{serial.stdout}\n{serial.stderr}"
    parallel = run_refine(root2, 2)
    assert parallel.returncode == 0, f"{parallel.stdout}\n{parallel.stderr}"

    outputs1 = load_outputs(root1)
    outputs2 = load_outputs(root2)
    assert len(outputs1) == 2 and len(outputs2) == 2, "pair出力が一意・全数であること"
    assert set(outputs1) == set(outputs2) == {(0, 1), (1, 2)}
    for pair, arrays in outputs1.items():
        for key in COMPARE_KEYS:
            other = outputs2[pair][key]
            if arrays[key].dtype == bool:
                assert np.array_equal(arrays[key], other), f"{pair} {key} (bool)"
            else:
                # C++ solver は同一入力・同一configで決定的。丸めはfloat32保存のため tol 1e-6
                assert np.allclose(arrays[key], other, rtol=0, atol=1e-6, equal_nan=True), (
                    f"{pair} {key} が workers=1 と workers=2 で不一致"
                )
    config = json.loads((root2 / "configs" / "stage_configs" / "sceneflow_refine.json").read_text())
    assert config["workers"] == 2
    assert config["input_scan"]["pair_count"] == 2
    assert [entry["pair_index"] for entry in config["pairs"]] == [[0, 1], [1, 2]], "順序固定"


def test_pair_failure_propagates_nonzero(tmp_path):
    root = tmp_path / "bad"
    build_inputs(root, corrupt=(1, 2))
    result = run_refine(root, 2)
    assert result.returncode != 0
    assert "sceneflow_refine" in result.stderr and "[1, 2]" in result.stderr


def test_preflight_rejects_out_of_range_before_first_pair(tmp_path):
    root = tmp_path / "oor"
    build_inputs(root, pairs=((0, 1), (2, 3)), low=0.1)  # 0.1 < min_depth=0.5
    result = run_refine(root, 2, min_depth=0.5)
    assert result.returncode != 0
    assert "preflight" in result.stderr
    outputs_dir = root / "stages" / "sceneflow" / "pairs" / "outputs"
    assert not outputs_dir.is_dir() or not list(outputs_dir.glob("*.npz")), (
        "範囲外入力は1pair目開始前に拒否され、出力を書かない"
    )
