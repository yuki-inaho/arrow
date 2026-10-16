"""``scripts/verify_static_world.py`` 独立検証ツールのテスト。

fixture は ``tests/test_static_world_pipeline.py`` の ``fixture()`` を importlib で
呼んで生成し（fixture 生成のために root consumer を import してよい）、出力は
producer CLI（importlib 経由の ``main()``）で作る。検証コード自体は producer と
``arrow.static_world`` を import しない。

破壊テストは module 共有の pristine 出力を毎回 tmp_path へ copytree し、コピー側だけを
変更する（original は変更しない = save before mutate / restore each）。npz を書き換える
場合は manifest の artifact bytes/hash を再計算して、対象チェックそのものが発火する
ことを確かめる。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
ARTIFACT_SUFFIXES = {".mp4", ".png", ".ply", ".rrd", ".rbl", ".npz"}


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def verifier():
    return _load_module("verify_static_world_module", REPO / "scripts" / "verify_static_world.py")


@pytest.fixture(scope="module")
def producer_module():
    return _load_module("build_static_world_module", REPO / "scripts" / "build_static_world.py")


def _run_producer(producer, input_root, output_root):
    status = producer.main(
        [
            "--input-dir", str(input_root),
            "--output-dir", str(output_root),
            "--dataset", "grass",
            "--model-stride", "1",
            "--splat-radius", "1",
            "--min-free-gib", "0",
            "--rerun-max-points", "1000",
        ]
    )
    assert status == 0


def _write_dynamic_fixture(root, *, frames=3, flow_x=0.13, fx=40.0, base_depth=2.0, slope=0.02):
    """Constant per-frame residual on a sloped depth ramp: tracking gap, then donors."""
    directory = Path(root) / "grass"
    for stage in ("final", "sr", "geometry"):
        (directory / "stages" / stage).mkdir(parents=True)
    (directory / "frames_rgb").mkdir()
    h, w = 24, 32
    K = np.array([[fx, 0.0, 15.5], [0.0, fx, 11.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    records = []
    for i in range(frames):
        depth = (base_depth + slope * np.arange(w, dtype=np.float32))[None, :].repeat(h, axis=0)
        valid = np.ones((h, w), dtype=bool)
        flow = np.zeros((h, w, 3), dtype=np.float32)
        flow[..., 0] = flow_x
        flow_valid = valid.copy() if i < frames - 1 else np.zeros_like(valid)
        np.savez(directory / f"stages/final/{i:06d}.npz", final_depth=depth, final_valid=valid)
        np.savez(directory / f"stages/sr/{i:06d}.npz", sr_flow=flow, sr_valid=flow_valid)
        np.savez(
            directory / f"stages/geometry/{i:06d}.npz", arrow_K=K,
            arrow_camera_to_metric=np.eye(4), arrow_K_valid=True, arrow_pose_valid=True,
        )
        image = np.zeros((h, w, 3), np.uint8)
        image[:] = [120 + i * 10, 200, 40]
        path = directory / f"frames_rgb/{i:06d}.png"
        Image.fromarray(image).save(path)
        records.append({"frame_index": i, "source_index": i, "rgb_path": str(path)})
    (directory / "frames.json").write_text(
        json.dumps({"dataset": "grass", "frame_count": frames, "original_size": [h, w], "frames": records})
    )
    return Path(root)


def _build_dynamic(producer, root, **kwargs):
    input_root = Path(root) / "input"
    output_root = Path(root) / "output"
    _write_dynamic_fixture(input_root, **kwargs)
    _run_producer(producer, input_root, output_root)
    return input_root, output_root


@pytest.fixture(scope="module")
def built(tmp_path_factory, producer_module):
    root = tmp_path_factory.mktemp("static_world_audit")
    input_root = root / "input"
    output_root = root / "output"
    pipeline = _load_module("pipeline_fixture_module", REPO / "tests" / "test_static_world_pipeline.py")
    pipeline.fixture(input_root)
    _run_producer(producer_module, input_root, output_root)
    return input_root, output_root


@pytest.fixture(scope="module")
def dynamic_built(tmp_path_factory, producer_module):
    root = tmp_path_factory.mktemp("dynamic_birth")
    return _build_dynamic(producer_module, root, frames=3, flow_x=0.13)


@pytest.fixture(scope="module")
def large_motion_built(tmp_path_factory, producer_module):
    root = tmp_path_factory.mktemp("large_motion")
    return _build_dynamic(producer_module, root, frames=4, flow_x=0.5, fx=20.0, base_depth=3.0, slope=0.0)


@pytest.fixture()
def copied(built, tmp_path):
    input_root, output_root = built
    copied_output = tmp_path / "output"
    shutil.copytree(output_root, copied_output)
    return input_root, copied_output


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _refresh_artifacts(output_root):
    directory = Path(output_root) / "grass" / "static_world"
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    artifacts = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix in ARTIFACT_SUFFIXES:
            artifacts.append({"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    manifest["artifacts"] = artifacts
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _mutate_npz(path, mutate):
    data = dict(np.load(path, allow_pickle=False))
    mutate(data)
    np.savez_compressed(path, **data)


def _dataset_dir(output_root):
    return Path(output_root) / "grass" / "static_world"


# ---------------------------------------------------------------------------
# 正常系
# ---------------------------------------------------------------------------


def test_verifier_accepts_pristine_fixture(verifier, built):
    input_root, output_root = built
    report = verifier.verify_dataset(input_root, output_root, "grass")

    assert report["ok"], report["failures"]
    assert report["failures"] == []
    counters = report["counters"]
    assert counters["frame_count"] == 3
    assert counters["pair_count"] == 2
    assert counters["model_points"] == 767
    assert counters["frame0_anchors"] == 767
    assert counters["later_anchors"] == 0
    assert counters["rgbd_frames"] == 3
    assert counters["motion_pairs"] == 2
    assert counters["corrected_anchors"] == 0  # 既存 fixture は later anchor が無く補正を exercise しない
    # 手計算の出生位置は float32 直列化誤差のみ、深度と camera-z は一致する。
    assert report["max_errors"]["birth_world"] < 1e-5
    assert report["max_errors"]["rgbd_depth"] <= 1e-5


def test_verifier_accepts_corrected_birth_fixture(verifier, dynamic_built):
    input_root, output_root = dynamic_built
    report = verifier.verify_dataset(input_root, output_root, "grass")

    assert report["ok"], report["failures"]
    counters = report["counters"]
    assert counters["later_anchors"] > 0
    assert counters["corrected_anchors"] > 0
    assert counters["corrected_anchors_checked"] == counters["corrected_anchors"]
    assert counters["birth_envelope_checked"] == counters["corrected_anchors"]
    assert report["max_errors"]["birth_world"] < 1e-5


def test_verifier_accepts_large_legit_motion(verifier, large_motion_built):
    input_root, output_root = large_motion_built
    report = verifier.verify_dataset(input_root, output_root, "grass")

    assert report["ok"], report["failures"]
    # 「動く点は canonical と一致しない」ため hard 上限は存在しない。旧 5*depth_gate/1.0 の
    # 閾値なら誤拒否する大きさの diagnostic が出ることを記録する。
    assert report["max_errors"]["id_world_reprojection"] > 1.0


def _copy_output(dynamic_built, tmp_path):
    _, output_root = dynamic_built
    copied_output = tmp_path / "output"
    shutil.copytree(output_root, copied_output)
    return dynamic_built[0], copied_output


def _first_corrected(model):
    return int(np.flatnonzero(model["correction_support"] >= 4)[0])


def test_detects_wrong_birth_correction(verifier, dynamic_built, tmp_path):
    input_root, copied_output = _copy_output(dynamic_built, tmp_path)
    path = _dataset_dir(copied_output) / "model.npz"
    model = dict(np.load(path, allow_pickle=False))
    corrected = _first_corrected(model)
    model["birth_correction"][corrected] += np.float32(0.05)
    np.savez_compressed(path, **model)
    _refresh_artifacts(copied_output)

    report = verifier.verify_dataset(input_root, copied_output, "grass")
    assert not report["ok"]
    assert any(
        ("birth_correction" in failure) or ("envelope" in failure) or ("first-observed" in failure)
        for failure in report["failures"]
    ), report["failures"]


def test_detects_invalid_correction_support(verifier, dynamic_built, tmp_path):
    input_root, copied_output = _copy_output(dynamic_built, tmp_path)
    path = _dataset_dir(copied_output) / "model.npz"
    model = dict(np.load(path, allow_pickle=False))
    model["correction_support"][_first_corrected(model)] = 3
    np.savez_compressed(path, **model)
    _refresh_artifacts(copied_output)

    report = verifier.verify_dataset(input_root, copied_output, "grass")
    assert not report["ok"]
    assert any("correction_support" in failure for failure in report["failures"]), report["failures"]


def test_detects_zero_support_with_nonzero_correction(verifier, dynamic_built, tmp_path):
    input_root, copied_output = _copy_output(dynamic_built, tmp_path)
    path = _dataset_dir(copied_output) / "model.npz"
    model = dict(np.load(path, allow_pickle=False))
    model["correction_support"][_first_corrected(model)] = 0
    np.savez_compressed(path, **model)
    _refresh_artifacts(copied_output)

    report = verifier.verify_dataset(input_root, copied_output, "grass")
    assert not report["ok"]
    assert any("zero correction_support" in failure for failure in report["failures"]), report["failures"]


def test_verifier_cli_writes_report(verifier, built, tmp_path):
    input_root, output_root = built
    report_path = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable, str(REPO / "scripts" / "verify_static_world.py"),
            "--input-dir", str(input_root),
            "--output-dir", str(output_root),
            "--dataset", "grass",
            "--report", str(report_path),
        ],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(report_path.read_text())
    assert report["ok"] is True
    assert report["datasets"]["grass"]["ok"] is True


def test_verifier_media_checks_on_fixture(verifier, built):
    input_root, output_root = built
    # --check-media は必須指定: ffprobe と installed .venv/bin/rerun が両方必要。
    report = verifier.verify_dataset(input_root, output_root, "grass", check_media=True)
    assert report["ok"], report["failures"]
    media = report["media"]
    assert media["grass_static_rgb.mp4"] == {"frames": 3, "width": 32, "height": 24}
    assert media["grass_static_depth.mp4"] == {"frames": 3, "width": 32, "height": 24}
    assert media["grass_static_rgbd.mp4"] == {"frames": 3, "width": 1260, "height": 384}
    assert media["grass_static_world.rrd"]["returncode"] == 0


def test_check_media_fails_without_required_rerun_binary(verifier, built, tmp_path):
    input_root, output_root = built
    report = verifier.verify_dataset(
        input_root, output_root, "grass", check_media=True, rerun_bin=tmp_path / "no-such-rerun"
    )
    assert not report["ok"]
    assert any("rerun binary" in failure for failure in report["failures"]), report["failures"]


def test_fixture_static_camera_outputs_are_byte_identical(built):
    _, output_root = built
    directory = _dataset_dir(output_root) / "rgbd"
    with np.load(directory / "000000.npz") as first, np.load(directory / "000001.npz") as second:
        for key in ("rgb", "depth", "valid", "model_point_index", "reference_support"):
            np.testing.assert_array_equal(first[key], second[key])


# ---------------------------------------------------------------------------
# 破壊検知（各テストは pristine 出力のコピーだけを変更する）
# ---------------------------------------------------------------------------


def test_detects_corrupted_total_flow(verifier, copied):
    input_root, output_root = copied
    path = _dataset_dir(output_root) / "motion" / "000000.npz"

    def mutate(data):
        data["total_flow"] = data["total_flow"] + np.float32(1.0)

    _mutate_npz(path, mutate)
    _refresh_artifacts(output_root)
    report = verifier.verify_dataset(input_root, output_root, "grass")
    assert not report["ok"]
    assert any("total_flow" in failure for failure in report["failures"]), report["failures"]


def test_detects_corrupted_world_nonrigid(verifier, copied):
    input_root, output_root = copied
    path = _dataset_dir(output_root) / "motion" / "000000.npz"

    def mutate(data):
        data["world_nonrigid"] = data["world_nonrigid"] + np.float32(0.5)

    _mutate_npz(path, mutate)
    _refresh_artifacts(output_root)
    report = verifier.verify_dataset(input_root, output_root, "grass")
    assert not report["ok"]
    assert any("world_nonrigid" in failure for failure in report["failures"]), report["failures"]


def test_detects_wrong_point_depth(verifier, copied):
    input_root, output_root = copied
    path = _dataset_dir(output_root) / "rgbd" / "000001.npz"

    def mutate(data):
        depth = data["depth"]
        depth[data["valid"]] += np.float32(0.25)
        data["depth"] = depth

    _mutate_npz(path, mutate)
    _refresh_artifacts(output_root)
    report = verifier.verify_dataset(input_root, output_root, "grass")
    assert not report["ok"]
    assert any("camera-z" in failure for failure in report["failures"]), report["failures"]


def test_detects_out_of_range_anchor_id(verifier, copied):
    input_root, output_root = copied
    path = _dataset_dir(output_root) / "correspondence" / "000001.npz"

    def mutate(data):
        anchor_id = data["anchor_id"]
        anchor_id.ravel()[int(np.argmax(data["valid"]))] = 999999
        data["anchor_id"] = anchor_id

    _mutate_npz(path, mutate)
    _refresh_artifacts(output_root)
    report = verifier.verify_dataset(input_root, output_root, "grass")
    assert not report["ok"]
    assert any("anchor_id" in failure for failure in report["failures"]), report["failures"]


def test_detects_out_of_range_point_index(verifier, copied):
    input_root, output_root = copied
    path = _dataset_dir(output_root) / "rgbd" / "000001.npz"

    def mutate(data):
        index_map = data["model_point_index"]
        index_map.ravel()[int(np.argmax(data["valid"]))] = 999999
        data["model_point_index"] = index_map

    _mutate_npz(path, mutate)
    _refresh_artifacts(output_root)
    report = verifier.verify_dataset(input_root, output_root, "grass")
    assert not report["ok"]
    assert any("point_index" in failure for failure in report["failures"]), report["failures"]


def test_detects_substituted_color(verifier, copied):
    input_root, output_root = copied
    path = _dataset_dir(output_root) / "rgbd" / "000001.npz"

    def mutate(data):
        rgb = data["rgb"]
        rows, cols = np.nonzero(data["valid"])
        original = rgb[rows[0], cols[0]].copy()
        rgb[rows[0], cols[0]] = original ^ np.uint8(255)
        data["rgb"] = rgb

    _mutate_npz(path, mutate)
    _refresh_artifacts(output_root)
    report = verifier.verify_dataset(input_root, output_root, "grass")
    assert not report["ok"]
    assert any("color" in failure for failure in report["failures"]), report["failures"]


def test_detects_wrong_count(verifier, copied):
    input_root, output_root = copied
    manifest_path = _dataset_dir(output_root) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["frames"][0]["canonical_pixel_count"] += 1
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    report = verifier.verify_dataset(input_root, output_root, "grass")
    assert not report["ok"]
    assert any("canonical_pixel_count" in failure for failure in report["failures"])


def test_detects_missing_frame_file(verifier, copied):
    input_root, output_root = copied
    (_dataset_dir(output_root) / "rgbd" / "000002.npz").unlink()

    report = verifier.verify_dataset(input_root, output_root, "grass")
    assert not report["ok"]
    assert any("rgbd" in failure for failure in report["failures"]), report["failures"]


def test_detects_input_pin_hash_mismatch(verifier, copied):
    input_root, output_root = copied
    manifest_path = _dataset_dir(output_root) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["input_pins"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    report = verifier.verify_dataset(input_root, output_root, "grass")
    assert not report["ok"]
    assert any("input pin sha256" in failure for failure in report["failures"])


def test_detects_substituted_model_sha(verifier, copied):
    input_root, output_root = copied
    path = _dataset_dir(output_root) / "rgbd" / "000001.npz"

    def mutate(data):
        metadata = json.loads(str(data["metadata"]))
        metadata["model_sha256"] = "deadbeef"
        data["metadata"] = np.asarray(json.dumps(metadata))

    _mutate_npz(path, mutate)
    _refresh_artifacts(output_root)
    report = verifier.verify_dataset(input_root, output_root, "grass")
    assert not report["ok"]
    assert any("metadata model_sha256" in failure for failure in report["failures"])


def test_detects_modified_model_colors(verifier, copied):
    input_root, output_root = copied
    path = _dataset_dir(output_root) / "model.npz"

    def mutate(data):
        colors = data["colors"]
        colors[0] = [255 - value for value in colors[0]]
        data["colors"] = colors

    _mutate_npz(path, mutate)
    _refresh_artifacts(output_root)
    report = verifier.verify_dataset(input_root, output_root, "grass")
    assert not report["ok"]
    assert any("sha256" in failure or "RGB color" in failure for failure in report["failures"])


# ---------------------------------------------------------------------------
# 独立性
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("available_donors", [0, 1, 3])
def test_rejects_correction_without_minimum_donors(verifier, available_donors):
    """A claimed four-donor correction must fail even when no median is reproducible."""
    labels = np.full((17, 17), -1, dtype=np.int64)
    for row, col in [(7, 7), (7, 8), (7, 9)][:available_donors]:
        labels[row, col] = 0
    issues = verifier.Issues()
    verifier._verify_birth_donors(
        1, np.array([0, 1]), np.array([0, 8 * 17 + 8]),
        np.zeros((2, 3)), np.zeros((2, 3)), np.array([0, 4]),
        labels, np.ones((17, 17)), np.zeros((17, 17, 3), dtype=np.uint8),
        np.eye(3), np.eye(4), issues, {}, {},
    )
    assert any("fewer donors" in message for message in issues.failures)


def test_verifier_does_not_import_producer_or_core():
    text = (REPO / "scripts" / "verify_static_world.py").read_text()
    assert "from arrow" not in text
    assert "import arrow" not in text
    assert "spec_from_file_location" not in text
    assert "importlib" not in text
