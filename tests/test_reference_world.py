"""Reference-only exports retain moving surfaces and never admit later geometry."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    root = tmp_path_factory.mktemp("reference_world")
    audit = load("reference_test_audit", ROOT / "tests/test_static_world_verifier.py")
    producer = load("reference_test_producer", ROOT / "scripts/build_static_world.py")
    source, output = audit._build_dynamic(producer, root, frames=3, flow_x=0.13)
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        reference = load("reference_test_export", ROOT / "scripts/build_reference_world.py")
    finally:
        sys.path.pop(0)
    assert reference.main(["--input-dir", str(source), "--source-dir", str(output),
                           "--dataset", "grass", "--min-free-gib", "0"]) == 0
    verifier = load("reference_test_verify", ROOT / "scripts/verify_static_world.py")
    return source, output, verifier


def test_reference_preserves_all_frame0_shapes_without_late_fusion(exported):
    source, output, verifier = exported
    with np.load(output / "grass/static_world/model.npz") as parent:
        assert (parent["birth_frame"] > 0).any()
        expected = parent["points"][parent["birth_frame"] == 0]
    with np.load(output / "grass/reference_world/model.npz") as model:
        np.testing.assert_array_equal(model["points"], expected)
        assert len(model["points"]) > 0
    report = verifier.verify_reference_dataset(source, output, "grass", check_media=True,
                                               rerun_bin=ROOT / ".venv/bin/rerun")
    assert report["ok"], report["failures"]
    assert report["counters"]["rgbd_frames"] == 3
    a = output / "grass/reference_world/rgbd"
    with np.load(a / "000000.npz") as first, np.load(a / "000002.npz") as last:
        np.testing.assert_array_equal(first["rgb"], last["rgb"])
        np.testing.assert_array_equal(first["depth"], last["depth"])


def test_reference_verifier_rejects_depth_change_after_hash_refresh(exported, tmp_path):
    source, original, verifier = exported
    output = tmp_path / "output"
    target = output / "grass/reference_world"
    shutil.copytree(original / "grass/reference_world", target)
    path = target / "rgbd/000001.npz"
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["depth"][arrays["valid"]] += 0.1
    np.savez_compressed(path, **arrays)
    manifest = json.loads((target / "manifest.json").read_text())
    for item in manifest["artifacts"]:
        p = target / item["path"]
        item.update(bytes=p.stat().st_size, sha256=verifier.sha256_file(p))
    (target / "manifest.json").write_text(json.dumps(manifest))
    report = verifier.verify_reference_dataset(source, output, "grass")
    assert not report["ok"]
    assert any("camera-z" in message for message in report["failures"])


def test_reference_projection_uses_float64_before_pixel_rounding(exported):
    """Real near-half-pixel counterexample: float32 inversion rounds y to 438 instead of 437."""
    _, _, verifier = exported
    K = np.array([[961.8662719726562, 0, 359.70953369140625],
                  [0, 1121.3387451171875, 639.71044921875], [0, 0, 1]], dtype=np.float32)
    M = np.array([[0.9998472332954407, -0.012349291704595089, 0.01236974447965622, -0.04420299082994461],
                  [0.012395717203617096, 0.9999163746833801, -0.0036843358539044857, 0.01457926444709301],
                  [-0.012323214672505856, 0.00383710372261703, 0.999916672706604, 0.004363355226814747],
                  [0, 0, 0, 1]], dtype=np.float32)
    points = np.array([[0.1516454964876175, -0.6851125359535217, 3.8118252754211426]])
    inverse = np.linalg.inv(M.astype(np.float64))
    camera = points @ inverse[:3, :3].T + inverse[:3, 3]
    size = (440, 400)
    valid = np.zeros(size, bool)
    valid[436, 394] = True
    depth = np.full(size, np.nan, np.float32)
    depth[valid] = camera[0, 2]
    indices = np.full(size, -1, np.int32)
    indices[valid] = 0
    rgbd = {"rgb": np.zeros((*size, 3), np.uint8), "depth": depth, "valid": valid,
            "model_point_index": indices, "reference_support": valid, "K": K,
            "camera_to_world": M, "frame_index": 1, "source_index": 1,
            "metadata": np.asarray(json.dumps({"units": "arrow_provisional", "metric_calibrated": False,
                "depth_kind": "camera_z", "world_is_static": True, "model_sha256": "fixture"}))}
    issues = verifier.Issues()
    verifier._verify_rgbd(1, {"source_index": 1}, rgbd, K, M, points, np.zeros((1, 3), np.uint8),
                         np.zeros(1, np.int32), 1, size, 2, "fixture", issues, {})
    assert not issues.failures
