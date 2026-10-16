"""scripts/infer_bullettime.py (BulletTime 推論アダプタ) の契約テスト。

方針:
  - 期待値は実装のミラーではなく、upstream commit e0331619 のコードと手計算から
    独立に定めた既知数を使う（plucker の identity/translation/rotation、
    latent padding の具体値、時間順序と無変換契約）。
  - 形状・framecount・K 一定性・target 一定性などの違反は fail することを確認する。
  - strict checkpoint loader は「未初期化重みの黙認禁止」を、値まで含めて確認する。

実行（隔離環境）:
  temp/grass_static_review/env/venv/bin/python -m pytest tests/test_bullettime_adapter.py -q
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = REPO_ROOT / "scripts" / "infer_bullettime.py"
UPSTREAM_DEFAULT = REPO_ROOT / "temp" / "grass_static_review" / "BulletTime"


def _load_adapter():
    spec = importlib.util.spec_from_file_location("infer_bullettime_under_test", ADAPTER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # dataclass が cls.__module__ を sys.modules から引くため、exec 前に登録する。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


adapter = _load_adapter()


def test_reference_encoding_uses_execution_device_before_offload_hook():
    calls = []

    class Video:
        def to(self, *, device, dtype):
            calls.append((device, dtype))
            return "gpu_input"

    class Vae:
        device = "cpu"  # Model is offloaded when encode_video starts.
        dtype = torch.bfloat16
        config = SimpleNamespace(scaling_factor=2.0)

        def encode(self, video):
            assert video == "gpu_input"
            return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: torch.tensor([3.])))

    pipe = SimpleNamespace(vae=Vae(), _execution_device="cuda:0")
    result = adapter.encode_reference_with_offload(pipe, Video())
    assert calls == [("cuda:0", torch.bfloat16)]
    assert result.tolist() == [6.0]

CENTER_K = torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])


def _identity_pose() -> torch.Tensor:
    return torch.eye(4).unsqueeze(0)


def _controls_npz(
    tmp_path: Path,
    *,
    T: int = 9,
    H: int = 32,
    W: int = 32,
    rgb: np.ndarray | None = None,
    c2w: np.ndarray | None = None,
    K: np.ndarray | None = None,
    source_time: np.ndarray | None = None,
    target_time: np.ndarray | None = None,
    fps: np.ndarray | float = np.float32(5.0),
    drop: str | None = None,
) -> Path:
    if rgb is None:
        rgb = (np.arange(T * H * W * 3, dtype=np.uint8) % 251).reshape(T, H, W, 3)
    if c2w is None:
        c2w = np.tile(np.eye(4, dtype=np.float64), (T, 1, 1))
    if K is None:
        K = np.tile(np.array([[16.0, 0.0, 15.5], [0.0, 16.0, 15.5], [0.0, 0.0, 1.0]]), (T, 1, 1))
    if source_time is None:
        source_time = (np.arange(T, dtype=np.float32) / np.float32(30.0)).astype(np.float32)
    if target_time is None:
        target_time = np.zeros(T, dtype=np.float32)
    payload = {
        "rgb": rgb,
        "camera_to_world": c2w,
        "K": K,
        "source_time": source_time,
        "target_time": target_time,
        "fps": np.asarray(fps, dtype=np.float32),
    }
    if drop is not None:
        payload.pop(drop)
    path = tmp_path / "controls.npz"
    np.savez(path, **payload)
    return path


# ---------------------------------------------------------------------------
# Plucker 既知数（identity / translation / rotation）
# ---------------------------------------------------------------------------


def test_plucker_identity_center_pixel_and_corner_hand_values():
    plucker = adapter.compute_plucker_ray_map(CENTER_K, _identity_pose(), 2, 2)
    assert plucker.shape == (1, 2, 2, 6)
    center = plucker[0, 0, 0]
    # 画素 (0,0) の中心 (u,v)=(0.5,0.5) は cx=cy=0.5, fx=fy=1 で (0,0,1) 方向。
    assert torch.allclose(center[:3], torch.tensor([0.0, 0.0, 1.0]), atol=1e-7)
    assert torch.allclose(center[3:], torch.zeros(3), atol=1e-7)

    corner = plucker[0, 1, 1]
    expected_dir = torch.tensor([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    assert torch.allclose(corner[:3], expected_dir, atol=1e-6)
    assert torch.allclose(corner[3:], torch.zeros(3), atol=1e-6)


def test_plucker_translation_moves_moments():
    pose = torch.eye(4).unsqueeze(0)
    pose[0, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
    plucker = adapter.compute_plucker_ray_map(CENTER_K, pose, 2, 2)
    # 中心画素 d=(0,0,1) に対し moment = t x d = (2,-1,0)（手計算）。
    assert torch.allclose(plucker[0, 0, 0, 3:], torch.tensor([2.0, -1.0, 0.0]), atol=1e-6)
    # 角画素 d=(1,1,1)/sqrt(3): moment = t x d = (-1,2,-1)/sqrt(3)（手計算）。
    expected_moment = torch.tensor([-1.0, 2.0, -1.0]) / np.sqrt(3.0)
    assert torch.allclose(plucker[0, 1, 1, 3:], expected_moment, atol=1e-6)


def test_plucker_rotation_90z_rotates_directions():
    pose = torch.eye(4).unsqueeze(0)
    pose[0, :3, :3] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pose[0, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
    plucker = adapter.compute_plucker_ray_map(CENTER_K, pose, 2, 2)
    corner = plucker[0, 1, 1]
    expected_dir = torch.tensor([-1.0, 1.0, 1.0]) / np.sqrt(3.0)  # Rz90 @ (1,1,1)/sqrt3
    expected_moment = torch.tensor([-1.0, -4.0, 3.0]) / np.sqrt(3.0)  # t x d （手計算）
    assert torch.allclose(corner[:3], expected_dir, atol=1e-6)
    assert torch.allclose(corner[3:], expected_moment, atol=1e-6)


def test_normalize_camera_poses_first_pose_identity_and_relative_pose():
    angle = np.deg2rad(90.0)
    rot0 = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[0, :3, :3] = rot0
    poses[0, :3, 3] = [1.0, 1.0, 1.0]
    poses[1] = poses[0] @ np.array([[1, 0, 0, 0.5], [0, 1, 0, 0.0], [0, 0, 1, 0.25], [0, 0, 0, 1.0]])
    poses[2] = poses[0].copy()
    c2ws = torch.tensor(poses, dtype=torch.float64)
    normalized = adapter.normalize_camera_poses(c2ws, c2ws[0])
    assert torch.allclose(normalized[0], torch.eye(4, dtype=torch.float64), atol=1e-9)
    expected = np.linalg.inv(poses[0]) @ poses[1]  # 独立計算
    assert torch.allclose(normalized[1], torch.tensor(expected), atol=1e-9)
    assert torch.allclose(normalized[2], torch.eye(4, dtype=torch.float64), atol=1e-9)


# ---------------------------------------------------------------------------
# 入り口契約: sampling / 順序 / rescale / 形状
# ---------------------------------------------------------------------------


def test_sample_indices_matches_upstream_arange_step4():
    assert adapter.sample_indices(53).tolist() == list(range(0, 53, 4))
    assert adapter.sample_indices(53).numel() == 14  # N = (53-1)//4+1
    assert adapter.sample_indices(49).tolist() == list(range(0, 49, 4))
    assert adapter.sample_indices(49).numel() == 13


def test_build_camera_inputs_shapes_order_and_outdoor_rescale(tmp_path):
    T = 9
    positions = np.tile(np.eye(4), (T, 1, 1))
    for frame in range(T):
        positions[frame, :3, 3] = [0.0, 15.0 + frame, 30.0 + frame]  # first norm ~33.5 (>20)
    K = np.tile(np.array([[16.0, 0.0, 7.5], [0.0, 16.0, 7.5], [0.0, 0.0, 1.0]]), (T, 1, 1))
    path = _controls_npz(tmp_path, T=T, H=16, W=16, c2w=positions, K=K)
    controls = adapter.load_controls(path)
    camera, camera_params, info = adapter.build_camera_inputs(controls)

    assert info["scene_rescale_applied"] is True
    assert info["n_poses"] == 3  # sampled [0,4,8]
    assert camera.shape == (1, 6, 16, 16, 6)
    assert camera_params["tgt_c2ws"].shape == (3, 4, 4)
    assert camera_params["ref_c2ws"].shape == (3, 4, 4)
    assert camera_params["tgt_intrinsics"].shape == (3, 3)
    assert camera_params["image_width"] == 16 and camera_params["image_height"] == 16

    # 同一軌跡契約: target 前半と reference 後半は一致。
    assert torch.equal(camera[:, :3], camera[:, 3:])
    # 正規化後 first pose は恒等。
    assert torch.allclose(camera_params["tgt_c2ws"][0], torch.eye(4), atol=1e-5)

    # resample 後 pose1 (frame4) の原点 = (t4 - t0)/10 = (0, 0.4, 0.4)（手計算）。
    expected_translation = torch.tensor([0.0, 0.4, 0.4])
    assert torch.allclose(camera_params["tgt_c2ws"][1, :3, 3], expected_translation, atol=1e-5)
    assert torch.allclose(camera_params["ref_c2ws"][1, :3, 3], expected_translation, atol=1e-5)

    # plucker 画素 (0,0): d と moment = cross(t, d) を K から独立に手計算。
    d = np.array([-7.5 / 16.0, -7.5 / 16.0, 1.0])
    d = d / np.linalg.norm(d)
    expected_moment = np.cross(np.array([0.0, 0.4, 0.4]), d)
    assert torch.allclose(camera[0, 4, 0, 0, :3], torch.tensor(d, dtype=torch.float32), atol=1e-5)
    assert torch.allclose(
        camera[0, 4, 0, 0, 3:], torch.tensor(expected_moment, dtype=torch.float32), atol=1e-5
    )


def test_integer_pixel_intrinsics_reach_both_model_consumers(tmp_path):
    K = np.tile(np.array([[16., 0., 0.], [0., 16., 0.], [0., 0., 1.]]), (9, 1, 1))
    controls = adapter.load_controls(_controls_npz(tmp_path, K=K))
    camera, params, info = adapter.build_camera_inputs(controls)
    # Pixel (0,0) lies on the optical axis in the integer-center input.
    assert torch.equal(camera[0, 0, 0, 0, :3], torch.tensor([0., 0., 1.]))
    assert params['tgt_intrinsics'][0, 2].item() == 0.5
    assert params['ref_intrinsics'][1, 2].item() == 0.5
    assert np.array_equal(controls.K, K)
    assert info['K_integer_center_input'][0][2] == 0.0


def test_build_camera_inputs_no_rescale_within_threshold(tmp_path):
    T = 9
    positions = np.tile(np.eye(4), (T, 1, 1))
    for frame in range(T):
        positions[frame, :3, 3] = [0.0, 1.5 + frame, 3.0 + frame]  # first norm ~3.35 (<=20)
    K = np.tile(np.array([[16.0, 0.0, 7.5], [0.0, 16.0, 7.5], [0.0, 0.0, 1.0]]), (T, 1, 1))
    path = _controls_npz(tmp_path, T=T, H=16, W=16, c2w=positions, K=K)
    controls = adapter.load_controls(path)
    camera, camera_params, info = adapter.build_camera_inputs(controls)
    assert info["scene_rescale_applied"] is False
    # 無 rescale: pose1 (frame4) の原点 = t4 - t0 = (0, 4, 4)（手計算）。
    expected_translation = torch.tensor([0.0, 4.0, 4.0])
    assert torch.allclose(camera_params["tgt_c2ws"][1, :3, 3], expected_translation, atol=1e-5)
    d = np.array([-7.5 / 16.0, -7.5 / 16.0, 1.0])
    d = d / np.linalg.norm(d)
    expected_moment = np.cross(np.array([0.0, 4.0, 4.0]), d)
    assert torch.allclose(
        camera[0, 4, 0, 0, 3:], torch.tensor(expected_moment, dtype=torch.float32), atol=1e-5
    )


# ---------------------------------------------------------------------------
# 時間契約と latent padding（upstream 具体値）
# ---------------------------------------------------------------------------


def test_time_steps_order_and_second_unit_passthrough(tmp_path):
    T = 5
    source = (np.arange(T, dtype=np.float32) / np.float32(30.0)).astype(np.float32)
    path = _controls_npz(tmp_path, T=T, source_time=source)
    controls = adapter.load_controls(path)

    time_steps, info = adapter.build_time_steps(controls, "freeze")
    assert time_steps.shape == (1, 2 * T)
    assert torch.equal(time_steps[0, :T], torch.zeros(T))  # 前半 = target（zero 定数）
    assert torch.equal(time_steps[0, T:], torch.tensor(source))  # 後半 = source 無変換
    assert info["mode"] == "freeze"

    control, _ = adapter.build_time_steps(controls, "source_time")
    assert torch.equal(control[0, :T], torch.tensor(source))
    assert torch.equal(control[0, T:], torch.tensor(source))
    # 無変換（×30 しない）: 例 3/30 がそのままであることを厳密比較で確認。
    assert control[0, 3].item() == source[3].item()


def test_freeze_requires_constant_target_time(tmp_path):
    T = 5
    target = np.arange(T, dtype=np.float32) / np.float32(30.0)
    path = _controls_npz(tmp_path, T=T, target_time=target)
    controls = adapter.load_controls(path)
    with pytest.raises(adapter.ControlError, match="constant target_time"):
        adapter.build_time_steps(controls, "freeze")


def test_time_unit_warning_flags_frame_indices(tmp_path):
    T = 53
    frames = np.arange(T, dtype=np.float32)  # index そのもの → 30 倍時間の疑い
    path = _controls_npz(tmp_path, T=T, source_time=frames)
    controls = adapter.load_controls(path)
    assert any("frame indices" in warning for warning in controls.warnings)

    seconds = frames / np.float32(30.0)
    path2 = _controls_npz(tmp_path, T=T, source_time=seconds)
    controls2 = adapter.load_controls(path2)
    assert not any("frame indices" in warning for warning in controls2.warnings)


def test_latent_padding_contract_hand_values():
    # T=4k+1 に対し N=(T-1)//4+1, additional=(2-N%2)%2, ncopy=N%2, L=(N+a)+(N+n)
    expected = {
        9: (3, 1, 1, 8),
        13: (4, 0, 0, 8),
        17: (5, 1, 1, 12),
        49: (13, 1, 1, 28),
        53: (14, 0, 0, 28),
    }
    for T, (n_poses, additional, ncopy, total) in expected.items():
        info = adapter.latent_padding_info(T, patch_size_t=2)
        assert info["N_sampled_poses"] == n_poses, T
        assert info["additional_gen_latent_tail"] == additional, T
        assert info["ncopy_ref_latent_head"] == ncopy, T
        assert info["L_total_hidden_frames"] == total, T
        assert info["camera_frames_passed"] == 2 * n_poses, T


def test_rgb_to_reference_video_range_and_channel_order(tmp_path):
    T, H, W = 5, 16, 16
    rgb = np.zeros((T, H, W, 3), dtype=np.uint8)
    rgb[:, :, :, 0] = 255  # red
    rgb[0, 0, 0] = [0, 128, 255]
    tensor = adapter.rgb_to_reference_video(rgb)
    assert tensor.shape == (1, 3, T, H, W)
    assert tensor.dtype == torch.float32
    assert tensor[0, 0].max().item() == 1.0 and tensor[0, 0].min().item() == -1.0
    # 128 -> 2*128/255 - 1（手計算）
    assert tensor[0, 1, 0, 0, 0].item() == pytest.approx(2.0 * 128.0 / 255.0 - 1.0, abs=1e-7)
    assert tensor[0, 2].max().item() == 1.0  # blue channel preserved


# ---------------------------------------------------------------------------
# controls 検証（形状・framecount・K・fps）
# ---------------------------------------------------------------------------


def test_controls_rejects_varying_k(tmp_path):
    T = 9
    K = np.tile(np.eye(3), (T, 1, 1))
    K[4, 0, 0] = 1.5  # 5% ずれ
    path = _controls_npz(tmp_path, T=T, K=K)
    with pytest.raises(adapter.ControlError, match="K varies"):
        adapter.load_controls(path)


def test_controls_rejects_bad_framecount(tmp_path):
    path = _controls_npz(tmp_path, T=50)  # 4k+1 でない
    with pytest.raises(adapter.ControlError, match=r"4k\+1"):
        adapter.load_controls(path)


def test_controls_rejects_non_multiple_16(tmp_path):
    path = _controls_npz(tmp_path, H=15)
    with pytest.raises(adapter.ControlError, match="multiples of 16"):
        adapter.load_controls(path)


def test_controls_rejects_non_uint8_rgb(tmp_path):
    T, H, W = 9, 16, 16
    rgb = np.zeros((T, H, W, 3), dtype=np.float32)
    path = _controls_npz(tmp_path, rgb=rgb)
    with pytest.raises(adapter.ControlError, match="uint8"):
        adapter.load_controls(path)


def test_controls_rejects_missing_key(tmp_path):
    path = _controls_npz(tmp_path, drop="K")
    with pytest.raises(adapter.ControlError, match="missing keys"):
        adapter.load_controls(path)


def test_controls_rejects_framecount_mismatch(tmp_path):
    T = 9
    c2w = np.tile(np.eye(4), (T + 4, 1, 1))
    path = _controls_npz(tmp_path, T=T, c2w=c2w)
    with pytest.raises(adapter.ControlError, match="camera_to_world"):
        adapter.load_controls(path)


def test_controls_rejects_bad_k_shape(tmp_path):
    T = 9
    K = np.tile(np.eye(2), (T, 1, 1))
    path = _controls_npz(tmp_path, T=T, K=K)
    with pytest.raises(adapter.ControlError, match="K must be"):
        adapter.load_controls(path)


def test_controls_rejects_nonpositive_fps(tmp_path):
    path = _controls_npz(tmp_path, fps=np.float32(0.0))
    with pytest.raises(adapter.ControlError, match="fps"):
        adapter.load_controls(path)


# ---------------------------------------------------------------------------
# strict checkpoint loader（未初期化重みの黙認禁止）
# ---------------------------------------------------------------------------


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)

    def forward(self, x):  # pragma: no cover - テストでは未使用
        return self.fc(x)


def _save_shard(path: Path, tensors: dict) -> None:
    from safetensors.torch import save_file

    save_file(tensors, str(path))


def test_strict_loader_loads_module_prefixed_values(tmp_path):
    model = _TinyModel()
    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.bias.zero_()
    saved_weight = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    saved_bias = torch.tensor([1.0, -1.0])
    _save_shard(
        tmp_path / "model.safetensors",
        {"module.fc.weight": saved_weight, "module.fc.bias": saved_bias},
    )
    stats = adapter.strict_load_transformer(model, tmp_path)
    assert stats["loaded_keys"] == 2 and stats["model_keys"] == 2
    assert torch.equal(model.fc.weight, saved_weight)  # 値まで入っている
    assert torch.equal(model.fc.bias, saved_bias)


def test_strict_loader_rejects_missing_key(tmp_path):
    model = _TinyModel()
    _save_shard(tmp_path / "model.safetensors", {"fc.weight": torch.zeros(2, 2)})
    with pytest.raises(adapter.CheckpointError, match="missing 1"):
        adapter.strict_load_transformer(model, tmp_path)


def test_strict_loader_rejects_unexpected_key(tmp_path):
    model = _TinyModel()
    _save_shard(
        tmp_path / "model.safetensors",
        {
            "fc.weight": torch.zeros(2, 2),
            "fc.bias": torch.zeros(2),
            "other.weight": torch.zeros(2, 2),
        },
    )
    with pytest.raises(adapter.CheckpointError, match="unexpected 1"):
        adapter.strict_load_transformer(model, tmp_path)


def test_strict_loader_rejects_shape_mismatch(tmp_path):
    model = _TinyModel()
    _save_shard(
        tmp_path / "model.safetensors",
        {"fc.weight": torch.zeros(4, 1), "fc.bias": torch.zeros(2)},
    )
    with pytest.raises(adapter.CheckpointError, match="shape_mismatch 1"):
        adapter.strict_load_transformer(model, tmp_path)


def test_strict_loader_reads_sharded_index(tmp_path):
    model = _TinyModel()
    weight = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    bias = torch.tensor([0.5, 0.25])
    _save_shard(tmp_path / "model-00001.safetensors", {"fc.weight": weight})
    _save_shard(tmp_path / "model-00002.safetensors", {"fc.bias": bias})
    index = {
        "metadata": {"total_size": 24},
        "weight_map": {"fc.weight": "model-00001.safetensors", "fc.bias": "model-00002.safetensors"},
    }
    (tmp_path / "model.safetensors.index.json").write_text(str(index).replace("'", '"'))
    stats = adapter.strict_load_transformer(model, tmp_path)
    assert stats["shard_count"] == 2 and stats["loaded_keys"] == 2
    assert torch.equal(model.fc.weight, weight) and torch.equal(model.fc.bias, bias)


# ---------------------------------------------------------------------------
# 公式 import（training stack 非ロード）
# ---------------------------------------------------------------------------


def test_official_modules_import_bypasses_training_stack():
    upstream = Path(os.environ.get("BULLETTIME_UPSTREAM", str(UPSTREAM_DEFAULT)))
    if not (upstream / adapter.CAM_SUBDIR).is_dir():
        pytest.skip(f"upstream checkout not available: {upstream}")

    official = adapter.load_official_modules(upstream)
    assert "finetune" not in sys.modules  # finetune/models/__init__.py を踏んでいない
    for path_str in official.module_files.values():
        assert Path(path_str).parent == (upstream / adapter.CAM_SUBDIR).resolve()

    pipeline_module = importlib.import_module(
        adapter.NAMESPACE_PACKAGE + ".pipeline_cogvideox_adaln_rope_time_cam"
    )
    # pipeline 内部の isinstance 判定と同じクラスであること（DPM 分岐の前提）。
    assert pipeline_module.CogVideoXDPMScheduler is official.scheduler_cls
    assert "adaln_rope_time_cam" in official.transformer_cls.__module__
    assert official.pipeline_cls.__module__.startswith(adapter.NAMESPACE_PACKAGE)
