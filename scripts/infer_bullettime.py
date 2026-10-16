#!/usr/bin/env python3
"""BulletTime 推論アダプタ (bounded assignment, nonce 4cefffabc80b)。

正本:
  - temp/grass_static_review/BulletTime  commit e0331619d74d09ea149874cbde145a3a0a3faf7a
  - temp/grass_static_review/engine.md (bounded assignment)
  - temp/workdoc_Oct16-2026_guided_freeze.md

本スクリプトは公式 finetune 実装の次のクラスを「そのまま」使用する推論専用アダプタ:
  finetune/models/cogvideox1_5_cam/cogvideox_transformer_3d_adaln_rope_time_cam.py
  finetune/models/cogvideox1_5_cam/pipeline_cogvideox_adaln_rope_time_cam.py
  finetune/models/cogvideox1_5_cam/scheduling_dpm_cogvideox.py

import 戦略（訓練スタックの回避）:
  ``finetune.models.__init__`` は import 時に cogvideox1_5_cam を含む全サブモジュールを
  eager import し、lora_trainer -> finetune.trainer -> DeepSpeed 類まで引き込む。
  これを避けるため importlib で ``__path__`` だけを持つ名前空間パッケージを作り、
  相対 import を解決させる（training 依存を一切ロードしない）。詳細は
  ``load_official_modules`` を参照。

入力契約（公式 validate 経路 run_eval / get_inference_data_sample(real_world) と同一）:
  controls NPZ (pickle 禁止, allow_pickle=False):
    rgb             uint8  [T, H, W, 3]        T = 4k+1, H/W は 16 の倍数
    camera_to_world float64[T, 4, 4]          camera-to-world (c2w)。source/target 共通の
                                              実測 source 軌跡。crop/resize しない
    K               float64[T, 3, 3]          単一ストリーム前提。全 frame 一定であることを
                                              厳密検査（varying K は黙って捨てない）
    source_time     float32[T]                公式時間規約: 秒 = frame_index / 30
    target_time     float32[T]                freeze 用の一定値（契約は定数 zero）
    fps             scalar float              MP4 書き出し fps。時間値の変換には使わない

  公式との対応（e033161 時点で実コードを確認して確定した規約）:
    - カメラは ``extract_cam_realworld`` と同じく ``arange(T)[::4]`` で N=(T-1)//4+1 本に
      間引き、first pose の翻訳 norm が 20 超なら公式 outdoor 規則どおり全翻訳を /10 し、
      「reference(=source) の first pose」を基準に source/target 双方を joint 正規化する。
    - plucker は cam_utils.compute_plucker_ray_map と同一式で full pixel 解像度 HxW に対し
      生成し、``cat([tgt, ref], dim=0)`` の順（前半=target, 後半=reference）で渡す。
    - time_steps は ``cat([target_time, source_time])`` の [1, 2T]（公式 run_eval と同じ順）。
      公式 rope は fps=30 を内部で乗算するため値は「秒」で渡す（load_time の既定も
      arange(T)/30）。本アダプタは NPZ 値を無変換で渡す。
    - latent の時系列 padding は公式 pipeline/transformer 内部が行う（追加生成 latent は
      patch_size_t まで末尾増、reference latent は先頭 frame コピーを ncopy = N % p_t だけ
      前置）。本アダプタは camera plucker を厳密に 2N frame で渡し、内部 pad と同期させる。

重み読み込み（strict）:
  transformer は base model の transformer/config.json から from_config で構築し、
  公式 trainer と同じく ``_init_additional_weights`` で camera/time モジュールを付与した後、
  BulletTime safetensors (checkpoint/fp32_ckpt または direct dir) を読み込む。
  公式 fallback loader は missing/shape mismatch を黙認するが、本アダプタは
  missing / unexpected / shape mismatch のいずれも fail させる（未初期化重みの黙認禁止）。

実行規約:
  - device は CUDA 必須（生成時）。BF16 + ``enable_model_cpu_offload`` + VAE tiling/slicing。
    全モジュールを .to(cuda) する blanket move はしない（公式 validate と同じ手順）。
  - ``--dry-run`` は checkpoint の重みを load せず、形状・規約・設定・依存の報告のみ行う。
  - 生成物は raw ネットワーク出力（PNG/MP4/manifest）のみ。合成や source copy、metric RGBD
    表記は行わない。出力ディレクトリが既に非空なら fail。

使い方::

    # dry-run（checkpoint 非ロード）
    temp/grass_static_review/env/venv/bin/python scripts/infer_bullettime.py \
        --upstream-dir temp/grass_static_review/BulletTime \
        --base-model temp/grass_static_review/models/cogvideo \
        --checkpoint temp/grass_static_review/models/bullettime \
        --controls <controls.npz> --mode freeze --dry-run

    # 実生成（root が実モデルで実行）
    temp/grass_static_review/env/venv/bin/python scripts/infer_bullettime.py \
        --upstream-dir ... --base-model ... --checkpoint ... \
        --controls <controls.npz> --output-dir <out> --steps 50 --seed 42 \
        --guidance 6 --mode freeze
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import subprocess
import sys
import time
import traceback
import types
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.version

UPSTREAM_COMMIT = "e0331619d74d09ea149874cbde145a3a0a3faf7a"
CAM_SUBDIR = "finetune/models/cogvideox1_5_cam"
NAMESPACE_PACKAGE = "bullettime_official_cam"
OFFICIAL_TIME_FPS = 30.0  # get_3d_rotary_pos_embed_with_time 内の hardcoded fps
SCENE_RESCALE_THRESHOLD = 20.0  # extract_cam_realworld / extract_cam_simulator
SCENE_RESCALE_FACTOR = 10.0
K_CONST_RTOL = 1e-4
K_CONST_ATOL = 1e-3
TARGET_TIME_CONST_ATOL = 1e-6
VAE_SPATIAL_FACTOR_FALLBACK = 8  # 2^(len(vae.block_out_channels)-1) for CogVideoX VAE

# 説明文は「草地/アスファルト/駐車線/コンクリートブロック」を明示。negative には
# static / still picture 等の静止語を入れない（freeze が目的のため）。
DEFAULT_PROMPT = (
    "A realistic outdoor scene of a green grass lawn beside an asphalt parking area, "
    "white painted parking lines on the asphalt, a concrete block at the edge of the "
    "pavement, natural daylight, lifelike textures, high detail."
)

CONTROL_KEYS = {"rgb", "camera_to_world", "K", "source_time", "target_time", "fps"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AdapterError(RuntimeError):
    """アダプタの契約違反（fail-fast, 黙認しない）。"""


class ControlError(AdapterError):
    """controls NPZ の契約違反。"""


class CheckpointError(AdapterError):
    """checkpoint の key/shape 契約違反。"""


class UpstreamImportError(AdapterError):
    """公式コードの import 契約違反。"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# controls NPZ
# ---------------------------------------------------------------------------


@dataclass
class Controls:
    path: Path
    sha256: str
    rgb: np.ndarray
    camera_to_world: np.ndarray
    K: np.ndarray
    source_time: np.ndarray
    target_time: np.ndarray
    fps: float
    warnings: list[str] = field(default_factory=list)

    @property
    def T(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def H(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def W(self) -> int:
        return int(self.rgb.shape[2])


def load_controls(path: Path) -> Controls:
    """controls NPZ を strict に解析する（allow_pickle=False）。"""
    path = Path(path)
    if not path.is_file():
        raise ControlError(f"controls not found: {path}")
    warnings: list[str] = []
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        missing = CONTROL_KEYS - keys
        if missing:
            raise ControlError(f"controls missing keys: {sorted(missing)} (have {sorted(keys)})")
        extra = keys - CONTROL_KEYS
        if extra:
            warnings.append(f"controls has unexpected extra keys (ignored): {sorted(extra)}")

        rgb = data["rgb"]
        if rgb.dtype != np.uint8:
            raise ControlError(f"rgb must be uint8, got {rgb.dtype}")
        if rgb.ndim != 4 or rgb.shape[3] != 3:
            raise ControlError(f"rgb must be [T,H,W,3], got {rgb.shape}")
        T, H, W = int(rgb.shape[0]), int(rgb.shape[1]), int(rgb.shape[2])
        if T < 5 or (T - 1) % 4 != 0:
            raise ControlError(f"rgb T must be 4k+1 (k>=1), got T={T}")
        if H % 16 != 0 or W % 16 != 0:
            raise ControlError(f"rgb H/W must be multiples of 16, got {H}x{W}")

        c2w = data["camera_to_world"]
        if not np.issubdtype(c2w.dtype, np.floating):
            raise ControlError(f"camera_to_world must be float, got {c2w.dtype}")
        if c2w.shape != (T, 4, 4):
            raise ControlError(f"camera_to_world must be [{T},4,4], got {c2w.shape}")
        if not np.isfinite(c2w).all():
            raise ControlError("camera_to_world contains non-finite values")
        bottom = c2w[:, 3, :]
        if not np.allclose(bottom, np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-5):
            warnings.append(
                "camera_to_world last row deviates from [0,0,0,1] "
                f"(max abs dev {float(np.abs(bottom - np.array([0, 0, 0, 1.0])).max()):.3g})"
            )
        rotation = c2w[:, :3, :3]
        orth_err = float(np.abs(rotation @ np.swapaxes(rotation, 1, 2) - np.eye(3)).max())
        if orth_err > 1e-4:
            warnings.append(f"camera_to_world rotation is not orthonormal (max err {orth_err:.3g})")

        K = data["K"]
        if not np.issubdtype(K.dtype, np.floating):
            raise ControlError(f"K must be float, got {K.dtype}")
        if K.shape != (T, 3, 3):
            raise ControlError(f"K must be [{T},3,3], got {K.shape}")
        if not np.isfinite(K).all():
            raise ControlError("K contains non-finite values")
        if not np.allclose(K, K[0], rtol=K_CONST_RTOL, atol=K_CONST_ATOL):
            worst = int(np.argmax(np.abs(K - K[0]).max(axis=(1, 2))))
            raise ControlError(
                "K varies across frames; the official pipeline accepts exactly one K per "
                f"stream (frame {worst} deviates; worst abs dev "
                f"{float(np.abs(K - K[0]).max()):.6g}). Constant K is required."
            )

        source_time = data["source_time"]
        target_time = data["target_time"]
        for name, arr in (("source_time", source_time), ("target_time", target_time)):
            if not np.issubdtype(arr.dtype, np.floating):
                raise ControlError(f"{name} must be float, got {arr.dtype}")
            if arr.shape != (T,):
                raise ControlError(f"{name} must be [{T}], got {arr.shape}")
            if not np.isfinite(arr).all():
                raise ControlError(f"{name} contains non-finite values")

        diffs = np.diff(source_time.astype(np.float64))
        if (diffs < -1e-6).any():
            warnings.append("source_time is not monotonic non-decreasing")
        if (source_time < -1e-6).any():
            warnings.append("source_time contains negative values")

        fps_arr = np.asarray(data["fps"]).astype(np.float64)
        if fps_arr.ndim != 0:
            raise ControlError(f"fps must be scalar, got shape {fps_arr.shape}")
        fps = float(fps_arr)
        if not np.isfinite(fps) or fps <= 0:
            raise ControlError(f"fps must be finite and > 0, got {fps}")

    warn = time_unit_warning(source_time, T)
    if warn:
        warnings.append(warn)

    return Controls(
        path=path,
        sha256=sha256_file(path),
        rgb=rgb,
        camera_to_world=c2w,
        K=K,
        source_time=source_time.astype(np.float32),
        target_time=target_time.astype(np.float32),
        fps=fps,
        warnings=warnings,
    )


def time_unit_warning(source_time: np.ndarray, T: int) -> str | None:
    """時間値の単位を公式規約（秒 = frame_index/30）と照合し、疑わしければ警告を返す。

    公式 load_time はファイル値を無変換で使い、RoPE 側が fps=30 を乗算する。値が
    video の frame index そのもの（0..T-1）だと 30 倍の時間位置になるため、
    その場合は root が NPZ を修正できるよう明示警告する（勝手な変換はしない）。
    """
    limit = (T - 1) / OFFICIAL_TIME_FPS
    value = float(np.max(source_time)) if source_time.size else 0.0
    if value > limit * (1 + 1e-6) + 1e-9:
        return (
            f"source_time max={value:.6g} exceeds official seconds convention "
            f"(T-1)/30={limit:.6g}: values look like frame indices, but official "
            "load_time/rope treat them as seconds (rope multiplies by 30). "
            "Adapter passes values through unchanged; fix the NPZ if this is unintended."
        )
    return None


def rgb_to_reference_video(rgb: np.ndarray) -> torch.Tensor:
    """[T,H,W,3] uint8 -> [1,3,T,H,W] float32 in [-1,1]（公式 frame_transform と同一）。"""
    x = torch.from_numpy(np.ascontiguousarray(rgb)).to(torch.float32)
    x = x.div(255.0).mul(2.0).sub(1.0)
    x = x.permute(0, 3, 1, 2).contiguous()  # [T,C,H,W]
    x = x.permute(1, 0, 2, 3).contiguous()  # [C,T,H,W]
    return x.unsqueeze(0)  # [1,C,T,H,W]


# ---------------------------------------------------------------------------
# カメラ（cam_utils.py の式を数値そのまま流用。diffusers 依存のない torch のみ）
# ---------------------------------------------------------------------------


def normalize_camera_poses(c2ws: torch.Tensor, reference_c2w: torch.Tensor) -> torch.Tensor:
    """cam_utils.normalize_camera_poses と同一: first reference pose を基準に正規化。"""
    assert c2ws.ndim == 3 and c2ws.shape[1:] == (4, 4), "Input must be [N, 4, 4]"
    if reference_c2w.ndim == 2:
        reference_c2w = reference_c2w.unsqueeze(0)
    origin_inv = torch.linalg.inv(reference_c2w)
    return origin_inv @ c2ws


def compute_plucker_ray_map(
    intrinsics: torch.Tensor, c2ws: torch.Tensor, H: int, W: int, device: str = "cpu"
) -> torch.Tensor:
    """cam_utils.compute_plucker_ray_map (commit e033161) と同一式・同一正規化。

    Args:
        intrinsics: [3,3]; c2ws: [N,4,4]; H/W: pixel 解像度
    Returns:
        [N,H,W,6] Plucker (dirs xyz, moments xyz)
    """
    N = c2ws.shape[0]
    intrinsics = intrinsics.to(device)
    c2ws = c2ws.to(device)
    dtype = c2ws.dtype

    i, j = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    i = i + 0.5
    j = j + 0.5

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x = (j - cx) / fx  # [H, W]
    y = (i - cy) / fy
    z = torch.ones_like(x)

    dirs_cam = torch.stack([x, y, z], dim=-1)  # [H, W, 3]
    dirs_cam = dirs_cam / torch.norm(dirs_cam, dim=-1, keepdim=True)
    dirs_cam = dirs_cam.view(1, H, W, 3).expand(N, H, W, 3)  # [N, H, W, 3]

    R = c2ws[:, :3, :3]  # [N, 3, 3]
    T = c2ws[:, :3, 3]  # [N, 3]

    dirs_cam_flat = dirs_cam.view(N, -1, 3)  # [N, H*W, 3]
    dirs_world_flat = torch.bmm(dirs_cam_flat, R.transpose(1, 2))  # [N, H*W, 3]
    dirs_world_flat = dirs_world_flat / torch.norm(dirs_world_flat, dim=-1, keepdim=True)
    dirs_world = dirs_world_flat.view(N, H, W, 3)

    origins_world = T.view(N, 1, 1, 3).expand(N, H, W, 3)

    moments = torch.cross(origins_world, dirs_world, dim=-1)  # [N, H, W, 3]
    plucker = torch.cat([dirs_world, moments], dim=-1)  # [N, H, W, 6]
    return plucker


def build_camera_inputs(controls: Controls) -> tuple[torch.Tensor, dict, dict]:
    """公式 real_world 推論経路と同じカメラ入力を構築する。

    手順 (extract_cam_realworld / get_inference_data_sample と同一順序):
      1. arange(T)[::4] で N=(T-1)//4+1 本に間引き
      2. first pose 翻訳 norm > 20 なら outdoor 規則で全翻訳 /10
      3. reference(=source) first pose を基準に tgt/ref を joint 正規化
         （本契約では source/target とも実測 source 軌跡 = 同一配列だが、
          upstream と同じくストリーム別に正規化して渡す）
      4. cam_utils 式の plucker を full pixel 解像度で生成し [tgt; ref] で連結

    Returns:
        camera [1, 2N, H, W, 6] float32, camera_params dict, info dict
    """
    T, H, W = controls.T, controls.H, controls.W
    indices = sample_indices(T)
    sampled = np.asarray(controls.camera_to_world)[indices.numpy()]
    c2ws = torch.tensor(sampled, dtype=torch.float32)  # [N,4,4]

    first_dist = float(torch.linalg.norm(c2ws[0, :3, 3]))
    rescaled = False
    if first_dist > SCENE_RESCALE_THRESHOLD:
        c2ws = c2ws.clone()
        c2ws[:, :3, 3] /= SCENE_RESCALE_FACTOR
        rescaled = True

    # Controls use integer pixel centers (OpenCV/ARROW). Official Plucker
    # samples j+0.5 and PRoPE normalizes edge coordinates by image width.
    # Shift BOTH model consumers; leave rectification/source K unchanged.
    K_integer = torch.tensor(np.asarray(controls.K[0]), dtype=torch.float32)
    K = K_integer.clone()
    K[0, 2] += 0.5
    K[1, 2] += 0.5
    reference_c2w = c2ws[0]
    tgt_norm = normalize_camera_poses(c2ws, reference_c2w)
    ref_norm = normalize_camera_poses(c2ws, reference_c2w)

    tgt_plucker = compute_plucker_ray_map(K, tgt_norm, H, W)
    ref_plucker = compute_plucker_ray_map(K, ref_norm, H, W)
    camera = torch.cat([tgt_plucker, ref_plucker], dim=0)  # [2N,H,W,6] tgt first

    camera_params = {
        "tgt_c2ws": tgt_norm,  # [N,4,4]  (pipeline が内部で latent 長まで先頭 pose を複製)
        "ref_c2ws": ref_norm,  # [N,4,4]
        "tgt_intrinsics": K,  # [3,3]     (公式 pipeline は 1 K / stream 前提)
        "ref_intrinsics": K.clone(),  # [3,3]
        "image_width": int(W),
        "image_height": int(H),
    }
    info = {
        "n_poses": int(c2ws.shape[0]),
        "sample_indices_head": indices[:5].tolist(),
        "first_pose_translation_norm": first_dist,
        "scene_rescale_applied": rescaled,
        "scene_rescale_factor": SCENE_RESCALE_FACTOR if rescaled else 1.0,
        "normalized_first_pose_is_identity": bool(
            torch.allclose(tgt_norm[0], torch.eye(4), atol=1e-5)
        ),
        "camera_shape": list(camera.unsqueeze(0).shape),
        "plucker_spatial": [int(H), int(W)],
        "K_used": K.tolist(),
        "K_integer_center_input": K_integer.tolist(),
        "K_convention": "integer-center input to edge-center model: cx/cy + 0.5",
    }
    return camera.unsqueeze(0), camera_params, info


def sample_indices(T: int) -> torch.Tensor:
    """公式 indices_tensor.tolist()[::4] と同じ間引き（latent frame 同期）。"""
    return torch.arange(int(T))[::4]


# ---------------------------------------------------------------------------
# 時間・latent padding
# ---------------------------------------------------------------------------


def build_time_steps(controls: Controls, mode: str) -> tuple[torch.Tensor, dict]:
    """world_time_steps [1, 2T] を構築する（前半=target, 後半=source。公式 run_eval 順）。

    時間値は NPZ を無変換で渡す（coordinator TIME_CONTRACT_CORRECTION: model time =
    source i/30, target 0。物理 i/5 は metadata 側であり inference では使わない）。
    """
    source = torch.tensor(np.asarray(controls.source_time), dtype=torch.float32)
    target = torch.tensor(np.asarray(controls.target_time), dtype=torch.float32)
    if mode == "freeze":
        spread = float(target.max() - target.min())
        if spread > TARGET_TIME_CONST_ATOL:
            raise ControlError(
                f"mode=freeze requires constant target_time, got spread={spread:.6g} "
                f"(atol={TARGET_TIME_CONST_ATOL})"
            )
        target_part = target
    elif mode == "source_time":
        target_part = source
    else:
        raise ControlError(f"unknown mode: {mode!r}")

    time_steps = torch.cat([target_part, source], dim=0).unsqueeze(0)  # [1, 2T]
    info = {
        "mode": mode,
        "order": "first T = target stream, second T = reference(source) stream",
        "time_steps_shape": list(time_steps.shape),
        "target_time_constant": bool(mode == "freeze"),
        "target_time_value": float(target_part.mean()),
        "source_time_first": float(source[0]),
        "source_time_last": float(source[-1]),
        "source_time_max": float(source.max()),
        "unit": "seconds (official load_time convention); rope multiplies by 30 internally",
        "official_time_fps": OFFICIAL_TIME_FPS,
        "output_fps": float(controls.fps),
        "convention_note": (
            "model time = source_time i/30 (rope fps=30 hardcoded, time AdaLN units); "
            "physical timestamps i/5 and output fps=5 are metadata only; "
            "no implicit rescaling by the adapter"
        ),
    }
    return time_steps, info


def latent_padding_info(T: int, patch_size_t: int = 2) -> dict:
    """公式 pipeline/transformer の latent 時系列 padding を事前計算して報告する。

    - 生成ストリーム: latent_frames = (T-1)//4+1 = N。N % p_t != 0 のとき
      additional = p_t - N % p_t を「末尾」に追加 (num_frames += additional*4)。
    - reference ストリーム: ncopy = N % p_t 個の先頭 frame コピーを「前置」。
    - 全 hidden 長 L = (N + additional) + (N + ncopy)。plucker は 2N frame で渡し、
      transformer 内部が per-stream に N % p_t を前置して L/2 に合わせる。
    """
    N = (T - 1) // 4 + 1
    additional_gen = (patch_size_t - N % patch_size_t) % patch_size_t
    ncopy_ref = N % patch_size_t
    L = (N + additional_gen) + (N + ncopy_ref)
    return {
        "T_pixel_frames": int(T),
        "N_sampled_poses": int(N),
        "latent_frames_raw": int(N),
        "additional_gen_latent_tail": int(additional_gen),
        "ncopy_ref_latent_head": int(ncopy_ref),
        "L_total_hidden_frames": int(L),
        "camera_frames_passed": int(2 * N),
        "note": "padding semantics replicated by official pipeline/transformer internals",
    }


# ---------------------------------------------------------------------------
# 公式コードの import（namespace package、訓練スタック非ロード）
# ---------------------------------------------------------------------------


@dataclass
class OfficialModules:
    transformer_cls: Any
    pipeline_cls: Any
    scheduler_cls: Any
    cam_dir: Path
    module_files: dict

    def describe(self) -> dict:
        return {
            "namespace_package": NAMESPACE_PACKAGE,
            "cam_dir": str(self.cam_dir),
            "modules": self.module_files,
            "transformer_cls": f"{self.transformer_cls.__module__}.{self.transformer_cls.__qualname__}",
            "pipeline_cls": f"{self.pipeline_cls.__module__}.{self.pipeline_cls.__qualname__}",
            "scheduler_cls": f"{self.scheduler_cls.__module__}.{self.scheduler_cls.__qualname__}",
            "training_stack_imported": "finetune" in sys.modules,
        }


def load_official_modules(upstream_dir: Path | str) -> OfficialModules:
    """公式クラス群を training stack の eager import なしで読み込む。

    ``finetune/models/__init__.py`` は全サブモジュール（trainer 含む）を import するため、
    パッケージとしての ``finetune.models`` を経由せず、``__path__`` のみを持つ名前空間
    パッケージを importlib で作って相対 import (``from .cogvideox_transformer_3d import`` 等)
    を解決させる。これにより diffusers/transformers だけを依存として読み込む。
    """
    upstream_dir = Path(upstream_dir)
    cam_dir = upstream_dir / CAM_SUBDIR
    if not cam_dir.is_dir():
        raise UpstreamImportError(f"official cam module dir not found: {cam_dir}")

    existing = sys.modules.get(NAMESPACE_PACKAGE)
    if existing is not None:
        registered = Path(getattr(existing, "__path__", [""])[0]).resolve()
        if registered != cam_dir.resolve():
            raise UpstreamImportError(
                f"{NAMESPACE_PACKAGE} already registered for {registered}, not {cam_dir}"
            )
    else:
        package = types.ModuleType(NAMESPACE_PACKAGE)
        package.__path__ = [str(cam_dir)]
        package.__package__ = NAMESPACE_PACKAGE
        sys.modules[NAMESPACE_PACKAGE] = package

    transformer_mod = importlib.import_module(
        NAMESPACE_PACKAGE + ".cogvideox_transformer_3d_adaln_rope_time_cam"
    )
    pipeline_mod = importlib.import_module(
        NAMESPACE_PACKAGE + ".pipeline_cogvideox_adaln_rope_time_cam"
    )
    scheduler_mod = importlib.import_module(NAMESPACE_PACKAGE + ".scheduling_dpm_cogvideox")

    module_files = {}
    for name, module in (("transformer", transformer_mod), ("pipeline", pipeline_mod),
                         ("scheduler", scheduler_mod)):
        if module.__file__ is None:
            raise AdapterError(f"Official module {name} has no source file")
        module_files[name] = str(Path(module.__file__))

    return OfficialModules(
        transformer_cls=transformer_mod.CogVideoXTransformer3DModel,
        pipeline_cls=pipeline_mod.CogVideoXPipeline,
        scheduler_cls=scheduler_mod.CogVideoXDPMScheduler,
        cam_dir=cam_dir,
        module_files=module_files,
    )


def upstream_commit(upstream_dir: Path | str) -> str | None:
    """read-only の commit 記録（Git mutation は行わない）。"""
    try:
        result = subprocess.run(
            ["git", "-C", str(upstream_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# checkpoint 解決と strict ロード
# ---------------------------------------------------------------------------


def resolve_checkpoint_dir(checkpoint: Path | str) -> tuple[Path, str]:
    """checkpoint/fp32_ckpt または safetensors 直接配置 dir を解決する。"""
    path = Path(checkpoint)
    if not path.is_dir():
        raise CheckpointError(f"checkpoint path not found: {path}")

    def has_weights(folder: Path) -> bool:
        if (folder / "diffusion_pytorch_model.safetensors.index.json").is_file():
            return True
        return any(folder.glob("*.safetensors"))

    fp32 = path / "fp32_ckpt"
    if fp32.is_dir() and has_weights(fp32):
        return fp32, "fp32_ckpt"
    if has_weights(path):
        return path, "direct"
    raise CheckpointError(
        f"no safetensors found in {path} or {path / 'fp32_ckpt'} "
        "(expected fp32_ckpt/diffusion_pytorch_model*.safetensors[.index.json] or *.safetensors)"
    )


def checkpoint_metadata(ckpt_dir: Path) -> dict:
    """重みを load せずファイル構成と index key 数のみ報告する（dry-run 用）。"""
    index_files = sorted(ckpt_dir.glob("*.safetensors.index.json"))
    shards = sorted({p.name for p in ckpt_dir.glob("*.safetensors")})
    key_count = None
    index_name = None
    if index_files:
        if len(index_files) > 1:
            raise CheckpointError(f"multiple index files: {[p.name for p in index_files]}")
        index_name = index_files[0].name
        index = json.loads(index_files[0].read_text())
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise CheckpointError(f"invalid index (no weight_map): {index_files[0]}")
        key_count = len(weight_map)
        indexed_shards = sorted(set(weight_map.values()))
        missing = [s for s in indexed_shards if not (ckpt_dir / s).is_file()]
        if missing:
            raise CheckpointError(f"index references missing shards: {missing}")
        shards = indexed_shards
    return {
        "dir": str(ckpt_dir),
        "index_file": index_name,
        "index_sha256": sha256_file(index_files[0]) if index_files else None,
        "shard_count": len(shards),
        "shards": shards,
        "index_key_count": key_count,
        "note": "metadata only; no tensor loaded",
    }


def strict_load_transformer(model: torch.nn.Module, ckpt_dir: Path) -> dict:
    """BulletTime safetensors を strict にロードする。

    公式 fallback loader (trainer.py load_transformer_safetensors_into_model) は
    missing key と shape mismatch を黙認するが、本アダプタでは:
      - checkpoint key が model に無い  -> unexpected -> fail
      - model key が checkpoint に無い  -> missing    -> fail（未初期化重みの黙認禁止）
      - shape 不一致                     -> fail
    DDP 保存由来の ``module.`` prefix は剥がして照合する（upstream の逆写像）。
    """
    from safetensors import safe_open

    ckpt_dir = Path(ckpt_dir)
    state = model.state_dict()
    index_files = sorted(ckpt_dir.glob("*.safetensors.index.json"))
    if index_files:
        index = json.loads(index_files[0].read_text())
        weight_map = index["weight_map"]
        shard_names = sorted(set(weight_map.values()))
    else:
        shard_names = sorted(p.name for p in ckpt_dir.glob("*.safetensors"))

    if not shard_names:
        raise CheckpointError(f"no safetensors shards in {ckpt_dir}")

    seen: set[str] = set()
    unexpected: list[str] = []
    shape_mismatch: list[str] = []
    dtype_counts: Counter = Counter()

    for shard_name in shard_names:
        shard_path = ckpt_dir / shard_name
        if not shard_path.is_file():
            raise CheckpointError(f"missing shard file: {shard_path}")
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                target_key = key[len("module.") :] if key.startswith("module.") else key
                if target_key not in state:
                    unexpected.append(key)
                    continue
                tensor = handle.get_tensor(key)
                if tuple(tensor.shape) != tuple(state[target_key].shape):
                    shape_mismatch.append(
                        f"{key}: ckpt {tuple(tensor.shape)} vs model {tuple(state[target_key].shape)}"
                    )
                    continue
                state[target_key].copy_(tensor)
                dtype_counts[str(tensor.dtype)] += 1
                seen.add(target_key)

    missing = [key for key in state.keys() if key not in seen]

    if missing or unexpected or shape_mismatch:
        def head(items: list[str], n: int = 8) -> str:
            return "; ".join(items[:n]) + (f" ... (+{len(items) - n} more)" if len(items) > n else "")

        parts = []
        if missing:
            parts.append(f"missing {len(missing)}: {head(missing)}")
        if unexpected:
            parts.append(f"unexpected {len(unexpected)}: {head(unexpected)}")
        if shape_mismatch:
            parts.append(f"shape_mismatch {len(shape_mismatch)}: {head(shape_mismatch)}")
        raise CheckpointError("strict checkpoint load failed: " + " | ".join(parts))

    return {
        "dir": str(ckpt_dir),
        "shard_count": len(shard_names),
        "shards": shard_names,
        "model_keys": len(state),
        "loaded_keys": len(seen),
        "missing": 0,
        "unexpected": 0,
        "shape_mismatch": 0,
        "source_dtypes": dict(dtype_counts),
    }


def build_transformer(
    base_model: Path, official: OfficialModules, ckpt_dir: Path
) -> tuple[torch.nn.Module, dict, dict]:
    """base config から transformer を構築し、公式手順で additional weights を付与後、
    BulletTime safetensors を strict ロードする。"""
    config_path = Path(base_model) / "transformer" / "config.json"
    if not config_path.is_file():
        raise AdapterError(f"base transformer config not found: {config_path}")
    config = official.transformer_cls.load_config(str(config_path))
    model = official.transformer_cls.from_config(config)
    # 公式 CogVideoX1_5CamLoraTrainer.load_components と同じ手順・同じ引数。
    # (conv_dim は上流実装では未使用の変数であり cam_dim=128 は hardcode。provenance の
    # ため trainer と同じ 16 を渡す)
    model._init_additional_weights(conv_dim=16)
    model.eval()
    stats = strict_load_transformer(model, ckpt_dir)
    model_config = json.loads(config_path.read_text())
    return model, model_config, stats


# ---------------------------------------------------------------------------
# pipeline 構築（公式 validate 手順: BF16 + CPU offload + VAE tiling/slicing）
# ---------------------------------------------------------------------------


def encode_reference_with_offload(pipe, video: torch.Tensor) -> torch.Tensor:
    """Preserve upstream encoding while respecting the VAE execution device.

    Upstream reads vae.device before the encode hook moves its weights to CUDA.
    diffusers 0.34 apply_forward_hook calls pre_forward without input arguments,
    so it cannot move that CPU tensor. Use the pipeline execution device first.
    Sampling and scaling remain identical to official encode_video.
    """
    vae = pipe.vae
    video = video.to(device=pipe._execution_device, dtype=vae.dtype)
    return vae.encode(video).latent_dist.sample() * vae.config.scaling_factor


def load_pipeline_bundle(args, official: OfficialModules) -> tuple[Any, dict]:
    """公式 pipeline を構成する。device は CUDA 必須（生成は GPU 前提）。"""
    if not torch.cuda.is_available():
        raise AdapterError("CUDA is required for generation (use --dry-run for CPU checks)")
    device = "cuda"
    base_model = Path(args.base_model)
    ckpt_dir, ckpt_kind = resolve_checkpoint_dir(args.checkpoint)

    # Load optional inference backends lazily; they live in the isolated environment.
    AutoencoderKLCogVideoX = getattr(importlib.import_module("diffusers"), "AutoencoderKLCogVideoX")
    transformers = importlib.import_module("transformers")
    AutoTokenizer, T5EncoderModel = transformers.AutoTokenizer, transformers.T5EncoderModel

    load_start = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(str(base_model), subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(str(base_model), subfolder="text_encoder")
    vae = AutoencoderKLCogVideoX.from_pretrained(str(base_model), subfolder="vae")
    scheduler = official.scheduler_cls.from_pretrained(str(base_model), subfolder="scheduler")
    transformer, model_config, ckpt_stats = build_transformer(base_model, official, ckpt_dir)

    pipe = official.pipeline_cls(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
    )
    # 公式 validate (CogVideoX1_5CamLoraTrainer.validate) と同じ順序:
    # enable_model_cpu_offload -> pipe.to(dtype=weight_dtype) -> VAE tiling/slicing
    pipe.enable_model_cpu_offload(device=device)
    pipe.to(dtype=torch.bfloat16)
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.encode_video = types.MethodType(encode_reference_with_offload, pipe)
    load_seconds = time.monotonic() - load_start

    bundle_info = {
        "device": device,
        "transformer_dtype": "bfloat16",
        "offload": "enable_model_cpu_offload",
        "vae_tiling": True,
        "vae_slicing": True,
        "compatibility": "encode_video uses pipeline execution device before VAE encode hook",
        "vae_scale_factor_spatial": int(pipe.vae_scale_factor_spatial),
        "vae_scale_factor_temporal": int(pipe.vae_scale_factor_temporal),
        "checkpoint_kind": ckpt_kind,
        "checkpoint": ckpt_stats,
        "transformer_config": model_config,
        "load_seconds": load_seconds,
        "gpu": torch.cuda.get_device_name(0),
    }
    return pipe, bundle_info


# ---------------------------------------------------------------------------
# 出力（raw 生成 RGB のみ。合成・source copy なし）
# ---------------------------------------------------------------------------


def write_outputs(raw_video: torch.Tensor, output_dir: Path, fps: float) -> dict:
    """raw video tensor [1,3,F,H,W] (VAE decode 出力, [-1,1]) を PNG/MP4 に保存する。

    公式 save_video_tensor と同じ変換のみ: clamp(-1,1) -> (x+1)/2 -> uint8。
    """
    imageio = importlib.import_module("imageio")
    from PIL import Image

    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    tensor = raw_video.detach().float().cpu()
    if tensor.dim() != 5 or tensor.shape[0] != 1:
        raise AdapterError(f"unexpected raw video shape: {tuple(tensor.shape)}")
    frames_u8 = []
    frame_records = []
    for index in range(tensor.shape[2]):
        frame = tensor[0, :, index].clamp(-1.0, 1.0)
        frame = ((frame + 1.0) * 0.5 * 255.0).round().to(torch.uint8)
        array = frame.permute(1, 2, 0).numpy()  # [H,W,3]
        frames_u8.append(array)
        frame_path = frames_dir / f"frame_{index:04d}.png"
        Image.fromarray(array).save(frame_path)
        frame_records.append({"path": str(frame_path), "sha256": sha256_file(frame_path)})

    mp4_path = output_dir / "generated.mp4"
    with imageio.get_writer(
        str(mp4_path), fps=float(fps), codec="libx264", pixelformat="yuv420p"
    ) as writer:
        for array in frames_u8:
            writer.append_data(array)

    combined = hashlib.sha256()
    for record in frame_records:
        combined.update(f"{Path(record['path']).name}:{record['sha256']}\n".encode())
    return {
        "n_frames": len(frame_records),
        "frames_dir": str(frames_dir),
        "frames_sha256_combined": combined.hexdigest(),
        "frames": frame_records,
        "mp4": {
            "path": str(mp4_path),
            "bytes": mp4_path.stat().st_size,
            "sha256": sha256_file(mp4_path),
            "fps": float(fps),
        },
    }


# ---------------------------------------------------------------------------
# 実行本体
# ---------------------------------------------------------------------------


def package_versions() -> dict:
    from importlib import metadata

    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in (
        "torch",
        "torchvision",
        "diffusers",
        "transformers",
        "accelerate",
        "peft",
        "huggingface-hub",
        "numpy",
        "safetensors",
        "pillow",
        "scipy",
        "einops",
        "sentencepiece",
        "imageio",
        "imageio-ffmpeg",
    ):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def base_model_report(base_model: Path | str | None, official: OfficialModules, H: int, W: int) -> dict:
    """base model 設定の要約（存在時のみ）。RoPE grid 上限の事前検査込み。"""
    report: dict = {"path": str(base_model) if base_model else None, "present": False}
    if not base_model:
        return report
    config_path = Path(base_model) / "transformer" / "config.json"
    report["transformer_config_present"] = config_path.is_file()
    if not config_path.is_file():
        return report
    config = official.transformer_cls.load_config(str(config_path))
    report["present"] = True
    report["transformer_config"] = {
        key: config.get(key)
        for key in (
            "patch_size",
            "patch_size_t",
            "in_channels",
            "num_layers",
            "num_attention_heads",
            "attention_head_dim",
            "sample_frames",
            "sample_height",
            "sample_width",
            "text_embed_dim",
            "use_rotary_positional_embeddings",
        )
    }
    patch_size = int(config.get("patch_size", 2))
    grid_h = H // (VAE_SPATIAL_FACTOR_FALLBACK * patch_size)
    grid_w = W // (VAE_SPATIAL_FACTOR_FALLBACK * patch_size)
    max_h = int(config.get("sample_height", 96)) // patch_size
    max_w = int(config.get("sample_width", 170)) // patch_size
    report["rope_grid"] = {
        "grid_h": grid_h,
        "grid_w": grid_w,
        "max_h": max_h,
        "max_w": max_w,
        "ok": grid_h <= max_h and grid_w <= max_w,
        "note": "grid = H//(8*patch); slice RoPE max = sample_h/w // patch",
    }
    report["scheduler_config_present"] = (Path(base_model) / "scheduler" / "scheduler_config.json").is_file()
    return report


def check_rope_grid_or_fail(grid_h: int, grid_w: int, patch_size: int, config: dict) -> None:
    max_h = int(config["sample_height"]) // patch_size
    max_w = int(config["sample_width"]) // patch_size
    if grid_h > max_h or grid_w > max_w:
        raise AdapterError(
            f"resolution too large for trained RoPE slice grid: grid=({grid_h},{grid_w}) "
            f"max=({max_h},{max_w}) (H//16 <= sample_height//patch and W//16 <= sample_width//patch)"
        )


def run_dry_run(args) -> int:
    """checkpoint 重みを load せず、形状・規約・設定を報告する。"""
    controls = load_controls(Path(args.controls))
    official = load_official_modules(args.upstream_dir)
    camera, camera_params, camera_info = build_camera_inputs(controls)
    time_steps, time_info = build_time_steps(controls, args.mode)
    padding = latent_padding_info(controls.T, patch_size_t=2)
    reference_video = rgb_to_reference_video(controls.rgb)

    report = {
        "status": "dry_run_ok",
        "at": utc_now(),
        "argv": sys.argv,
        "upstream": {
            "dir": str(args.upstream_dir),
            "commit": upstream_commit(args.upstream_dir),
            "expected_commit": UPSTREAM_COMMIT,
            "modules": official.describe(),
        },
        "controls": {
            "path": str(controls.path),
            "sha256": controls.sha256,
            "T": controls.T,
            "H": controls.H,
            "W": controls.W,
            "fps": controls.fps,
            "rgb_dtype": str(controls.rgb.dtype),
            "camera_dtype": str(controls.camera_to_world.dtype),
            "K_dtype": str(controls.K.dtype),
            "warnings": controls.warnings,
            "camera": camera_info,
            "time": time_info,
        },
        "inputs_that_would_be_passed": {
            "reference_video": list(reference_video.shape),
            "camera": list(camera.shape),
            "time_steps": list(time_steps.shape),
            "camera_params": {
                "tgt_c2ws": list(camera_params["tgt_c2ws"].shape),
                "ref_c2ws": list(camera_params["ref_c2ws"].shape),
                "tgt_intrinsics": list(camera_params["tgt_intrinsics"].shape),
                "ref_intrinsics": list(camera_params["ref_intrinsics"].shape),
                "image_width": camera_params["image_width"],
                "image_height": camera_params["image_height"],
            },
            "num_frames": controls.T,
            "height": controls.H,
            "width": controls.W,
        },
        "latent_padding": padding,
        "conventions": {
            "camera_order": "camera frames = [target N ; reference N]; tgt first",
            "time_order": "time_steps = cat([target_time, source_time]) [1, 2T]",
            "time_unit": "seconds (frame_index/30); rope multiplies by 30 internally",
            "plucker": "cam_utils formula, full pixel resolution HxW, float32",
            "camera_normalization": "joint, by first sampled source pose; outdoor >20 rule applies",
            "K": "constant integer-center controls; cx/cy +0.5 for Plucker and PRoPE",
            "no_crop": "no resize/crop inside inference; rgb must already be inference resolution",
        },
        "base_model": base_model_report(args.base_model, official, controls.H, controls.W),
        "checkpoint": None,
        "device": {
            "cuda_available": torch.cuda.is_available(),
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
        },
        "versions": package_versions(),
    }

    if args.checkpoint and Path(args.checkpoint).is_dir():
        ckpt_dir, kind = resolve_checkpoint_dir(args.checkpoint)
        report["checkpoint"] = {"kind": kind, **checkpoint_metadata(ckpt_dir)}

    if report["base_model"].get("rope_grid", {}).get("ok") is False:
        raise AdapterError(f"rope grid check failed: {report['base_model']['rope_grid']}")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def run_generation(args) -> int:
    controls = load_controls(Path(args.controls))
    official = load_official_modules(args.upstream_dir)

    if not args.base_model or not args.checkpoint or not args.output_dir:
        raise AdapterError("--base-model, --checkpoint and --output-dir are required for generation")

    base_model = Path(args.base_model)
    if not (base_model / "transformer" / "config.json").is_file():
        raise AdapterError(f"base model incomplete (no transformer/config.json): {base_model}")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise AdapterError(f"output dir is occupied: {output_dir} (refusing to overwrite)")
    output_dir.mkdir(parents=True, exist_ok=True)

    camera, camera_params, camera_info = build_camera_inputs(controls)
    time_steps, time_info = build_time_steps(controls, args.mode)
    reference_video = rgb_to_reference_video(controls.rgb)

    started = utc_now()
    t0 = time.monotonic()
    result: dict = {
        "status": "running",
        "at": started,
        "argv": sys.argv,
        "mode": args.mode,
        "seed": int(args.seed),
        "steps": int(args.steps),
        "guidance": float(args.guidance),
        "upstream": {
            "dir": str(args.upstream_dir),
            "commit": upstream_commit(args.upstream_dir),
            "expected_commit": UPSTREAM_COMMIT,
            "modules": official.describe(),
        },
        "base_model": str(base_model),
        "controls": {
            "path": str(controls.path),
            "sha256": controls.sha256,
            "T": controls.T,
            "H": controls.H,
            "W": controls.W,
            "fps": controls.fps,
            "rgb_sha256": sha256_array(controls.rgb),
            "camera_to_world_sha256": sha256_array(controls.camera_to_world),
            "K_sha256": sha256_array(controls.K),
            "source_time_sha256": sha256_array(controls.source_time),
            "target_time_sha256": sha256_array(controls.target_time),
            "warnings": controls.warnings,
            "camera": camera_info,
            "time": time_info,
        },
        "inputs": {
            "reference_video_shape": list(reference_video.shape),
            "camera_shape": list(camera.shape),
            "time_steps_shape": list(time_steps.shape),
            "latent_padding": latent_padding_info(controls.T, patch_size_t=2),
        },
        "prompt": DEFAULT_PROMPT,
        "negative_prompt": None,
        "output_meaning": (
            "raw model RGB frames; no compositing, no source copy, not metric RGB-D"
        ),
        "versions": package_versions(),
    }

    try:
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

        pipe, bundle_info = load_pipeline_bundle(args, official)
        result["runtime"] = bundle_info
        check_rope_grid_or_fail(
            controls.H // (bundle_info["vae_scale_factor_spatial"] * int(bundle_info["transformer_config"]["patch_size"])),
            controls.W // (bundle_info["vae_scale_factor_spatial"] * int(bundle_info["transformer_config"]["patch_size"])),
            int(bundle_info["transformer_config"]["patch_size"]),
            bundle_info["transformer_config"],
        )

        # 乱数規約: 初期ノイズは generator、VAE latent の sample() は global RNG を使うため
        # seed を両方に固定する（同一入力+同一seed の再現性確保）。
        torch.manual_seed(int(args.seed))
        torch.cuda.manual_seed_all(int(args.seed))
        generator = torch.Generator(device="cuda").manual_seed(int(args.seed))

        infer_start = time.monotonic()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            video_tensor, _video = pipe(
                reference_video=reference_video,
                camera=camera,
                camera_params=camera_params,
                time_steps=time_steps,
                prompt=DEFAULT_PROMPT,
                negative_prompt=None,
                height=controls.H,
                width=controls.W,
                num_frames=controls.T,
                num_inference_steps=int(args.steps),
                guidance_scale=float(args.guidance),
                generator=generator,
                output_type="pil",
            )
        infer_seconds = time.monotonic() - infer_start

        expected_shape = (1, 3, controls.T, controls.H, controls.W)
        if tuple(video_tensor.shape) != expected_shape:
            raise AdapterError(f"generated shape {tuple(video_tensor.shape)} != {expected_shape}")
        if not torch.isfinite(video_tensor).all():
            raise AdapterError("generated RGB contains nonfinite values")

        save_start = time.monotonic()
        outputs = write_outputs(video_tensor, output_dir, controls.fps)
        save_seconds = time.monotonic() - save_start

        result["status"] = "ok"
        result["runtime"]["infer_seconds"] = infer_seconds
        result["runtime"]["save_seconds"] = save_seconds
        result["runtime"]["total_seconds"] = time.monotonic() - t0
        result["runtime"]["peak_gpu_memory_gib"] = (
            torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
        )
        result["runtime"]["peak_memory_scope"] = "inference after model load; allocated bytes"
        result["runtime"]["scheduler_noise_seed"] = 42
        result["outputs"] = outputs
        result["at_end"] = utc_now()
        manifest_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({k: result[k] for k in ("status", "mode", "runtime", "outputs")}, indent=2, ensure_ascii=False))
        return 0
    except Exception as error:  # noqa: BLE001 - 例外は manifest に記録して非ゼロ終了
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
        result["at_end"] = utc_now()
        result["total_seconds"] = time.monotonic() - t0
        (output_dir / "manifest_failed.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )
        print(result["error"], file=sys.stderr)
        return 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BulletTime inference adapter (official pipeline, strict checkpoint load)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--upstream-dir", required=True, help="BulletTime checkout directory")
    parser.add_argument("--base-model", default=None, help="CogVideoX base model directory")
    parser.add_argument("--checkpoint", default=None, help="BulletTime checkpoint dir (fp32_ckpt/ or direct)")
    parser.add_argument("--controls", required=True, help="controls NPZ (no pickles)")
    parser.add_argument("--output-dir", default=None, help="output directory (must be unoccupied)")
    parser.add_argument("--steps", type=int, default=50, help="num_inference_steps")
    parser.add_argument("--seed", type=int, default=42, help="random seed (generator + global RNG)")
    parser.add_argument("--guidance", type=float, default=6.0, help="classifier-free guidance scale")
    parser.add_argument("--mode", choices=["freeze", "source_time"], default="freeze")
    parser.add_argument("--dry-run", action="store_true", help="report shapes/conventions only")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        if args.dry_run:
            return run_dry_run(args)
        return run_generation(args)
    except AdapterError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
