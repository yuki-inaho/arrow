"""scripts/depth_detail_models.py（pipeline scope）の実consumerに対する独立CPU統合テスト。

core scope: tests/test_depth_detail_pipeline.py。実装はmirrorせず、既知の
非identityカメラ軌道・既知depthから解析的期待値を先に定める。

入力は save_result の実schemaで直列化した合成 Result NPZ（[B,N,...] batch次元保持）。
検証する必須反例:
1. geometry: [B,N,...]保持、middle参考pose→ref0は inv(P0)@Pi、det(R)=1、
   scaleはdepth/translationへ1回だけ。
2. sceneflow_prep: depth_b→a 投影は inv(M_a)@M_b、sourceK_b/targetK_a、
   original_to_input affineとpixel Kのhalf-pixel対応（非identity入力）。
3. overlap境界: frame20が別clip（clip_020）由来でも world c2w で連続し、
   raw camera 参照（clip local c2r）を混ぜない。
4. lingbot --device cuda は CUDA 利用不可時に CPU fallback せず fail。
5. motion_candidate_from_sr: 非0 residual motion の解析fixtureで
   candidate=(inv(T)@[D1obs*r1(q1)]-S).z を厳密検証（pipeline実装をimportして実行）。
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Protocol

import numpy as np
import pytest
import torch

from arrow.types import Predictions, Queries, Result
from arrow.utils.results_io import save_result

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "depth_detail_models.py"

# ---- 合成データの固定パラメータ（解析的期待値の基準）----
FRAME_COUNT = 44
CLIP_FRAMES = 24
REF_LOCAL = CLIP_FRAMES // 2  # middle
ORIGINAL_H = 180
ORIGINAL_W = 240
INPUT_H = 90
INPUT_W = 120
LONG_SIDE_A = 120  # scenario A の sceneflow 低解像度 long side（240 -> 0.5倍）
Z0 = 2.0
POSE_GRID = 4
DEPTH_GRID = 8
ORIGINAL_AFFINE = np.array([[0.5, 0.0, 0.25], [0.0, 0.5, 0.25]])  # input = 0.5*original + 0.25
CLIP0_NAME = "clip_000.npz"
CLIP1_NAME = "clip_020.npz"
POST_H = 8
POST_W = 8
POST_LOW = 4
RENDER_H = 6
RENDER_W = 8
RENDER_STRIDE = 2
RENDER_FRAMES = 2
RENDER_K = np.array([[8.0, 0.0, 3.5], [0.0, 10.0, 2.5], [0.0, 0.0, 1.0]])
RENDER_ROTATION = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 非identity
RENDER_TRANSLATION = np.array([1.0, 2.0, 3.0])
RENDER_S = np.array([0.25, -0.5, 0.75])  # 既知のSR 3D変位（pixel (0,0)）
MOTION_T = np.array([0.008, 0.004, 0.002])
MOTION_RY = 0.0015
MOTION_RX = 0.0007
POSE_SCALE_FACTOR = 2.0  # scenario B の clip Sim3 scale

DEPTH_U = np.linspace(12.0, 108.0, DEPTH_GRID)
DEPTH_V = np.linspace(9.0, 81.0, DEPTH_GRID)
POSE_U = np.linspace(25.0, 95.0, POSE_GRID)
POSE_V = np.linspace(18.0, 72.0, POSE_GRID)
POSE_DEPTHS = np.array([1.7, 1.95, 2.2, 2.45])


def _rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rigid(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = translation
    return out


def cameras(count: int = FRAME_COUNT) -> np.ndarray:
    """既知の非identityカメラ軌道 C_f（world c2w）。C_0 = I。

    y/x 回転の合成で非可換な動きにし、a↔b 反転の区別を可能にする。
    """
    out = np.empty((count, 4, 4))
    for f in range(count):
        rotation = _rot_y(MOTION_RY * f) @ _rot_x(MOTION_RX * f)
        out[f] = _rigid(rotation, MOTION_T * f)
    return out


def project(K: np.ndarray, points: np.ndarray) -> np.ndarray:
    z = points[..., 2]
    x = K[0, 0] * points[..., 0] / z + K[0, 2]
    y = K[1, 1] * points[..., 1] / z + K[1, 2]
    return np.stack([x, y], axis=-1)


def k_original(frame: int) -> np.ndarray:
    return np.array(
        [[140.0 + 0.2 * frame, 0.0, 119.5], [0.0, 132.0 + 0.2 * frame, 89.5], [0.0, 0.0, 1.0]]
    )


def k_input(frame: int) -> np.ndarray:
    """入力画素K（original_to_input affine の適用後）。"""
    K = k_original(frame)
    return np.array(
        [
            [0.5 * K[0, 0], 0.0, 0.5 * K[0, 2] + 0.25],
            [0.0, 0.5 * K[1, 1], 0.5 * K[1, 2] + 0.25],
            [0.0, 0.0, 1.0],
        ]
    )


def low_geometry() -> dict:
    """long side -> low size/sx/sy/offset（画素中心規約、解析式）。"""
    scale = LONG_SIDE_A / max(ORIGINAL_H, ORIGINAL_W)
    low_h = max(2, int(round(ORIGINAL_H * scale)))
    low_w = max(2, int(round(ORIGINAL_W * scale)))
    sx = low_w / ORIGINAL_W
    sy = low_h / ORIGINAL_H
    return {
        "low_size": (low_h, low_w),
        "sx": sx,
        "sy": sy,
        "offset": (0.5 * sx - 0.5, 0.5 * sy - 0.5),
    }


def expected_k_low(frame: int) -> np.ndarray:
    """K_original -> low の解析期待値（u_low = sx*u_high + offset）。"""
    geom = low_geometry()
    K = k_original(frame)
    return np.array(
        [
            [geom["sx"] * K[0, 0], 0.0, geom["sx"] * K[0, 2] + geom["offset"][0]],
            [0.0, geom["sy"] * K[1, 1], geom["sy"] * K[1, 2] + geom["offset"][1]],
            [0.0, 0.0, 1.0],
        ]
    )


def _clip_arrays(global_indices: list[int], with_depth: bool):
    """既知幾何からpose/depthクエリ配列を作る（batch次元 [1, N, ...]）。"""
    cam = cameras()
    F = len(global_indices)
    Kp = POSE_GRID * POSE_GRID
    uu, vv = np.meshgrid(POSE_U, POSE_V)
    uu, vv = uu.ravel(), vv.ravel()
    zz = POSE_DEPTHS[np.arange(Kp) % len(POSE_DEPTHS)]
    K_ref = k_input(global_indices[REF_LOCAL])
    rays = np.stack([(uu - K_ref[0, 2]) / K_ref[0, 0], (vv - K_ref[1, 2]) / K_ref[1, 1], np.ones(Kp)], -1)
    P_ref = rays * zz[:, None]

    C_ref = cam[global_indices[REF_LOCAL]]
    own_pts = np.empty((F, Kp, 3))
    shared_pts = np.empty((F, Kp, 3))
    own_xy = np.empty((F, Kp, 2))
    shared_xy = np.empty((F, Kp, 2))
    for f, g in enumerate(global_indices):
        c2r = np.linalg.inv(C_ref) @ cam[g]
        p_h = np.concatenate([P_ref, np.ones((Kp, 1))], axis=1)
        own = (np.linalg.inv(c2r) @ p_h.T).T[:, :3]
        own_pts[f] = own
        shared_pts[f] = P_ref
        own_xy[f] = project(k_input(g), own)
        shared_xy[f] = project(K_ref, P_ref)

    uv_own = own_xy / np.array([INPUT_W, INPUT_H], dtype=np.float64)
    uv_shared = shared_xy / np.array([INPUT_W, INPUT_H], dtype=np.float64)
    assert np.all((uv_own >= 0.0) & (uv_own <= 1.0)), "pose own uv out of range"
    assert np.all((uv_shared >= 0.0) & (uv_shared <= 1.0)), "pose shared uv out of range"
    assert np.all(own_pts[..., 2] > 0.5), "pose own points behind camera"

    depth_pts = None
    depth_xy = None
    if with_depth:
        duu, dvv = np.meshgrid(DEPTH_U, DEPTH_V)
        duu, dvv = duu.ravel(), dvv.ravel()
        Kd = DEPTH_GRID * DEPTH_GRID
        depth_pts = np.empty((F, Kd, 3))
        depth_xy = np.empty((F, Kd, 2))
        for f, g in enumerate(global_indices):
            Kf = k_input(g)
            r = np.stack([(duu - Kf[0, 2]) / Kf[0, 0], (dvv - Kf[1, 2]) / Kf[1, 1], np.ones(Kd)], -1)
            depth_pts[f] = r * Z0
            depth_xy[f] = project(Kf, depth_pts[f])
        uv_depth = depth_xy / np.array([INPUT_W, INPUT_H], dtype=np.float64)
        assert np.all((uv_depth >= 0.0) & (uv_depth <= 1.0)), "depth uv out of range"

    n_pose = 2 * F * Kp
    n_depth = F * DEPTH_GRID * DEPTH_GRID if with_depth else 0
    xy = np.zeros((1, n_pose + n_depth, 2), dtype=np.float32)
    points = np.zeros((1, n_pose + n_depth, 3), dtype=np.float32)
    confidence = np.ones((1, n_pose + n_depth), dtype=np.float32)  # 実schemaは [B, N]
    uv = np.zeros((1, n_pose + n_depth, 2), dtype=np.float32)
    visibility = np.ones((1, n_pose + n_depth), dtype=np.float32)  # 実schemaは [B, N]
    xy[0, :n_pose] = np.concatenate([own_xy.reshape(-1, 2), shared_xy.reshape(-1, 2)], 0)
    points[0, :n_pose] = np.concatenate([own_pts.reshape(-1, 3), shared_pts.reshape(-1, 3)], 0)
    uv[0, :n_pose] = np.concatenate([uv_own.reshape(-1, 2), uv_shared.reshape(-1, 2)], 0)
    if with_depth:
        assert depth_xy is not None and depth_pts is not None, (
            "depth配列が未構築（fixture内部不整合）"
        )
        xy[0, n_pose:] = depth_xy.reshape(-1, 2)
        points[0, n_pose:] = depth_pts.reshape(-1, 3)
        uv[0, n_pose:] = (depth_xy / np.array([INPUT_W, INPUT_H])).reshape(-1, 2)

    frames_local = np.repeat(np.arange(F), Kp)
    frame_index = np.concatenate([frames_local, frames_local])
    reference_index = np.concatenate([frames_local, np.full(F * Kp, REF_LOCAL)])
    if with_depth:
        frame_index = np.concatenate([frame_index, np.repeat(np.arange(F), DEPTH_GRID * DEPTH_GRID)])
        reference_index = np.concatenate(
            [reference_index, np.repeat(np.arange(F), DEPTH_GRID * DEPTH_GRID)]
        )
    return {
        "xy": xy,
        "points": points,
        "confidence": confidence,
        "uv": uv,
        "visibility": visibility,
        "frame_index": frame_index,
        "reference_index": reference_index,
        "n_pose": n_pose,
        "n_depth": n_depth,
    }


def write_result_clip(
    path: Path,
    name: str,
    global_indices: list[int],
    with_depth: bool,
    ref_local: int = REF_LOCAL,
) -> None:
    """save_result（実schema）で合成 Result NPZ を書き出す。"""
    arrays = _clip_arrays(global_indices, with_depth)
    F = len(global_indices)
    queries = Queries(
        torch.from_numpy(arrays["xy"]),
        torch.from_numpy(arrays["frame_index"]),
        torch.from_numpy(arrays["frame_index"]),
        torch.from_numpy(arrays["reference_index"]),
    )
    predictions = Predictions(
        points=torch.from_numpy(arrays["points"]),
        point_confidence=torch.from_numpy(arrays["confidence"]),
        uv=torch.from_numpy(arrays["uv"]),
        visibility_logits=torch.from_numpy(arrays["visibility"]),
        normals=torch.zeros((1, arrays["points"].shape[1], 3), dtype=torch.float32),
        displacement=torch.zeros((1, arrays["points"].shape[1], 3), dtype=torch.float32),
    )
    groups = [
        {
            "task": "pose",
            "start": 0,
            "count": int(arrays["n_pose"]),
            "points_per_frame": 2 * (arrays["n_pose"] // (2 * len(global_indices))),
            "reference_frame": int(ref_local),
            "pose": {
                "confidence_keep": 0.5,
                "weight_power": 2.0,
                "min_points": 8,
                "visibility_logit_threshold": 0.0,
            },
        }
    ]
    if with_depth:
        groups.append(
            {
                "task": "depth",
                "start": int(arrays["n_pose"]),
                "count": int(arrays["n_depth"]),
                "reference_frame": "source",
            }
        )
    metadata = {
        "format_version": 1,
        "dataset": "synthetic_pipeline_fixture",
        "model": "synthetic-known-geometry",
        "groups": groups,
        "frame_info": [
            {
                "index": int(g),
                "original_to_input": ORIGINAL_AFFINE.tolist(),
                "input_size": [INPUT_H, INPUT_W],
                "original_size": [ORIGINAL_H, ORIGINAL_W],
            }
            for g in global_indices
        ],
    }
    result = Result(
        images=torch.zeros((F, 3, ORIGINAL_H, ORIGINAL_W), dtype=torch.uint8),
        queries=queries,
        predictions=predictions,
        metadata=metadata,
    )
    save_result(result, path)


def write_frames_json(path: Path, count: int) -> None:
    records = []
    for index in range(count):
        records.append(
            {
                "frame_index": index,
                "source_index": index,
                "name": f"frame_{index:06d}",
                "relative_path": f"frames/{index:06d}.npz",
                "rgb_path": str(path.parent / "frames_rgb" / f"{index:06d}.png"),
                "height": ORIGINAL_H,
                "width": ORIGINAL_W,
                "timestamp": None,
                "source": "synthetic",
            }
        )
    payload = {
        "dataset": "synthetic",
        "source": "synthetic",
        "frame_count": count,
        "original_size": [ORIGINAL_H, ORIGINAL_W],
        "frames": records,
        "created_at": "2026-10-16T00:00:00Z",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def write_rgb_pngs(root: Path, count: int) -> None:
    from PIL import Image

    frames_dir = root / "frames_rgb"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        rgb = np.zeros((ORIGINAL_H, ORIGINAL_W, 3), dtype=np.uint8)
        rgb[..., 0] = (np.arange(ORIGINAL_W) % 256).astype(np.uint8)
        rgb[..., 1] = index % 256
        Image.fromarray(rgb).save(frames_dir / f"{index:06d}.png")


def write_ndr_fixtures(root: Path, count: int) -> None:
    out_dir = root / "stages" / "ndr"
    out_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        np.savez_compressed(
            out_dir / f"{index:06d}.npz",
            ndr_depth=np.full((ORIGINAL_H, ORIGINAL_W), Z0, dtype=np.float32),
            ndr_valid=np.ones((ORIGINAL_H, ORIGINAL_W), dtype=bool),
            metadata=np.array(json.dumps({"stage": "ndr-fixture", "index": index})),
        )


def write_manifest(root: Path, clips: list[dict]) -> None:
    clips_dir = root / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    (clips_dir / "manifest.json").write_text(
        json.dumps(
            {"clips": clips, "signature": {"clip_overlap": 4}},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )


def write_alignments(root: Path, entries: list[dict]) -> None:
    (root / "alignments.json").write_text(
        json.dumps({"clips": entries}, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def build_dataset_a(root: Path) -> None:
    """44フレーム・2クリップ（overlap 4）。pose+depth付きの実schema Result NPZ。"""
    cam = cameras()
    write_frames_json(root / "frames.json", FRAME_COUNT)
    write_rgb_pngs(root, FRAME_COUNT)
    write_ndr_fixtures(root, FRAME_COUNT)
    clips_dir = root / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    write_result_clip(clips_dir / CLIP0_NAME, "clip0", list(range(0, CLIP_FRAMES)), True)
    write_result_clip(clips_dir / CLIP1_NAME, "clip1", list(range(20, 44)), True)
    write_manifest(
        root,
        [
            {"name": CLIP0_NAME, "start": 0, "end": CLIP_FRAMES, "status": "done"},
            {"name": CLIP1_NAME, "start": 20, "end": 44, "status": "done"},
        ],
    )
    c20 = cam[20]
    write_alignments(
        root,
        [
            {
                "name": CLIP0_NAME,
                "scale": 1.0,
                "rotation": np.eye(3).tolist(),
                "translation": [0.0, 0.0, 0.0],
                "note": "world = clip0 frame0",
            },
            {
                "name": CLIP1_NAME,
                "scale": 1.0,
                "rotation": c20[:3, :3].tolist(),
                "translation": c20[:3, 3].tolist(),
                "note": "world = C_0^-1 C_20 を適用",
            },
        ],
    )


def build_dataset_b(root: Path) -> None:
    """24フレーム・1クリップ。scale=2.0 の一度だけ適用検証用（pose+depth）。"""
    write_frames_json(root / "frames.json", CLIP_FRAMES)
    clips_dir = root / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    write_result_clip(clips_dir / CLIP0_NAME, "clip0b", list(range(0, CLIP_FRAMES)), True)
    write_manifest(root, [{"name": CLIP0_NAME, "start": 0, "end": CLIP_FRAMES, "status": "done"}])
    write_alignments(
        root,
        [
            {
                "name": CLIP0_NAME,
                "scale": POSE_SCALE_FACTOR,
                "rotation": np.eye(3).tolist(),
                "translation": [0.0, 0.0, 0.0],
                "note": "scale once test",
            }
        ],
    )


def build_dataset_post(root: Path) -> None:
    """sceneflow_post / sr 用の小型fixture（解析的期待値の基準）。"""
    from PIL import Image

    rows, cols = np.mgrid[0:POST_H, 0:POST_W]
    frames_index = list(range(8))
    rgb_dir = root / "frames_rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index in frames_index:
        rgb = np.zeros((POST_H, POST_W, 3), dtype=np.uint8)
        rgb[..., 0] = 40 + index
        rgb[..., 1] = (10 * index) % 256
        Image.fromarray(rgb).save(rgb_dir / f"{index:06d}.png")
        records.append(
            {
                "frame_index": index,
                "source_index": index,
                "name": f"post_frame_{index}",
                "rgb_path": str(rgb_dir / f"{index:06d}.png"),
                "height": POST_H,
                "width": POST_W,
                "timestamp": None,
                "source": "synthetic_post",
            }
        )
    (root / "frames.json").write_text(
        json.dumps(
            {
                "dataset": "synthetic_post",
                "source": "synthetic_post",
                "frame_count": len(records),
                "original_size": [POST_H, POST_W],
                "frames": records,
                "created_at": "2026-10-16T00:00:00Z",
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )

    pairs_dir = root / "stages" / "sceneflow" / "pairs"
    inputs_dir = pairs_dir / "inputs"
    outputs_dir = pairs_dir / "outputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    ndr_dir = root / "stages" / "ndr"
    ndr_dir.mkdir(parents=True, exist_ok=True)
    prep_dir = root / "stages" / "prep"
    prep_dir.mkdir(parents=True, exist_ok=True)

    pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    entries = []
    delta_low = {
        0: np.zeros((POST_LOW, POST_LOW)),
        2: np.full((POST_LOW, POST_LOW), 0.03),
        4: np.full((POST_LOW, POST_LOW), 0.5),
        6: np.zeros((POST_LOW, POST_LOW)),
    }
    support_low = {
        0: np.ones((POST_LOW, POST_LOW), dtype=bool),
        2: np.ones((POST_LOW, POST_LOW), dtype=bool),
        4: np.hstack(
            [
                np.ones((POST_LOW, POST_LOW // 2), dtype=bool),
                np.zeros((POST_LOW, POST_LOW - POST_LOW // 2), dtype=bool),
            ]
        ),
        6: np.ones((POST_LOW, POST_LOW), dtype=bool),
    }
    for a, b in pairs:
        name = f"pair_{a:06d}_{b:06d}"
        np.savez_compressed(
            inputs_dir / f"{name}.npz",
            depth0_low=np.full((POST_LOW, POST_LOW), 1.0),
            valid0_low=np.ones((POST_LOW, POST_LOW), dtype=bool),
            valid1_in_0=np.ones((POST_LOW, POST_LOW), dtype=bool),
        )
        np.savez_compressed(
            outputs_dir / f"{name}.npz",
            depth0_refined=(1.0 + delta_low[a]).astype(np.float32),
            depth1_refined=np.full((POST_LOW, POST_LOW), 1.0, dtype=np.float32),
            support=support_low[a],
            scene_flow=np.zeros((POST_LOW, POST_LOW, 3)),
            low_size=np.asarray([POST_LOW, POST_LOW], dtype=np.int32),
        )
        entries.append({"pair_index": [a, b], "clip_index": 0, "file": str(inputs_dir / f"{name}.npz")})
    (root / "stages" / "sceneflow" / "pairs.json").write_text(
        json.dumps(
            {
                "dataset": "synthetic_post",
                "low_size": [POST_LOW, POST_LOW],
                "pairs": entries,
                "counts": {"pairs": len(entries)},
                "units": "arrow_provisional",
                "created_at": "2026-10-16T00:00:00Z",
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )

    ndr_depth = {k: np.full((POST_H, POST_W), 2.0) for k in frames_index}
    ndr_depth[0] = 2.0 + 0.05 * ((rows + cols) % 2)  # 高周波チェッカーボード（細部保持検証）
    ndr_valid = {k: np.ones((POST_H, POST_W), dtype=bool) for k in frames_index}
    ndr_depth[6][3, 5] = np.nan  # 高解像度NDRのみ可能な単一画素ホール
    ndr_valid[6][3, 5] = False
    for index in frames_index:
        np.savez_compressed(
            ndr_dir / f"{index:06d}.npz",
            ndr_depth=ndr_depth[index].astype(np.float32),
            ndr_valid=ndr_valid[index],
            metadata=np.array(json.dumps({"stage": "ndr-fixture", "index": index})),
        )

    prep_edges = {k: np.zeros((POST_H, POST_W)) for k in frames_index}
    prep_edges[2][:, 4] = 1.0  # edge gate: 完全境界
    prep_edges[2][:, 5] = 0.5  # 半減衰
    for index in frames_index:
        np.savez_compressed(
            prep_dir / f"{index:06d}.npz",
            prep_edges=prep_edges[index],
            metadata=np.array(json.dumps({"stage": "prep-fixture", "index": index})),
        )


def render_depth(frame: int) -> np.ndarray:
    rows = np.arange(RENDER_H, dtype=np.float64)[:, None]
    return np.repeat(1.0 + float(frame) + rows, RENDER_W, axis=1)


def render_world_points(frame: int, use_stride: bool = True) -> np.ndarray:
    """worker_render の点群world座標の解析期待値。

    worker は depth_full[::stride,::stride] の index (a,b) を
    native = (a*stride, b*stride) (= 元画素 index) として K と整合させる。
    use_stride=False は stride無視（index をそのまま native 扱い）の旧式。
    """
    points = []
    depth = render_depth(frame)
    for a in range(RENDER_H // RENDER_STRIDE):
        for b in range(RENDER_W // RENDER_STRIDE):
            row = a * RENDER_STRIDE  # 元解像度の行
            col = b * RENDER_STRIDE  # 元解像度の列
            z = depth[row, col]
            native_x = col if use_stride else b
            native_y = row if use_stride else a
            x = (native_x - RENDER_K[0, 2]) * z / RENDER_K[0, 0]
            y = (native_y - RENDER_K[1, 2]) * z / RENDER_K[1, 1]
            points.append(RENDER_ROTATION @ np.array([x, y, z]) + RENDER_TRANSLATION)
    return np.asarray(points)


def render_expected_arrow(frame: int) -> dict:
    depth = render_depth(frame)
    z = depth[0, 0]
    local = np.array(
        [
            (0.0 - RENDER_K[0, 2]) * z / RENDER_K[0, 0],
            (0.0 - RENDER_K[1, 2]) * z / RENDER_K[1, 1],
            z,
        ]
    )
    return {
        "origin": RENDER_ROTATION @ local + RENDER_TRANSLATION,
        "vector": RENDER_S @ RENDER_ROTATION.T,  # 回転のみ適用（translationなし）
    }


def build_dataset_render(root: Path) -> None:
    """worker_render 用の小型fixture（known K / stride>1 / 非identity world）。"""
    from PIL import Image

    rgb_dir = root / "frames_rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(RENDER_FRAMES):
        rgb = np.zeros((RENDER_H, RENDER_W, 3), dtype=np.uint8)
        rgb[..., 0] = 30 + 20 * index
        rgb[..., 1] = np.arange(RENDER_W, dtype=np.uint8)[None, :] * 8
        rgb[..., 2] = np.arange(RENDER_H, dtype=np.uint8)[:, None] * 10
        Image.fromarray(rgb).save(rgb_dir / f"{index:06d}.png")
        records.append(
            {
                "frame_index": index,
                "source_index": index,
                "name": f"render_frame_{index}",
                "rgb_path": str(rgb_dir / f"{index:06d}.png"),
                "height": RENDER_H,
                "width": RENDER_W,
                "timestamp": None,
                "source": "synthetic_render",
            }
        )
    (root / "frames.json").write_text(
        json.dumps(
            {
                "dataset": "synthetic_render",
                "source": "synthetic_render",
                "frame_count": len(records),
                "original_size": [RENDER_H, RENDER_W],
                "frames": records,
                "created_at": "2026-10-16T00:00:00Z",
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )

    world = _rigid(RENDER_ROTATION, RENDER_TRANSLATION)
    for index in range(RENDER_FRAMES):
        depth = render_depth(index)
        temporal_dir = root / "stages" / "temporal"
        temporal_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            temporal_dir / f"{index:06d}.npz",
            temporal_depth=depth.astype(np.float32),
            temporal_valid=np.ones((RENDER_H, RENDER_W), dtype=bool),
        )
        geometry_dir = root / "stages" / "geometry"
        geometry_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            geometry_dir / f"{index:06d}.npz",
            arrow_K=RENDER_K,
            arrow_K_valid=np.bool_(True),
            arrow_pose_valid=np.bool_(True),
            arrow_pose_frame0_valid=np.bool_(True),
            arrow_camera_to_reference=np.eye(4),
            arrow_camera_to_reconstruct=world,
            arrow_camera_to_metric=world,
        )
        flow = np.zeros((RENDER_H, RENDER_W, 3))
        flow[0, 0] = RENDER_S
        sr_dir = root / "stages" / "sr"
        sr_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            sr_dir / f"{index:06d}.npz",
            sr_flow=flow.astype(np.float32),
            sr_valid=np.ones((RENDER_H, RENDER_W), dtype=bool),
            sr_support_count=np.int64(RENDER_H * RENDER_W),
            sr_flow_low=np.zeros((3, 4, 3), dtype=np.float32),
            sr_low_valid=np.ones((3, 4), dtype=bool),
            sr_pair_index=np.asarray([index, index + 1], dtype=np.int64),
        )
        for stage, key in (
            ("ndr", "ndr_depth"),
            ("anchor", "arrow_depth"),
            ("sceneflow_post", "sceneflow_depth"),
            ("lingbot", "lingbot_depth"),
        ):
            stage_dir = root / "stages" / stage
            stage_dir.mkdir(parents=True, exist_ok=True)
            valid_key = key.replace("_depth", "_valid")
            np.savez_compressed(
                stage_dir / f"{index:06d}.npz",
                **{key: depth.astype(np.float32), valid_key: np.ones((RENDER_H, RENDER_W), dtype=bool)},
            )
    (root / "stages").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / "stages" / "final.npz", depth_range=np.asarray([1.0, 8.0]))


# ---- worker_prep K_valid JSON正規化の回帰fixture ----

PREP_H = 6
PREP_W = 8
PREP_K = np.array([[6.0, 0.0, 3.5], [0.0, 7.0, 2.5], [0.0, 0.0, 1.0]])


def build_dataset_prep(root: Path) -> None:
    """worker_prep 用: 実 save_stage_frame で geometry/lingbot_fix を作る。

    フレーム0: K_valid=True + finite K → k_valid=True（np.bool_分岐）
    フレーム1: K_valid=True + NaN K → k_valid=False（np.bool_分岐のFalse側）
    フレーム2: K_valid=False → k_valid=False（短絡のPython False）
    """
    from PIL import Image

    module = load_pipeline_module()
    rgb_dir = root / "frames_rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(3):
        rgb = np.zeros((PREP_H, PREP_W, 3), dtype=np.uint8)
        rgb[..., 0] = 50 + 30 * index
        rgb[..., 1] = np.arange(PREP_W, dtype=np.uint8)[None, :] * 9
        rgb[..., 2] = np.arange(PREP_H, dtype=np.uint8)[:, None] * 7
        Image.fromarray(rgb).save(rgb_dir / f"{index:06d}.png")
        records.append(
            {
                "frame_index": index,
                "source_index": index,
                "name": f"prep_frame_{index}",
                "rgb_path": str(rgb_dir / f"{index:06d}.png"),
                "height": PREP_H,
                "width": PREP_W,
                "timestamp": None,
                "source": "synthetic_prep",
            }
        )
    (root / "frames.json").write_text(
        json.dumps(
            {
                "dataset": "synthetic_prep",
                "source": "synthetic_prep",
                "frame_count": len(records),
                "original_size": [PREP_H, PREP_W],
                "frames": records,
                "created_at": "2026-10-16T00:00:00Z",
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )

    k_nan = PREP_K.copy()
    k_nan[0, 0] = np.nan
    geometry_K = {0: PREP_K, 1: k_nan, 2: PREP_K}
    geometry_flag = {0: True, 1: True, 2: False}
    for index in range(3):
        module.save_stage_frame(
            root,
            "geometry",
            index,
            {"arrow_K": geometry_K[index], "arrow_K_valid": np.bool_(geometry_flag[index])},
            {"stage": "geometry", "fixture": "prep_k_valid_regression", "index": index},
        )
        module.save_stage_frame(
            root,
            "lingbot_fix",
            index,
            {"lingbot_depth": np.full((PREP_H, PREP_W), 3.0 + index, dtype=np.float32)},
            {"stage": "lingbot_fix", "fixture": "prep_k_valid_regression", "index": index},
        )


def build_dataset_render_empty_arrows(root: Path) -> None:
    """render fixture の最終frameだけSRを全false/NaNへ（矢印sample空）した回帰case。

    現行worker_renderはsampleが空のframeで /world/sr_arrows をlogしないため、
    latest-atで前frameの矢印が残り得る。明示empty rowの有無を検査する。
    """
    build_dataset_render(root)
    frame1 = root / "stages" / "sr" / "000001.npz"
    np.savez_compressed(
        frame1,
        sr_flow=np.full((RENDER_H, RENDER_W, 3), np.nan, dtype=np.float32),
        sr_valid=np.zeros((RENDER_H, RENDER_W), dtype=bool),
        sr_support_count=np.int64(0),
        sr_flow_low=np.zeros((3, 4, 3), dtype=np.float32),
        sr_low_valid=np.ones((3, 4), dtype=bool),
        sr_pair_index=np.asarray([1, 2], dtype=np.int64),
    )


# ---- RRD 読み戻し（Rerun 0.39: RrdReader + to_record_batch） ----


class _ArrowColumn(Protocol):
    def to_pylist(self) -> list: ...


class _RrdRecordBatch(Protocol):
    def column(self, name: str) -> _ArrowColumn: ...


def _rrd_entity_rows(rrd_path: Path, entity_path: str) -> list[tuple[list[int], _RrdRecordBatch]]:
    """entity_path の chunk を (frame値リスト, 生batch) として返す。"""
    from rerun.chunk import RrdReader

    rows: list[tuple[list[int], _RrdRecordBatch]] = []
    for chunk in RrdReader(str(rrd_path)).stream():
        if chunk.entity_path != entity_path:
            continue
        batch = chunk.to_record_batch()
        frames = [int(round(float(v))) for v in batch.column("frame").to_pylist()]
        rows.append((frames, batch))
    return rows


def rrd_points_by_frame(rrd_path: Path) -> dict[int, np.ndarray]:
    out: dict[int, list] = {}
    for frames, batch in _rrd_entity_rows(rrd_path, "/world/final_points"):
        for frame, pts in zip(frames, batch.column("Points3D:positions").to_pylist()):
            out.setdefault(frame, []).extend(pts)
    return {frame: np.asarray(pts, dtype=np.float64) for frame, pts in out.items()}


def rrd_arrows_by_frame(rrd_path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    out: dict[int, tuple[list, list]] = {}
    for frames, batch in _rrd_entity_rows(rrd_path, "/world/sr_arrows"):
        origins = batch.column("Arrows3D:origins").to_pylist()
        vectors = batch.column("Arrows3D:vectors").to_pylist()
        for frame, origin, vector in zip(frames, origins, vectors):
            entry = out.setdefault(frame, ([], []))
            entry[0].extend(origin)
            entry[1].extend(vector)
    return {
        frame: (np.asarray(origins, dtype=np.float64), np.asarray(vectors, dtype=np.float64))
        for frame, (origins, vectors) in out.items()
    }


def rrd_camera_by_frame(rrd_path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    out: dict[int, tuple[list, list]] = {}
    for frames, batch in _rrd_entity_rows(rrd_path, "/world/camera"):
        mats = batch.column("Transform3D:mat3x3").to_pylist()
        translations = batch.column("Transform3D:translation").to_pylist()
        for frame, mat, translation in zip(frames, mats, translations):
            entry = out.setdefault(frame, ([], []))
            entry[0].extend([mat])
            entry[1].extend([translation])
    return {
        frame: (
            # Rerun 0.39 の native schema: mat3x3 は列優先の flat 9列 → reshape(3,3).T で復元
            np.asarray(mats[0], dtype=np.float64).reshape(3, 3).T,
            np.asarray(translations[0], dtype=np.float64).reshape(3),
        )
        for frame, (mats, translations) in out.items()
    }


def run_worker(*args: str, env: dict | None = None, timeout: int = 240) -> subprocess.CompletedProcess:
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        env=full_env,
        timeout=timeout,
    )


def load_pipeline_module():
    """read-only で pipeline script を module として読み込む（実consumer検証用）。"""
    spec = importlib.util.spec_from_file_location("depth_detail_models_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None, (
        f"pipeline script を module として読めない: {SCRIPT}"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stage_path(root: Path, stage: str, frame: int) -> Path:
    return root / "stages" / stage / f"{frame:06d}.npz"


@pytest.fixture(scope="session")
def dataset_a(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("pipeline_sceneflow_a")
    build_dataset_a(root)
    return root


@pytest.fixture(scope="session")
def dataset_b(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("pipeline_anchor_b")
    build_dataset_b(root)
    return root


@pytest.fixture(scope="session")
def geometry_a(dataset_a) -> Path:
    proc = run_worker(
        "geometry",
        "--dataset-dir", str(dataset_a),
        "--dataset", "synthetic_a",
        "--frames", str(dataset_a / "frames.json"),
        "--clips-dir", str(dataset_a / "clips"),
        "--alignments", str(dataset_a / "alignments.json"),
    )
    assert proc.returncode == 0, (
        f"geometry(scenario A) failed rc={proc.returncode}\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    return dataset_a


@pytest.fixture(scope="session")
def geometry_b(dataset_b) -> Path:
    proc = run_worker(
        "geometry",
        "--dataset-dir", str(dataset_b),
        "--dataset", "synthetic_b",
        "--frames", str(dataset_b / "frames.json"),
        "--clips-dir", str(dataset_b / "clips"),
        "--alignments", str(dataset_b / "alignments.json"),
    )
    assert proc.returncode == 0, (
        f"geometry(scenario B) failed rc={proc.returncode}\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    return dataset_b


@pytest.fixture(scope="session")
def sceneflow_a(geometry_a) -> Path:
    proc = run_worker(
        "sceneflow_prep",
        "--dataset-dir", str(geometry_a),
        "--dataset", "synthetic_a",
        "--frames", str(geometry_a / "frames.json"),
        "--clips-dir", str(geometry_a / "clips"),
        "--long-side", str(LONG_SIDE_A),
    )
    assert proc.returncode == 0, (
        f"sceneflow_prep failed rc={proc.returncode}\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    return geometry_a


# ---------------------------------------------------------------------------
# 1) geometry: [B,N,...]保持 + inv(P0)@Pi + det(R)=1 + scale once
# ---------------------------------------------------------------------------


def test_synthetic_fixture_schema_roundtrip(dataset_a):
    from arrow.utils.results_io import load_result

    result = load_result(dataset_a / "clips" / CLIP0_NAME)
    assert result.metadata["format_version"] == 1
    assert result.queries.xy.ndim == 3 and result.queries.xy.shape[0] == 1
    assert result.predictions.points.ndim == 3 and result.predictions.points.shape[0] == 1
    assert result.metadata["groups"][0]["task"] == "pose"
    assert result.metadata["groups"][1]["reference_frame"] == "source"


def test_geometry_batch_dims_and_ref0_conversion(geometry_b):
    cam = cameras()
    for frame in range(CLIP_FRAMES):
        path = _stage_path(geometry_b, "geometry", frame)
        assert path.is_file(), f"geometry stage missing for frame {frame}"
        with np.load(path, allow_pickle=False) as data:
            c2r = np.asarray(data["arrow_camera_to_reference"], dtype=np.float64)
            recon = np.asarray(data["arrow_camera_to_reconstruct"], dtype=np.float64)
            metric = np.asarray(data["arrow_camera_to_metric"], dtype=np.float64)
            K = np.asarray(data["arrow_K"], dtype=np.float64)
            k_valid = bool(data["arrow_K_valid"])
        expected_c2r = np.linalg.inv(cam[REF_LOCAL]) @ cam[frame]
        expected_recon = np.linalg.inv(cam[0]) @ cam[frame]  # inv(P0) @ Pi
        assert np.allclose(c2r, expected_c2r, atol=1e-4), f"frame {frame}: c2r != inv(P_mid)@P_i"
        assert np.allclose(recon, expected_recon, atol=1e-4), f"frame {frame}: reconstruct != inv(P0)@Pi"
        assert abs(np.linalg.det(c2r[:3, :3]) - 1.0) < 1e-6, f"frame {frame}: det(R) != 1"
        assert abs(np.linalg.det(metric[:3, :3]) - 1.0) < 1e-6
        assert np.allclose(metric[:3, :3], expected_recon[:3, :3], atol=1e-4)
        # scale は translation へ 1 回だけ（2 回適用なら 4 倍になる）
        assert np.allclose(metric[:3, 3], POSE_SCALE_FACTOR * expected_recon[:3, 3], atol=1e-6)
        if np.linalg.norm(expected_recon[:3, 3]) > 0.03:
            assert not np.allclose(
                metric[:3, 3], (POSE_SCALE_FACTOR**2) * expected_recon[:3, 3], atol=1e-3
            ), "translation に scale が 2 回適用されている"
        assert k_valid, f"frame {frame}: K が valid 判定されていない"
        assert np.allclose(K, k_original(frame), atol=1e-3), f"frame {frame}: K 復元不一致"


def test_anchor_depth_scale_applied_once(dataset_b):
    proc = run_worker(
        "anchor",
        "--dataset-dir", str(dataset_b),
        "--dataset", "synthetic_b",
        "--frames", str(dataset_b / "frames.json"),
        "--clips-dir", str(dataset_b / "clips"),
        "--alignments", str(dataset_b / "alignments.json"),
    )
    assert proc.returncode == 0, (
        f"anchor failed rc={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    affinity = ORIGINAL_AFFINE
    expected_pixels = []
    for u_in in DEPTH_U:
        for v_in in DEPTH_V:
            u_orig = (u_in - affinity[0, 2]) / affinity[0, 0]
            v_orig = (v_in - affinity[1, 2]) / affinity[1, 1]
            expected_pixels.append((int(np.rint(v_orig)), int(np.rint(u_orig))))
    assert len(set(expected_pixels)) == DEPTH_GRID * DEPTH_GRID, "合成depthグリッドが衝突"
    for frame in range(CLIP_FRAMES):
        with np.load(_stage_path(dataset_b, "anchor", frame), allow_pickle=False) as data:
            depth = np.asarray(data["arrow_depth"], dtype=np.float64)
            valid = np.asarray(data["arrow_valid"], dtype=bool)
            scale = float(data["arrow_scale"])
            support = int(data["arrow_support_count"])
        assert scale == pytest.approx(POSE_SCALE_FACTOR)
        assert int(valid.sum()) == DEPTH_GRID * DEPTH_GRID, f"frame {frame}: valid 数不一致"
        assert support == DEPTH_GRID * DEPTH_GRID
        for row, col in expected_pixels:
            assert depth[row, col] == pytest.approx(
                POSE_SCALE_FACTOR * Z0, abs=1e-5
            ), f"frame {frame}: depth に scale が 1 回適用されていない"
        sample = expected_pixels[0]
        assert not np.allclose(depth[sample], (POSE_SCALE_FACTOR**2) * Z0, atol=1e-3)
        assert not np.allclose(depth[sample], Z0, atol=1e-3)


# ---------------------------------------------------------------------------
# 2) sceneflow_prep: inv(M_a)@M_b 方向 / sourceKb・targetKa / half-pixel
# ---------------------------------------------------------------------------


def _independent_reproject(transform, K_source, K_target, depth_source, out_shape):
    """テスト独立の最小再投影（integer pixel centre、rint、z-buffer=最近傍）。"""
    out_h, out_w = int(out_shape[0]), int(out_shape[1])
    rows, cols = np.mgrid[0 : depth_source.shape[0], 0 : depth_source.shape[1]]
    mask = np.isfinite(depth_source) & (depth_source > 0.0)
    z = depth_source[mask]
    x = (cols[mask] - K_source[0, 2]) * z / K_source[0, 0]
    y = (rows[mask] - K_source[1, 2]) * z / K_source[1, 1]
    pts = np.stack([x, y, z, np.ones_like(z)], axis=0)
    moved = transform @ pts
    zt = moved[2]
    u = K_target[0, 0] * moved[0] / zt + K_target[0, 2]
    v = K_target[1, 1] * moved[1] / zt + K_target[1, 2]
    ok = zt > 0.0
    col = np.rint(u).astype(np.int64)
    row = np.rint(v).astype(np.int64)
    ok &= (col >= 0) & (col < out_w) & (row >= 0) & (row < out_h)
    depth_out = np.full((out_h, out_w), np.nan)
    valid_out = np.zeros((out_h, out_w), dtype=bool)
    if ok.any():
        idx = np.nonzero(ok)[0]
        order = np.argsort(zt[idx], kind="stable")
        flat = row[idx][order] * out_w + col[idx][order]
        _, first = np.unique(flat, return_index=True)
        winners = idx[order][first]
        depth_out[row[winners], col[winners]] = zt[winners]
        valid_out[row[winners], col[winners]] = True
    return depth_out, valid_out


def test_sceneflow_pairs_counts_and_cross_clip_exclusion(sceneflow_a):
    payload = json.loads(
        (sceneflow_a / "stages" / "sceneflow" / "pairs.json").read_text(encoding="utf-8")
    )
    # 44フレーム連結で43ペア。境界ペア (19,20) も frame20 が clip0 に含まれるため
    # 共通clip（低index優先）で処理され、skip は発生しない。
    assert payload["counts"]["pairs"] == 43
    assert payload["counts"]["cross_clip_skipped"] == 0
    assert payload["counts"]["written"] == 43
    files = [item["file"] for item in payload["pairs"]]
    assert any("pair_000000_000001" in path for path in files)
    assert any("pair_000019_000020" in path for path in files)


def test_sceneflow_pair_direction_sourcek_and_half_pixel(sceneflow_a):
    cam = cameras()
    geom = low_geometry()
    inputs_dir = sceneflow_a / "stages" / "sceneflow" / "pairs" / "inputs"
    with np.load(inputs_dir / "pair_000000_000001.npz", allow_pickle=False) as data:
        K_low = np.asarray(data["K_low"], dtype=np.float64)
        K_low_source = np.asarray(data["K_low_source"], dtype=np.float64)
        depth1_in_0 = np.asarray(data["depth1_in_0"], dtype=np.float64)
        valid1_in_0 = np.asarray(data["valid1_in_0"], dtype=bool)
        metadata = json.loads(str(data["metadata"]))
    # target=frame0 のK、source=frame1 のK（swap 検出可能な非対称K）
    assert np.allclose(K_low, expected_k_low(0), atol=1e-6), "targetK_a 不一致"
    assert np.allclose(K_low_source, expected_k_low(1), atol=1e-6), "sourceK_b 不一致"
    assert not np.allclose(K_low, K_low_source, atol=1e-4), "Kのsource/target入替を検出できない"
    # 方向: inv(M_a)@M_b（M_f = C_f）
    expected_transform = np.linalg.inv(cam[0]) @ cam[1]
    stored = np.asarray(metadata["target_from_source"], dtype=np.float64)
    assert np.allclose(stored, expected_transform, atol=1e-4), "方向が inv(M_a)@M_b でない"
    reversed_transform = np.linalg.inv(cam[1]) @ cam[0]
    assert not np.allclose(stored, reversed_transform, atol=1e-6), "逆方向と区別できていない"
    # 独立再投影（half-pixel low K と direction の一貫性）
    depth_low = np.full(geom["low_size"], Z0, dtype=np.float64)
    ind_depth, ind_valid = _independent_reproject(
        expected_transform, expected_k_low(1), expected_k_low(0), depth_low, geom["low_size"]
    )
    mismatches = int(np.count_nonzero(ind_valid != valid1_in_0))
    assert mismatches <= 4, f"valid mask 不一致 pixels={mismatches}"
    both = ind_valid & valid1_in_0
    assert both.sum() > 0
    assert np.allclose(depth1_in_0[both], ind_depth[both], atol=1e-4), "再投影depth不一致"


def test_sceneflow_pair_inside_clip1_uses_aligned_world(sceneflow_a):
    cam = cameras()
    inputs_dir = sceneflow_a / "stages" / "sceneflow" / "pairs" / "inputs"
    with np.load(inputs_dir / "pair_000030_000031.npz", allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
    expected = np.linalg.inv(cam[30]) @ cam[31]
    stored = np.asarray(metadata["target_from_source"], dtype=np.float64)
    assert np.allclose(stored, expected, atol=1e-4), "clip1 内ペアのworld整合"


# ---------------------------------------------------------------------------
# 3) overlap 境界: frame20 は別 clip 由来でも world c2w で連続
# ---------------------------------------------------------------------------


def test_overlap_boundary_uses_world_c2w_without_raw_camera_mix(geometry_a):
    cam = cameras()
    metrics = {}
    c2rs = {}
    for frame in range(16, 24):
        with np.load(_stage_path(geometry_a, "geometry", frame), allow_pickle=False) as data:
            metrics[frame] = np.asarray(data["arrow_camera_to_metric"], dtype=np.float64)
            c2rs[frame] = np.asarray(data["arrow_camera_to_reference"], dtype=np.float64)
    for frame in range(16, 24):
        assert np.allclose(metrics[frame], cam[frame], atol=1e-4), (
            f"frame {frame}: world c2w が不連続（raw camera 参照の混在疑い）"
        )
    # frame16..19 は clip_000（ref=12）、frame20..23 は後 start の clip_020（ref=32）が最終書き込み
    for frame in range(16, 20):
        assert np.allclose(c2rs[frame], np.linalg.inv(cam[12]) @ cam[frame], atol=1e-4)
    for frame in range(20, 24):
        assert np.allclose(c2rs[frame], np.linalg.inv(cam[32]) @ cam[frame], atol=1e-4), (
            f"frame {frame}: clip_020 の local 参照が保存されていない"
        )
    clip0_local = np.linalg.inv(cam[12]) @ cam[20]
    assert not np.allclose(c2rs[20], clip0_local, atol=1e-3), "raw camera 参照が混ざっている"
    step = np.linalg.inv(metrics[19]) @ metrics[20]
    expected_step = np.linalg.inv(cam[19]) @ cam[20]
    assert np.allclose(step, expected_step, atol=1e-4), "境界ステップが world で連続していない"


# ---------------------------------------------------------------------------
# 4) --device cuda 利用不可は CPU fallback せず fail
# ---------------------------------------------------------------------------


def test_lingbot_cuda_unavailable_fails_without_cpu_fallback(dataset_b):
    proc = run_worker(
        "lingbot",
        "--dataset-dir", str(dataset_b),
        "--dataset", "synthetic_b",
        "--frames", str(dataset_b / "frames.json"),
        "--device", "cuda",
        env={"CUDA_VISIBLE_DEVICES": ""},
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, "CUDA不可時は fail すべき（CPU fallback 禁止）"
    assert "利用できません" in combined, f"明示エラーメッセージがない:\n{combined}"
    assert "cuda" in combined.lower()
    assert not (dataset_b / "stages" / "lingbot").exists(), "失敗時に lingbot 出力が残っている"


# ---------------------------------------------------------------------------
# 5) motion_candidate_from_sr: 非0 residual motion の解析fixture
# ---------------------------------------------------------------------------


def test_motion_candidate_from_sr_residual_motion_analytic():
    module = load_pipeline_module()
    fn = getattr(module, "motion_candidate_from_sr", None)
    assert fn is not None, (
        "motion_candidate_from_sr が pipeline script に未実装。"
        "統括 REVIEW_MOTION_CORRESPONDENCE / API_FIX_MOTION_CANDIDATE の helper が必要"
    )
    h, w = 90, 128
    K0 = np.array([[140.0, 0.0, 100.0], [0.0, 132.0, 60.0], [0.0, 0.0, 1.0]])
    K1 = K0.copy()
    translation = np.array([0.02, 0.01, 0.0])
    M0 = np.eye(4)
    M1 = _rigid(np.eye(3), translation)
    T = np.linalg.inv(M1) @ M0  # frame1 <- frame0

    z_bg = 3.0
    s_obj = np.array([0.05, 0.0, 0.06])  # 非0 residual motion（カメラ補償後の物体運動）
    depth0 = np.full((h, w), z_bg)
    sr_flow = np.zeros((h, w, 3))
    sr_valid = np.zeros((h, w), dtype=bool)
    rows, cols = np.mgrid[0:h, 0:w]
    object_mask = (rows >= 28) & (rows < 68) & (cols >= 40) & (cols < 100)
    sr_flow[object_mask] = s_obj
    sr_valid[object_mask] = True

    # frame1 観測: 動体は fronto-parallel 平面 z1 = z_bg + s_z（矩形の像）
    z1 = z_bg + s_obj[2]
    x0_lo = (40.0 - K0[0, 2]) / K0[0, 0] * z_bg
    x0_hi = (100.0 - K0[0, 2]) / K0[0, 0] * z_bg
    y0_lo = (28.0 - K0[1, 2]) / K0[1, 1] * z_bg
    y0_hi = (68.0 - K0[1, 2]) / K0[1, 1] * z_bg
    u1_lo = K1[0, 0] * (x0_lo + s_obj[0] - translation[0]) / z1 + K1[0, 2]
    u1_hi = K1[0, 0] * (x0_hi + s_obj[0] - translation[0]) / z1 + K1[0, 2]
    v1_lo = K1[1, 1] * (y0_lo + s_obj[1] - translation[1]) / z1 + K1[1, 2]
    v1_hi = K1[1, 1] * (y0_hi + s_obj[1] - translation[1]) / z1 + K1[1, 2]
    depth1 = np.full((h, w), z_bg)
    rect1 = (
        (cols >= np.ceil(u1_lo)) & (cols <= np.floor(u1_hi))
        & (rows >= np.ceil(v1_lo)) & (rows <= np.floor(v1_hi))
    )
    depth1[rect1] = z1
    nan_region = (rows >= 47) & (rows < 56) & (cols >= 52) & (cols < 61)
    depth1[nan_region] = np.nan
    trust_pixel = (40, 60)
    sr_flow[trust_pixel] = np.array([0.05, 0.0, 0.26])  # |D1obs - zpred1| = 0.2 > 0.05

    out = fn(depth0, K0, depth1, K1, T, sr_flow, sr_valid, 0.05, 0.05)
    for key in ("candidate", "support", "zpred1", "d1_observed", "q1"):
        assert key in out, f"motion_candidate_from_sr の key {key} がない: {sorted(out)}"
    support = np.asarray(out["support"], dtype=bool)
    candidate = np.asarray(out["candidate"], dtype=np.float64)
    zpred1 = np.asarray(out["zpred1"], dtype=np.float64)
    d1_observed = np.asarray(out["d1_observed"], dtype=np.float64)
    q1 = np.asarray(out["q1"], dtype=np.float64)
    assert support.shape == (h, w) and candidate.shape == (h, w)
    assert zpred1.shape == (h, w) and d1_observed.shape == (h, w)
    assert q1.shape == (h, w, 2)

    assert not support[~object_mask].any(), "SR valid=False では support=False"
    nan_affect = (rows >= 46) & (rows < 57) & (cols >= 49) & (cols < 60)
    interior = (
        (rows >= 33) & (rows < 63) & (cols >= 45) & (cols < 95) & object_mask & ~nan_affect
    )
    interior[trust_pixel] = False
    assert support[interior].all(), "支持があるはずの動体内部で support=False"
    assert np.allclose(candidate[interior], z_bg, atol=1e-6), (
        "candidate が (inv(T)@[D1obs*r1] - S).z になっていない"
    )
    assert not np.allclose(candidate[interior], z1, atol=1e-3), "残差運動 S が無視されている"
    assert np.allclose(zpred1[interior], z1, atol=1e-6)
    assert np.allclose(d1_observed[interior], z1, atol=1e-6)

    r0, c0 = 48, 70
    ray0 = np.array(
        [(c0 - K0[0, 2]) / K0[0, 0], (r0 - K0[1, 2]) / K0[1, 1], 1.0]
    ) * z_bg
    p1_0 = ray0 + s_obj
    p1_1 = T @ np.append(p1_0, 1.0)
    expected_q1 = np.array(
        [K1[0, 0] * p1_1[0] / p1_1[2] + K1[0, 2], K1[1, 1] * p1_1[1] / p1_1[2] + K1[1, 2]]
    )
    assert np.allclose(q1[r0, c0], expected_q1, atol=1e-6), "q1 投影が解析値と不一致"

    assert not support[trust_pixel], "trust 違反 (|D1obs - zpred1| > trust) で support=True"
    assert not support[51, 55], "D1obs 支持なし (NaN近傍) で support=True"
    assert np.isnan(d1_observed[51, 55])


# ---------------------------------------------------------------------------
# 6) splat_scene_flow: 非0 XY の forward splat（identity 再投影の拒否）
# ---------------------------------------------------------------------------


def test_splat_scene_flow_nonzero_xy_analytic():
    module = load_pipeline_module()
    fn = getattr(module, "splat_scene_flow", None)
    assert fn is not None, (
        "splat_scene_flow が pipeline script に未実装。"
        "統括 REVIEW_BACKWARD_XY の helper (P1_0 の forward z-buffer splat) が必要"
    )
    # 統括の解析例: fixed camera + Z=2 平面、fx=fy=2、S=(1,0,0)
    # → fx*S_x/Z = 1 pixel 右へ splat。最左列の支持は落ちる。
    h, w = 4, 6
    fx = fy = 2.0
    K = np.array([[fx, 0.0, 0.0], [0.0, fy, 0.0], [0.0, 0.0, 1.0]])
    depth = np.full((h, w), 2.0)
    flow = np.zeros((h, w, 3))
    flow[..., 0] = 1.0
    valid = np.ones((h, w), dtype=bool)
    out = fn(depth, K, K, np.eye(4), flow, valid)
    for key in ("depth", "valid"):
        assert key in out, f"splat_scene_flow の key {key} がない: {sorted(out)}"
    splat_depth = np.asarray(out["depth"], dtype=np.float64)
    splat_valid = np.asarray(out["valid"], dtype=bool)
    assert splat_depth.shape == (h, w) and splat_valid.shape == (h, w)
    # depth だけの再投影なら identity のまま全列 valid → 非0 XY の splat では拒否される
    assert not splat_valid[:, 0].any(), "非0 XY が無視されている（identity 再投影と同値）"
    assert splat_valid[:, 1:].all()
    assert int(splat_valid.sum()) == h * (w - 1)
    assert np.allclose(splat_depth[splat_valid], 2.0, atol=1e-9)
    assert np.all(np.isnan(splat_depth[~splat_valid]))


# ---------------------------------------------------------------------------
# 7) sceneflow_post: delta0保持・trust/edge gate・correction/solver mask区別
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def dataset_post(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("pipeline_post")
    build_dataset_post(root)
    return root


@pytest.fixture(scope="session")
def sceneflow_post_run(dataset_post) -> Path:
    proc = run_worker(
        "sceneflow_post",
        "--dataset-dir", str(dataset_post),
        "--dataset", "synthetic_post",
        "--frames", str(dataset_post / "frames.json"),
        "--trust-abs", "0.05",
        "--trust-rel", "0.05",
    )
    assert proc.returncode == 0, (
        f"sceneflow_post failed rc={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    return dataset_post


@pytest.fixture(scope="session")
def sr_run(sceneflow_post_run) -> Path:
    proc = run_worker(
        "sr",
        "--dataset-dir", str(sceneflow_post_run),
        "--dataset", "synthetic_post",
        "--frames", str(sceneflow_post_run / "frames.json"),
    )
    assert proc.returncode == 0, (
        f"sr failed rc={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    return sceneflow_post_run


def test_sceneflow_post_detail_preservation_analytic(sceneflow_post_run):
    root = sceneflow_post_run
    rows, cols = np.mgrid[0:POST_H, 0:POST_W]

    # (a) delta0 == 0 → NDR が画素ごと不変（高周波detailの非劣化）
    with np.load(_stage_path(root, "sceneflow_post", 0), allow_pickle=False) as data:
        depth_a = np.asarray(data["sceneflow_depth"], dtype=np.float64)
        valid_a = np.asarray(data["sceneflow_valid"], dtype=bool)
        correction_a = np.asarray(data["sceneflow_correction_support"], dtype=bool)
        solver_a = np.asarray(data["sceneflow_solver_support"], dtype=bool)
        fallback_a = np.asarray(data["sceneflow_fallback_mask"], dtype=bool)
    expected_a = (2.0 + 0.05 * ((rows + cols) % 2)).astype(np.float32)
    assert valid_a.all(), "delta0=0 で valid が変わっている"
    assert np.array_equal(depth_a, expected_a), "delta0=0 で NDR が画素ごと不変でない"
    assert bool(correction_a.all()) and bool(solver_a.all())
    assert not fallback_a.any()

    # (b) 既知 delta0=+0.03（trust内）: (1-edges) 重みで補正、edge=1 は不変
    with np.load(_stage_path(root, "sceneflow_post", 2), allow_pickle=False) as data:
        depth_b = np.asarray(data["sceneflow_depth"], dtype=np.float64)
        valid_b = np.asarray(data["sceneflow_valid"], dtype=bool)
        correction_b = np.asarray(data["sceneflow_correction_support"], dtype=bool)
    expected_b = np.full((POST_H, POST_W), 2.03)
    expected_b[:, 4] = 2.0  # edges=1.0 → delta_eff = 0
    expected_b[:, 5] = 2.015  # edges=0.5 → 半減衰
    assert valid_b.all()
    assert np.allclose(depth_b, expected_b, atol=1e-6), "trust内delta0のedge gate補正が不一致"
    assert bool(correction_b.all())

    # (c) trust外 + unsupported: NDR維持・valid維持・correction/solver mask区別
    with np.load(_stage_path(root, "sceneflow_post", 4), allow_pickle=False) as data:
        depth_c = np.asarray(data["sceneflow_depth"], dtype=np.float64)
        valid_c = np.asarray(data["sceneflow_valid"], dtype=bool)
        correction_c = np.asarray(data["sceneflow_correction_support"], dtype=bool)
        solver_c = np.asarray(data["sceneflow_solver_support"], dtype=bool)
        fallback_c = np.asarray(data["sceneflow_fallback_mask"], dtype=bool)
        correction_count = int(data["sceneflow_correction_count"])
        solver_count = int(data["sceneflow_solver_support_count"])
    assert valid_c.all(), "unsupported 領域で NDR valid が失われている（NaN化）"
    assert np.array_equal(depth_c, np.full((POST_H, POST_W), 2.0)), "trust外/unsupportedでNDRが変わっている"
    assert correction_count == 0 and not correction_c.any()
    expected_solver_cols = 2 * (POST_LOW // 2)  # 低解像度左半分 → 高解像度4列
    assert solver_count == POST_H * expected_solver_cols, "solver_support の支持数が解析値と不一致"
    assert np.array_equal(solver_c[:, :expected_solver_cols], np.ones((POST_H, expected_solver_cols), dtype=bool))
    assert not solver_c[:, expected_solver_cols:].any()
    assert bool(fallback_c.all()), "correction なし画素が fallback として記録されていない"


# ---------------------------------------------------------------------------
# 8) SR の high depth ガイドが高解像度 NDR である（coarse再拡大ではない）
# ---------------------------------------------------------------------------


def test_sr_high_depth_guide_is_highres_ndr(sr_run):
    root = sr_run
    with np.load(_stage_path(root, "sr", 6), allow_pickle=False) as data:
        sr_valid = np.asarray(data["sr_valid"], dtype=bool)
        sr_flow = np.asarray(data["sr_flow"], dtype=np.float64)
        low_valid = np.asarray(data["sr_low_valid"], dtype=bool)
    assert low_valid.all()
    # 高解像度NDRの単一画素ホール → SR valid もその1画素だけ落ちる。
    # coarse再拡大ガイドではホールが消える/2x2以上へ広がるため、この局所性が判別になる。
    assert not sr_valid[3, 5]
    assert bool(sr_valid[3, 6]) and bool(sr_valid[2, 5]) and bool(sr_valid[4, 5])
    assert int(sr_valid.sum()) == POST_H * POST_W - 1, "NDR以外のガイドを使っている疑い（ホール局所性）"
    assert np.all(np.isfinite(sr_flow[sr_valid]))
    assert np.allclose(sr_flow[sr_valid], 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# 9) render stride>1: 実RRD読み戻し（RrdReader）でworld点群/矢印/cameraを比較
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def dataset_render(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("pipeline_render")
    build_dataset_render(root)
    return root


@pytest.fixture(scope="session")
def render_run(dataset_render) -> Path:
    proc = run_worker(
        "render",
        "--dataset-dir", str(dataset_render),
        "--dataset", "synthetic_render",
        "--frames", str(dataset_render / "frames.json"),
        "--fps", "5",
        "--cell-width", "32",
        "--point-stride", str(RENDER_STRIDE),
    )
    assert proc.returncode == 0, (
        f"render failed rc={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    return dataset_render


def test_render_rrd_points_readback_matches_analytic_geometry(render_run):
    rrd_path = render_run / "visuals" / "synthetic_render_final_pointcloud.rrd"
    assert rrd_path.is_file(), "RRD が生成されていない"
    points = rrd_points_by_frame(rrd_path)
    assert set(points) == {0, 1}, f"frame join に失敗: frames={sorted(points)}"
    total = 0
    for frame in (0, 1):
        expected = render_world_points(frame, use_stride=True)
        got = points[frame]
        assert got.shape == expected.shape, (
            f"frame {frame}: 点数不一致 got={got.shape} expected={expected.shape}"
        )
        assert np.allclose(got, expected, atol=2e-5, rtol=1e-5), (
            f"frame {frame}: world点群が native(=index*stride) 逆投影の解析値と不一致"
        )
        old_style = render_world_points(frame, use_stride=False)
        assert not np.allclose(got, old_style, atol=1e-3), (
            "strideを無視した旧式の期待値と一致してしまっている"
        )
        total += got.shape[0]
    expected_count = RENDER_FRAMES * (RENDER_H // RENDER_STRIDE) * (RENDER_W // RENDER_STRIDE)
    assert total == expected_count == 24


def test_render_rrd_arrows_and_camera_rotation_only(render_run):
    rrd_path = render_run / "visuals" / "synthetic_render_final_pointcloud.rrd"
    arrows = rrd_arrows_by_frame(rrd_path)
    assert set(arrows) == {0, 1}, "SR矢印 layer が読めない"
    for frame in (0, 1):
        expected = render_expected_arrow(frame)
        origins, vectors = arrows[frame]
        assert origins.shape == (1, 3) and vectors.shape == (1, 3), "サンプル矢印数が解析値(1)と不一致"
        assert np.allclose(origins[0], expected["origin"], atol=2e-5, rtol=1e-5), "矢印originが不一致"
        assert np.allclose(vectors[0], expected["vector"], atol=2e-5, rtol=1e-5), (
            "SR vector が回転のみ (S @ R.T) になっていない"
        )
        assert not np.allclose(vectors[0], RENDER_S, atol=1e-3), "回転がvectorへ適用されていない"
        assert not np.allclose(vectors[0], expected["vector"] + RENDER_TRANSLATION, atol=1e-3), (
            "vector に translation が混入している"
        )
    camera = rrd_camera_by_frame(rrd_path)
    assert set(camera) >= {0, 1}, "camera layer が読めない"
    for frame in (0, 1):
        mat, translation = camera[frame]
        assert np.allclose(mat, RENDER_ROTATION, atol=1e-6), "camera mat3x3 が非identity world回転と不一致"
        assert np.allclose(translation, RENDER_TRANSLATION, atol=1e-6), "camera translation が不一致"


# ---------------------------------------------------------------------------
# 11) render: 最終frameの空SRで /world/sr_arrows に明示empty row（RED regression）
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def dataset_render_empty(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("pipeline_render_empty")
    build_dataset_render_empty_arrows(root)
    return root


@pytest.fixture(scope="session")
def render_empty_run(dataset_render_empty) -> Path:
    proc = run_worker(
        "render",
        "--dataset-dir", str(dataset_render_empty),
        "--dataset", "synthetic_render_empty",
        "--frames", str(dataset_render_empty / "frames.json"),
        "--fps", "5",
        "--cell-width", "32",
        "--point-stride", str(RENDER_STRIDE),
    )
    assert proc.returncode == 0, (
        f"render(empty arrows) failed rc={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    return dataset_render_empty


def test_render_rrd_empty_final_frame_has_explicit_empty_arrows_row(render_empty_run):
    """空SRの最終frameにも /world/sr_arrows の明示rowがあること（latest-at hold-last防止）。

    現行worker_renderはsampleが空のframeをlogしないため、このtestはREDであることが
    期待値（row不存在を拒否）。root承認後、pipeline側の empty Arrows3D row 追加で green 化する。
    """
    rrd_path = render_empty_run / "visuals" / "synthetic_render_empty_final_pointcloud.rrd"
    assert rrd_path.is_file(), "RRD が生成されていない"
    arrows = rrd_arrows_by_frame(rrd_path)
    # frame0: 正の矢印（世界回転のみ）
    assert 0 in arrows, "frame0 の矢印rowがない"
    origins0, vectors0 = arrows[0]
    assert vectors0.reshape(-1, 3).shape[0] > 0, "frame0 に正の矢印がない"
    expected = render_expected_arrow(0)
    assert np.allclose(
        vectors0.reshape(-1, 3)[0], expected["vector"], atol=2e-5, rtol=1e-5
    ), "frame0 のvectorが回転のみになっていない"
    # frame1: 明示empty row（row不存在はhold-lastのため拒否してREDにする）
    assert 1 in arrows, (
        "空SRのframe1に /world/sr_arrows の明示rowが無い（latest-atで前frameの矢印が残る）"
    )
    origins1, vectors1 = arrows[1]
    assert origins1.reshape(-1, 3).shape[0] == 0 and vectors1.reshape(-1, 3).shape[0] == 0, (
        "frame1 のrowは存在するが空でない"
    )


# ---------------------------------------------------------------------------
# 10) worker_prep: K_valid の Python bool 化（metadata JSON 実保存回帰）
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def dataset_prep(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("pipeline_prep")
    build_dataset_prep(root)
    return root


def test_worker_prep_k_valid_json_serializable_and_normals(dataset_prep):
    proc = run_worker(
        "prep",
        "--dataset-dir", str(dataset_prep),
        "--dataset", "synthetic_prep",
        "--frames", str(dataset_prep / "frames.json"),
    )
    assert proc.returncode == 0, (
        f"prep failed rc={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    for frame, expected_k_valid in ((0, True), (1, False), (2, False)):
        with np.load(_stage_path(dataset_prep, "prep", frame), allow_pickle=False) as data:
            edges = np.asarray(data["prep_edges"], dtype=np.float64)
            normals = np.asarray(data["prep_normal"], dtype=np.float64)
            normal_valid = np.asarray(data["prep_normal_valid"], dtype=bool)
            count = int(data["prep_normal_count"])
            # 実保存された metadata JSON を実読込（np.bool_混入ならここまで届かない）
            metadata = json.loads(str(data["metadata"]))
        assert edges.shape == (PREP_H, PREP_W)
        assert edges.min() >= 0.0 and edges.max() <= 1.0
        assert isinstance(metadata["params"]["k_valid"], bool), (
            f"frame {frame}: k_valid が Python bool でない: {type(metadata['params']['k_valid'])!r}"
        )
        assert metadata["params"]["k_valid"] is expected_k_valid
        assert count == int(normal_valid.sum())
        if expected_k_valid:
            assert int(normal_valid.sum()) > 0, f"frame {frame}: finite K で normal が無効"
            valid_normals = normals[normal_valid]
            assert np.all(np.isfinite(valid_normals))
            lengths = np.linalg.norm(valid_normals, axis=-1)
            assert np.allclose(lengths, 1.0, atol=1e-3), f"frame {frame}: 法線が単位長でない"
            assert np.all(np.abs(valid_normals[:, 2]) > 0.99), (
                f"frame {frame}: 平面depthの法線がz向きでない"
            )
        else:
            assert int(normal_valid.sum()) == 0, f"frame {frame}: invalid K で normal が有効"
            assert np.all(np.isnan(normals)), f"frame {frame}: invalid K の法線がNaNでない"
    config = json.loads(
        (dataset_prep / "configs" / "stage_configs" / "prep.json").read_text(encoding="utf-8")
    )
    assert config["frames_with_normals"] >= 1
