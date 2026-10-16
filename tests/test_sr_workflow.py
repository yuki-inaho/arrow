"""``scripts/render_sr_workflow.py`` の focused tests。

方針: numpy / Pillow / ffprobe だけで完結する小型の合成 data を使い、
表示用縮小・座標変換・SR 移流・空ペアの log・実 export（ffprobe/RRD 構造）を確認する。
入力 NPZ を一切書き換えないこと（source 不変）も検証する。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import render_sr_workflow as srw  # noqa: E402


# ---------------------------------------------------------------------------
# sparse depth の表示用縮小（旧 bilinear の暗さ対策）


def test_view_bounds_do_not_collapse_scene_for_one_extreme_point():
    grid = np.stack(np.meshgrid(np.linspace(-1, 1, 12), np.linspace(-1, 1, 12),
                               np.linspace(2, 4, 12)), axis=-1).reshape(-1, 3)
    points = np.vstack([grid, [1e6, 1e6, 1e6]])
    original = points.copy()
    bounds = srw.compute_scene_bounds([points], np.array([[0., 0., 0.]]))
    assert all(0 < high - low < 10 for low, high in bounds.values())
    np.testing.assert_array_equal(points, original)


def test_footprint_shrink_keeps_valid_scalar_color_not_dark_average():
    depth = np.full((4, 4), np.nan)
    valid = np.zeros((4, 4), dtype=bool)
    depth[1, 1] = 5.0
    valid[1, 1] = True
    out, coverage = srw.shrink_depth_footprint(depth, valid, (2, 2))
    assert out[0, 0] == pytest.approx(5.0)
    assert np.isnan(out).sum() == 3
    assert coverage[0, 0] == pytest.approx(0.25)
    assert coverage[1, 1] == 0.0
    # 元解像度で colorize した同じ画素の色と一致する（invalid の暗色が混ざらない）
    full = srw.colorize_scalar(depth, 0.0, 10.0, valid=valid)
    shrunk = srw.colorize_scalar(out, 0.0, 10.0, valid=np.isfinite(out))
    assert np.array_equal(shrunk[0, 0], full[1, 1])
    dark_mix = full[1, 1].astype(np.float64) * 0.25 + np.array([20, 20, 25]) * 0.75
    assert not np.allclose(shrunk[0, 0].astype(np.float64), dark_mix)


def test_footprint_shrink_empty_mask_stays_background():
    depth = np.full((4, 4), np.nan)
    valid = np.zeros((4, 4), dtype=bool)
    out, coverage = srw.shrink_depth_footprint(depth, valid, (2, 2))
    assert np.isnan(out).all()
    assert float(coverage.sum()) == 0.0
    colored = srw.colorize_scalar(out, 0.0, 10.0, valid=np.isfinite(out))
    assert np.array_equal(colored, np.full((2, 2, 3), [20, 20, 25], dtype=np.uint8))


def test_footprint_shrink_mixed_block_and_inputs_unchanged():
    depth = np.array([[np.nan, np.nan], [np.nan, 5.0]])
    valid = np.array([[False, False], [False, True]])
    depth_before = depth.copy()
    valid_before = valid.copy()
    out, coverage = srw.shrink_depth_footprint(depth, valid, (1, 1))
    assert out[0, 0] == pytest.approx(5.0)
    assert coverage[0, 0] == pytest.approx(0.25)
    assert np.array_equal(depth, depth_before, equal_nan=True)
    assert np.array_equal(valid, valid_before)
    # 有効画素は「元の値」のままで、周囲の invalid に引っ張られない
    assert not np.isnan(out[0, 0])


def test_footprint_shrink_rejects_zero_shape():
    depth = np.ones((2, 2))
    valid = np.ones((2, 2), dtype=bool)
    with pytest.raises(ValueError):
        srw.shrink_depth_footprint(depth, valid, (0, 2))


def test_footprint_shrink_handles_output_larger_than_input():
    """出力 > 入力（アップスケール）でも有効画素の色を保ち、ゼロ面積除算を起こさない。"""
    depth = np.full((2, 2), np.nan)
    valid = np.zeros((2, 2), dtype=bool)
    depth[0, 0] = 5.0
    valid[0, 0] = True
    out, coverage = srw.shrink_depth_footprint(depth, valid, (4, 4))
    assert out.shape == (4, 4) and coverage.shape == (4, 4)
    assert out[0, 0] == pytest.approx(5.0)
    assert out[1, 1] == pytest.approx(5.0)
    assert np.isnan(out[2, 2]) and coverage[2, 2] == 0.0
    assert np.isfinite(out[valid.repeat(2, 0).repeat(2, 1)]).all()
    colored = srw.colorize_scalar(out, 0.0, 10.0, valid=np.isfinite(out))
    full = srw.colorize_scalar(depth, 0.0, 10.0, valid=valid)
    assert np.array_equal(colored[0, 0], full[0, 0])


# ---------------------------------------------------------------------------
# ARROW 表示（native anchor 格子 → NEAREST 拡大、非格子は面積加重へ）


def test_anchor_grid_axes_detects_regular_sampling():
    valid = np.zeros((64, 48), dtype=bool)
    valid[1::5, 2::6] = True
    rows, cols, used = srw.anchor_grid_axes(valid)
    assert used
    assert np.array_equal(rows, np.arange(1, 64, 5))
    assert np.array_equal(cols, np.arange(2, 48, 6))
    rng = np.random.default_rng(3)
    scattered = np.zeros((64, 48), dtype=bool)
    scattered[rng.integers(0, 64, 200), rng.integers(0, 48, 200)] = True
    _, _, used_scattered = srw.anchor_grid_axes(scattered)
    assert not used_scattered
    empty = np.zeros((8, 8), dtype=bool)
    assert srw.anchor_grid_axes(empty)[2] is False


def test_sparse_arrow_display_grid_bins_then_nearest_upscales():
    valid = np.zeros((30, 20), dtype=bool)
    depth = np.full((30, 20), np.nan)
    values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    for i, row in enumerate((0, 10, 20)):
        for j, col in enumerate((0, 10)):
            valid[row, col] = True
            depth[row, col] = values[i, j]
    valid[20, 10] = False  # 1 セルだけ invalid
    depth[20, 10] = np.nan
    colored, coverage, used = srw.sparse_arrow_display(depth, valid, (12, 8), 1.0, 6.0)
    assert used
    assert coverage == pytest.approx(5.0 / 600.0)
    assert colored.shape == (12, 8, 3)
    full = srw.colorize_scalar(depth, 1.0, 6.0, valid=valid)
    assert np.array_equal(colored[0, 0], full[0, 0])
    block = colored[0:4, 0:4].reshape(-1, 3)  # セル (0,0) は均一ブロック（混色なし）
    assert (block == block[0]).all()
    # invalid セルは背景のまま（黒混じりの平均を作らない）
    assert np.array_equal(colored[-1, -1], np.array([20, 20, 25], dtype=np.uint8))


def test_sparse_arrow_display_falls_back_to_area_for_scatter():
    rng = np.random.default_rng(5)
    depth = np.full((40, 30), np.nan)
    valid = np.zeros((40, 30), dtype=bool)
    ys = rng.integers(0, 40, 120)
    xs = rng.integers(0, 30, 120)
    depth[ys, xs] = 4.0
    valid[ys, xs] = True
    colored, coverage, used = srw.sparse_arrow_display(depth, valid, (10, 12), 0.0, 10.0)
    assert not used
    assert coverage == pytest.approx(float(valid.sum()) / (40 * 30))
    assert colored.shape == (10, 12, 3)
    assert colored.dtype == np.uint8


# ---------------------------------------------------------------------------
# 座標系（native K・stride・(x, z, -y) Z-up・rotation-only）


def test_cloud_stride_uses_native_k_and_zup_display():
    height, width = 8, 6
    K = np.array([[100.0, 0.0, 30.0], [0.0, 120.0, 40.0], [0.0, 0.0, 1.0]])
    depth = np.full((height, width), 2.0)
    valid = np.ones((height, width), dtype=bool)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(height)[:, None]
    cloud = srw.cloud_for_frame(depth, valid, K, np.eye(4), rgb, stride=2)
    rows, cols = np.meshgrid(np.arange(0, height, 2), np.arange(0, width, 2), indexing="ij")
    native_y, native_x = rows.ravel(), cols.ravel()
    z = np.full(native_x.shape, 2.0)
    expected = np.stack(
        [(native_x - 30.0) * z / 100.0, (native_y - 40.0) * z / 120.0, z], axis=-1
    )
    assert np.allclose(cloud["points"], expected)
    # RGB は native pixel から取る（stride 縮小しても色の index がずれない）
    assert np.array_equal(cloud["rgb"][:, 0], native_y.astype(np.uint8))
    disp = srw.display(cloud["points"])
    assert np.allclose(disp[:, 0], expected[:, 0])
    assert np.allclose(disp[:, 1], expected[:, 2])
    assert np.allclose(disp[:, 2], -expected[:, 1])


def test_rotation_only_vectors_do_not_translate():
    angle = 0.7
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    metric = np.eye(4)
    metric[:3, :3] = rotation
    metric[:3, 3] = np.array([10.0, 20.0, 30.0])
    flow = np.array([[1.0, 2.0, 3.0], [0.5, -0.25, 1.0]])
    vectors = srw.rotation_only(flow, metric)
    assert np.allclose(vectors, flow @ rotation.T)
    assert np.allclose(np.linalg.norm(vectors, axis=-1), np.linalg.norm(flow, axis=-1))
    point = srw.to_metric(flow[0], metric)
    assert np.allclose(point, flow[0] @ rotation.T + metric[:3, 3])


def test_motion_2d_matches_projection_of_sr_displacement():
    height, width = 4, 5
    K = np.array([[100.0, 0.0, 2.0], [0.0, 200.0, 1.5], [0.0, 0.0, 1.0]])
    depth = np.full((height, width), 2.0)
    valid = np.ones((height, width), dtype=bool)
    flow = np.zeros((height, width, 3))
    flow[..., 0] = 0.6
    sr_valid = np.ones((height, width), dtype=bool)
    du, dv, ok = srw.sr_motion_2d(flow, sr_valid, depth, valid, K)
    assert ok.all()
    assert np.allclose(du, K[0, 0] * 0.6 / 2.0)
    assert np.allclose(dv, 0.0)
    # 残差ゼロなら移動ゼロ
    zeros = np.zeros((height, width, 3))
    du0, dv0, _ = srw.sr_motion_2d(zeros, sr_valid, depth, valid, K)
    assert np.allclose(du0, 0.0) and np.allclose(dv0, 0.0)


def test_normals_from_depth_face_camera():
    height, width = 8, 8
    K = np.array([[50.0, 0.0, 4.0], [0.0, 50.0, 4.0], [0.0, 0.0, 1.0]])
    depth = np.full((height, width), 3.0)
    valid = np.ones((height, width), dtype=bool)
    normals, ok = srw.normals_from_depth(depth, valid, K, stride=1)
    assert ok[2:-2, 2:-2].all()
    assert np.allclose(np.abs(normals[2:-2, 2:-2, 2]), 1.0, atol=1e-5)
    assert (normals[2:-2, 2:-2, 2] < 0).all()  # 前向きの面は camera 向き (-z)


# ---------------------------------------------------------------------------
# SR 変位サンプルと空ペア


def test_arrow_samples_empty_without_pair():
    height, width = 6, 6
    depth = np.full((height, width), 2.0)
    valid = np.ones((height, width), dtype=bool)
    sr_valid = np.zeros((height, width), dtype=bool)
    origins, vectors = srw.sr_arrow_samples(None, sr_valid, depth, valid, np.eye(3), np.eye(4))
    assert origins.shape == (0, 3) and vectors.shape == (0, 3)
    # pair はあるが valid 画素ゼロ（最後の frame）でも空で返る
    flow = np.full((height, width, 3), np.nan)
    origins, vectors = srw.sr_arrow_samples(flow, sr_valid, depth, valid, np.eye(3), np.eye(4))
    assert origins.shape == (0, 3) and vectors.shape == (0, 3)
    # 有効画素では stride グリッド分だけ返り、変位は rotation-only で写る
    sr_valid[::2, ::2] = True
    flow[::2, ::2] = (0.1, 0.0, 0.0)
    angle = 0.4
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    metric = np.eye(4)
    metric[:3, :3] = rotation
    metric[:3, 3] = [5.0, 6.0, 7.0]
    origins, vectors = srw.sr_arrow_samples(flow, sr_valid, depth, valid, np.eye(3), metric, stride=2)
    assert origins.shape == (9, 3)
    assert np.allclose(vectors, np.array([0.1, 0.0, 0.0]) @ rotation.T)


def test_seed_pixels_deterministic_and_bounded():
    sr_valid = np.zeros((40, 60), dtype=bool)
    sr_valid[5:35, 5:55] = True
    valid = np.ones((40, 60), dtype=bool)
    seeds_a = srw.seed_pixels(sr_valid, valid, picks=10)
    seeds_b = srw.seed_pixels(sr_valid, valid, picks=10)
    assert np.array_equal(seeds_a, seeds_b)
    assert len(seeds_a) <= 10
    assert (sr_valid[seeds_a[:, 0], seeds_a[:, 1]]).all()
    assert len(srw.seed_pixels(np.zeros((4, 4), bool), np.ones((4, 4), bool))) == 0


def test_finite_strips_split_on_dead_gap():
    line = np.array([[0.0, 0, 0], [1.0, 0, 0], [np.nan, 0, 0], [3.0, 0, 0], [4.0, 0, 0]])
    strips = srw.finite_strips(line)
    assert len(strips) == 2
    assert len(strips[0]) == 2 and len(strips[1]) == 2
    assert np.isnan(line).any()  # 入力は変更しない


# ---------------------------------------------------------------------------
# SR 移流（real advection）


def _advection_frame(depth_z: float, flow: np.ndarray | None, height: int = 4, width: int = 4) -> dict:
    K = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    depth = np.full((height, width), depth_z)
    valid = np.ones((height, width), dtype=bool)
    sr_valid = np.zeros((height, width), dtype=bool)
    if flow is not None:
        sr_valid[1, 1] = True
    return {
        "depth": depth, "valid": valid, "K": K, "metric": np.eye(4),
        "flow": flow, "sr_valid": sr_valid, "pair": np.array([0, 1]),
        "has_pair": flow is not None, "rgb": np.zeros((height, width, 3), dtype=np.uint8),
    }


def test_advance_tracks_uses_sr_flow_and_reanchors():
    flow = np.zeros((4, 4, 3))
    flow[1, 1] = (0.6, 0.0, 0.0)
    previous = _advection_frame(2.0, flow)
    frame = _advection_frame(2.0, np.zeros((4, 4, 3)))
    rows = np.array([1.0])
    cols = np.array([1.0])
    world = np.array([[2.0, 2.0, 2.0]])  # camera_point_at(1,1,2.0,K)
    alive = np.array([True])
    new_rows, new_cols, new_world, new_alive = srw.advance_tracks(
        rows, cols, world, alive, previous, frame, height=4, width=4
    )
    assert new_alive[0]
    # world + S = (2.6, 2, 2) → project(u=2.6→round 3, v=2) → 観測 depth で再アンカー
    assert np.allclose(new_world[0], [3.0, 2.0, 2.0])
    assert new_rows[0] == 2.0 and new_cols[0] == 3.0


def test_advance_tracks_dies_without_pair_or_invalid_projection():
    previous = _advection_frame(2.0, None)  # pair なし
    frame = _advection_frame(2.0, None)
    _, _, _, alive = srw.advance_tracks(
        np.array([1.0]), np.array([1.0]), np.array([[2.0, 2.0, 2.0]]), np.array([True]),
        previous, frame, height=4, width=4,
    )
    assert not alive[0]

    flow = np.zeros((4, 4, 3))
    flow[1, 1] = (100.0, 0.0, 0.0)  # 画面外へ射出 → bounds 拒否
    previous = _advection_frame(2.0, flow)
    frame = _advection_frame(2.0, flow)
    _, _, _, alive = srw.advance_tracks(
        np.array([1.0]), np.array([1.0]), np.array([[2.0, 2.0, 2.0]]), np.array([True]),
        previous, frame, height=4, width=4,
    )
    assert not alive[0]


def test_init_tracks_projects_seed_pixels():
    frame = _advection_frame(2.0, np.zeros((4, 4, 3)))
    frame["sr_valid"][:] = True
    rows, cols, world, alive = srw.init_tracks(frame)
    assert len(rows) == len(cols) == len(world) == len(alive)
    assert alive.all()
    for slot, (row, col) in enumerate(zip(rows, cols)):
        assert np.allclose(
            world[slot],
            srw.camera_point_at(float(col), float(row), float(frame["depth"][int(row), int(col)]),
                                frame["K"]),
        )


# ---------------------------------------------------------------------------
# 小型合成 dataset の実 export（ffprobe 枚数・RRD 構造・preview 表示・source 不変）


def _write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _build_fixture(root: Path, height: int = 64, width: int = 36,
                   metric: np.ndarray | None = None) -> Path:
    """frames_total=4、frame 2 以降が空ペアの小型 dataset を作る。"""
    dataset_dir = root / "grass"
    rgb_dir = dataset_dir / "frames_rgb"
    rgb_dir.mkdir(parents=True)
    total = 4
    K = np.array([[40.0, 0.0, width / 2], [0.0, 50.0, height / 2], [0.0, 0.0, 1.0]])
    metric_matrix = np.eye(4) if metric is None else np.asarray(metric, dtype=np.float64)
    rng = np.random.default_rng(20261016)
    frames = []
    for index in range(total):
        rgb = (rng.random((height, width, 3)) * 255).astype(np.uint8)
        rgb_path = rgb_dir / f"{index:06d}.png"
        Image.fromarray(rgb).save(rgb_path)
        frames.append({
            "frame_index": index,
            "name": f"frame_{index:06d}.png",
            "rgb_path": str(rgb_path),
            "rgb_sha256": hashlib.sha256(rgb_path.read_bytes()).hexdigest(),
            "height": height,
            "width": width,
        })
        z = 2.0 + 0.05 * index
        depth = np.full((height, width), z, dtype=np.float32)
        valid = np.ones((height, width), dtype=bool)
        _write_npz(dataset_dir / "stages" / "final" / f"{index:06d}.npz",
                   final_depth=depth, final_valid=valid)
        _write_npz(dataset_dir / "stages" / "geometry" / f"{index:06d}.npz",
                   arrow_K=K.astype(np.float32), arrow_K_valid=np.array(True),
                   arrow_pose_valid=np.array(True),
                   arrow_camera_to_metric=metric_matrix.astype(np.float32))
        has_pair = index < 2
        flow = np.full((height, width, 3), np.nan, dtype=np.float32)
        sr_valid = np.zeros((height, width), dtype=bool)
        if has_pair:
            flow[...] = (0.02, 0.0, 0.0)
            sr_valid[...] = True
        _write_npz(dataset_dir / "stages" / "sr" / f"{index:06d}.npz",
                   sr_flow=flow, sr_valid=sr_valid,
                   sr_support_count=np.int64(int(sr_valid.sum())),
                   sr_pair_index=np.array([index, index + 1] if has_pair else [-1, -1],
                                          dtype=np.int64))
        arrow_depth = np.full((height, width), np.nan, dtype=np.float32)
        arrow_valid = np.zeros((height, width), dtype=bool)
        arrow_valid[::3, ::3] = True
        arrow_depth[arrow_valid] = z
        _write_npz(dataset_dir / "stages" / "anchor" / f"{index:06d}.npz",
                   arrow_depth=arrow_depth, arrow_valid=arrow_valid)
        for stage, prefix in (("ndr", "ndr"), ("lingbot", "lingbot"), ("lingbot_fix", "lingbot"),
                              ("sceneflow_post", "sceneflow"), ("temporal", "temporal")):
            _write_npz(dataset_dir / "stages" / stage / f"{index:06d}.npz",
                       **{f"{prefix}_depth": depth.copy(), f"{prefix}_valid": valid.copy()})
    (dataset_dir / "frames.json").write_text(json.dumps({
        "dataset": "grass", "source": "synthetic", "frame_count": total,
        "original_size": [height, width], "frames": frames,
        "created_at": "2026-10-16T00:00:00+00:00",
    }, indent=1), encoding="utf-8")
    return dataset_dir


def _tree_hashes(root: Path) -> dict:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def _ffprobe_frame_count(path: Path) -> int:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    streams = json.loads(result.stdout)["streams"]
    assert streams, f"video stream がありません: {path}"
    return int(streams[0]["nb_read_frames"])


def test_summary_rows_use_current_frame_values_not_future_or_last():
    """frame-index 回帰: 「現在」行は stats['current']（そのフレームの確定値）を表示する。"""
    stats = {
        "frames_rendered": 3, "preview": True, "pixels": 100,
        "camera_valid": np.array([True, True, True]),
        "track_total": 48,
        "current": {"sr_rate": 0.9281, "final_rate": 1.0, "sr_count": 855374, "alive": 45},
    }
    rows = dict(srw.summary_rows(1, stats))
    assert rows["フレーム"] == "2/3 (preview)"
    assert rows["SR変位支持率（現在）"] == "92.81 %"
    assert rows["深度支持率（現在）"] == "100.00 %"
    assert rows["SR支持画素（現在）"] == "855,374"
    assert rows["SR移流軌跡（生存）"] == "45 / 48"
    # 別フレームの current を渡せばフレームごとに独立して反映される
    stats2 = dict(stats, current={"sr_rate": 0.0, "final_rate": 0.0, "sr_count": 0, "alive": 0})
    rows2 = dict(srw.summary_rows(2, stats2))
    assert rows2["SR変位支持率（現在）"] == "0.00 %"
    assert rows2["フレーム"] == "3/3 (preview)"


def test_preflight_fails_when_free_below_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(srw, "MIN_FREE_BYTES", 1 << 62)
    with pytest.raises(SystemExit):
        srw.preflight_disk(tmp_path / "out")


def test_cli_export_small_fixture_ffprobe_and_rrd(tmp_path, monkeypatch):
    monkeypatch.setattr(srw, "MIN_FREE_BYTES", 0)
    root = tmp_path / "depth_detail"
    dataset_dir = _build_fixture(root)
    source_hashes_before = _tree_hashes(dataset_dir)
    output = tmp_path / "sr_out"

    code = srw.main([
        "--input-dir", str(root), "--output-dir", str(output), "--dataset", "grass",
        "--max-frames", "3",
    ])
    assert code == 0
    visuals = output / "grass" / "sr_visuals"
    for name in ("grass_sr_multipanel.mp4", "grass_sr_multipanel.png", "grass_sr_workflow.rrd",
                 "grass_sr_workflow.rbl", "grass_sr_compare.mp4", "grass_sr_compare.png",
                 "grass_sr_manifest.json"):
        assert (visuals / name).is_file(), f"missing output: {name}"

    # 動画枚数（preview 3 フレーム、比較も同数）と静止画サイズ
    assert _ffprobe_frame_count(visuals / "grass_sr_multipanel.mp4") == 3
    assert _ffprobe_frame_count(visuals / "grass_sr_compare.mp4") == 3
    with Image.open(visuals / "grass_sr_multipanel.png") as image:
        assert image.size == (1920, 1080)

    # RRD / RBL の構造検証（magic + 非空）
    header = (visuals / "grass_sr_workflow.rrd").read_bytes()[:4]
    assert header == b"RRF2"
    assert (visuals / "grass_sr_workflow.rrd").stat().st_size > 0
    assert (visuals / "grass_sr_workflow.rbl").read_bytes()[:4] == b"RRF2"
    # .rbl は空テンプレートではなく 7 ビュー分の実 blueprint（空は ~8KiB、実体は数十 KiB）
    assert (visuals / "grass_sr_workflow.rbl").stat().st_size > 16384

    # manifest: preview 表記・frame IDs・空ペアの記録・source SHA
    manifest = json.loads((visuals / "grass_sr_manifest.json").read_text(encoding="utf-8"))
    assert manifest["preview"] is True
    assert manifest["frames_rendered"] == 3
    assert manifest["frames_total"] == 4
    assert manifest["frame_ids"] == [0, 1, 2]
    assert manifest["units"] == "arrow_provisional"
    assert manifest["metric_calibrated"] is False
    assert manifest["frames"][2]["has_pair"] is False  # 空ペアも記録される
    assert manifest["frames"][0]["rgb_sha256"] == source_hashes_before["frames_rgb/000000.png"]
    assert manifest["frames"][0]["sha256"]["final"]
    assert manifest["compare_arrow_display"]["grid"] == 3  # fixture の anchor は規則格子
    assert manifest["compare_arrow_display"]["area"] == 0

    # source 入力（NPZ/RGB/frames.json）は一切変更されていない
    assert _tree_hashes(dataset_dir) == source_hashes_before


def _read_rrd_entity(rrd_path: Path, entity: str):
    """実 RRD を RrdReader で読み戻す（read-only。frame timeline ごとに返す）。"""
    from rerun.chunk import RrdReader

    frames: list[int] = []
    payload: list = []
    for chunk in RrdReader(str(rrd_path)).stream():
        if chunk.entity_path != entity:
            continue
        batch = chunk.to_record_batch()
        frame_values = [int(round(float(value))) for value in batch.column("frame").to_pylist()]
        if entity.endswith("cloud"):
            for frame, points in zip(frame_values, batch.column("Points3D:positions").to_pylist()):
                frames.append(frame)
                payload.append(np.asarray(points, dtype=np.float64).reshape(-1, 3))
        else:
            origins = batch.column("Arrows3D:origins").to_pylist()
            vectors = batch.column("Arrows3D:vectors").to_pylist()
            for frame, origin, vector in zip(frame_values, origins, vectors):
                frames.append(frame)
                payload.append(
                    (np.asarray(origin, dtype=np.float64).reshape(-1, 3),
                     np.asarray(vector, dtype=np.float64).reshape(-1, 3))
                )
    return frames, payload


def _read_rrd_view_coordinates(rrd_path: Path) -> list[int] | None:
    """実 RRD の /world ViewCoordinates:xyz（ヘッダ宣言）を読み戻す。"""
    from rerun.chunk import RrdReader

    for chunk in RrdReader(str(rrd_path)).stream():
        if chunk.entity_path != "/world":
            continue
        batch = chunk.to_record_batch()
        if "ViewCoordinates:xyz" in batch.column_names:
            return (
                np.asarray(batch.column("ViewCoordinates:xyz").to_pylist())
                .reshape(-1).astype(int).tolist()
            )
    return None


def _right_hand_z_up_xyz() -> list[int]:
    """rerun 自身の定義から RIGHT_HAND_Z_UP の期待値を得る（core 監査と同じ手法）。"""
    import rerun as rr

    value = rr.ViewCoordinates.RIGHT_HAND_Z_UP
    coordinates = getattr(value, "coordinates", None)
    if coordinates is None:
        from rerun.components import ViewCoordinates as ViewCoordinatesComponent

        coordinates = getattr(ViewCoordinatesComponent.RIGHT_HAND_Z_UP, "coordinates", None)
    if coordinates is None:
        raise RuntimeError("RIGHT_HAND_Z_UP の coordinates を rerun 定義から取得できません")
    return [int(v.value) if hasattr(v, "value") else int(v) for v in coordinates]


def test_rrd_cloud_and_arrows_use_display_exactly_once(tmp_path, monkeypatch):
    """rotated+translated fixture の実 RRD を読み戻し、display(native) と数値一致を検証。"""
    monkeypatch.setattr(srw, "MIN_FREE_BYTES", 0)
    angle = 0.6
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                         [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    metric = np.eye(4)
    metric[:3, :3] = rotation
    metric[:3, 3] = [0.4, -0.3, 0.25]
    root = tmp_path / "dd"
    dataset_dir = _build_fixture(root, metric=metric)
    output = tmp_path / "out"
    assert srw.main(["--input-dir", str(root), "--output-dir", str(output),
                     "--dataset", "grass", "--max-frames", "2"]) == 0
    rrd = output / "grass" / "sr_visuals" / "grass_sr_workflow.rrd"
    # /world のヘッダ宣言（core 監査が要求する RIGHT_HAND_Z_UP。無いと null）
    assert _read_rrd_view_coordinates(rrd) == _right_hand_z_up_xyz()
    frames_cloud, clouds = _read_rrd_entity(rrd, "/world/cloud")
    frames_arrows, arrows = _read_rrd_entity(rrd, "/world/sr_arrows")
    assert 0 in frames_cloud and 0 in frames_arrows

    final = np.load(dataset_dir / "stages" / "final" / "000000.npz")
    geometry = np.load(dataset_dir / "stages" / "geometry" / "000000.npz")
    sr = np.load(dataset_dir / "stages" / "sr" / "000000.npz")
    frames_payload = json.loads((dataset_dir / "frames.json").read_text(encoding="utf-8"))
    rgb = np.asarray(
        Image.open(frames_payload["frames"][0]["rgb_path"]).convert("RGB"), dtype=np.uint8
    )
    depth = np.asarray(final["final_depth"], dtype=np.float64)
    valid = np.asarray(final["final_valid"], dtype=bool)
    K = np.asarray(geometry["arrow_K"], dtype=np.float64)
    metric_live = np.asarray(geometry["arrow_camera_to_metric"], dtype=np.float64)
    flow = np.asarray(sr["sr_flow"], dtype=np.float64)
    sr_valid = np.asarray(sr["sr_valid"], dtype=bool)

    cloud = srw.cloud_for_frame(depth, valid, K, metric_live, rgb)
    expected_points = srw.display(cloud["points"])
    got_points = clouds[frames_cloud.index(0)]
    assert got_points.shape == expected_points.shape
    assert np.allclose(got_points, expected_points, atol=1e-3)

    expected_o, expected_v = srw.sr_arrow_samples(flow, sr_valid, depth, valid, K, metric_live)
    got_o, got_v = arrows[frames_arrows.index(0)]
    assert got_o.shape == expected_o.shape and got_v.shape == expected_v.shape
    assert got_o.shape[0] > 0
    assert np.allclose(got_o, srw.display(expected_o), atol=1e-3)
    assert np.allclose(got_v, srw.display(expected_v), atol=1e-3)
    # 回転・並進 fixture では raw（display 未適用）とは一致しない = 二重変換や未変換を検出
    assert not np.allclose(got_o, expected_o, atol=1e-3)
    assert not np.allclose(got_v, expected_v, atol=1e-3)
