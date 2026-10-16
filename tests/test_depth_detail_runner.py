"""runner の統計/署名/argv 経路の CPU 統合テスト（TDD: RED -> GREEN）。

背景: 実 GPU 実行で prep 完了後の ndr 署名計算が
`stage_depth_percentiles(context, "lingbot_fix", "lingbot_depth", "lingbot_valid")` の
KeyError 'lingbot_valid' で停止した。valid mask は stage `lingbot` 側にあり depth は
`lingbot_fix` 側に保存されるレイアウトのため、runner 側で valid_stage を明示して解決する。

GPU/モデル推論は一切起動しない。stage NPZ は実 schema の save_stage_frame で生成する。
既存の results/ 実データは読み取り専用 preflight のみに使う。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

import depth_detail_models as models  # type: ignore[import-not-found]  # noqa: E402
import run_depth_detail as rdd  # type: ignore[import-not-found]  # noqa: E402

HEIGHT, WIDTH = 48, 64
FRAME_COUNT = 4
REAL_GRASS = REPO / "results" / "depth_detail_oct16" / "grass"


def _rgb() -> np.ndarray:
    rng = np.random.default_rng(7)
    return (rng.random((HEIGHT, WIDTH, 3)) * 255).astype(np.uint8)


def make_context(tmp_path: Path, dataset: str = "grass", frame_count: int = FRAME_COUNT) -> rdd.Context:
    source = tmp_path / "src"
    source.mkdir()
    rgb = _rgb()
    models.write_png(rgb, source / "frame_000000.png")
    root = tmp_path / dataset
    root.mkdir()
    records = []
    for index in range(frame_count):
        png = root / "frames_rgb" / f"{index:06d}.png"
        models.write_png(rgb, png)
        records.append(
            {
                "frame_index": index,
                "source_index": index,
                "name": f"frame_{index:06d}.png",
                "relative_path": f"frames/{index:06d}.npz",
                "rgb_path": str(png),
                "rgb_sha256": "0" * 64,
                "rgb_sha256_source": "fixture",
                "height": HEIGHT,
                "width": WIDTH,
                "timestamp": None,
                "source": "fixture",
            }
        )
    models.atomic_write_json(
        root / "frames.json",
        {"dataset": dataset, "source": "fixture", "frame_count": frame_count,
         "original_size": [HEIGHT, WIDTH], "frames": records},
    )
    args = rdd.build_parser().parse_args(["--dataset", dataset])
    args.output_dir = tmp_path
    return rdd.Context(
        args=args, dataset=dataset, source=source, root=root,
        frames_json=root / "frames.json", expected_frames=frame_count, video_fps=5.0,
    )


def write_depth_stage(root: Path, stage: str, key: str, depth: np.ndarray) -> None:
    for index in range(FRAME_COUNT):
        models.save_stage_frame(
            root, stage, index,
            {key: depth.astype(np.float32), f"{stage}_support_count": np.int64(int(np.isfinite(depth).sum()))},
            {"stage": stage, "fixture": True},
        )


def stage_ndr_and_upstream(context: rdd.Context) -> None:
    """ndr 署名に必要な staged 保存 schema（sparse anchor / dense lingbot / lingbot_fix / ndr）を作る。"""
    depth = 3.0 + 0.1 * np.arange(HEIGHT, dtype=np.float64)[:, None] * np.ones((1, WIDTH))
    valid = np.ones((HEIGHT, WIDTH), dtype=bool)
    valid[:, :4] = False  # support の一部を無効化（NaN/非正値除外の検証も兼ねる）
    for index in range(FRAME_COUNT):
        anchor_depth = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
        anchor_valid = np.zeros((HEIGHT, WIDTH), dtype=bool)
        anchor_depth[::4, ::4] = 2.9
        anchor_valid[::4, ::4] = True
        models.save_stage_frame(
            context.root, "anchor", index,
            {"arrow_depth": anchor_depth, "arrow_valid": anchor_valid},
            {"stage": "anchor", "fixture": True},
        )
        lingbot_depth = depth.copy()
        lingbot_depth[0, 0] = np.nan
        lingbot_depth[0, 1] = 0.0
        lingbot_valid = valid.copy()
        models.save_stage_frame(
            context.root, "lingbot", index,
            {"lingbot_depth_raw": lingbot_depth.astype(np.float32),
             "lingbot_valid": lingbot_valid},
            {"stage": "lingbot", "fixture": True},
        )
        models.save_stage_frame(
            context.root, "lingbot_fix", index,
            {"lingbot_depth": lingbot_depth.astype(np.float32)},
            {"stage": "lingbot_fix", "fixture": True},
        )
        ndr_depth = lingbot_depth * 1.01
        models.save_stage_frame(
            context.root, "ndr", index,
            {"ndr_depth": ndr_depth.astype(np.float32),
             "ndr_valid": lingbot_valid},
            {"stage": "ndr", "fixture": True},
        )


def write_signature_inputs(context: rdd.Context) -> None:
    models.atomic_write_json(
        context.clips_dir / "manifest.json",
        {"signature": {"clip_overlap": 4, "device": None},
         "clips": [{"name": "clip_0000-0004_s1.npz", "start": 0, "end": FRAME_COUNT,
                    "stride": 1, "frames": FRAME_COUNT, "status": "done"}]},
    )
    models.atomic_write_json(
        context.alignments,
        {"clips": [{"name": "clip_0000-0004_s1.npz", "scale": 1.0, "rotation": np.eye(3).tolist(),
                    "translation": [0.0, 0.0, 0.0]}]},
    )


# --- RED だった実障害 path: ndr_depth_bounds / group_signature(ndr) -----------------


def test_ndr_depth_bounds_reads_valid_from_lingbot_stage(tmp_path):
    context = make_context(tmp_path)
    stage_ndr_and_upstream(context)
    min_depth, max_depth, source = rdd.ndr_depth_bounds(context)
    assert min_depth > 0 and max_depth > min_depth
    assert "full-scan" in source


def test_group_signature_ndr_end_to_end(tmp_path):
    context = make_context(tmp_path)
    stage_ndr_and_upstream(context)
    payload = rdd.group_signature(context, "ndr")
    assert len(payload["signature_hash"]) == 64
    assert payload["params"]["max_depth"] > payload["params"]["min_depth"]
    assert payload["params"]["backend"] == "rust"


def test_ndr_depth_bounds_excludes_invalid_and_nonpositive(tmp_path):
    context = make_context(tmp_path)
    stage_ndr_and_upstream(context)
    stats = rdd.stage_depth_percentiles(
        context, "lingbot_fix", "lingbot_depth", "lingbot_valid", valid_stage="lingbot"
    )
    full = HEIGHT * WIDTH
    # valid[:, :4]=False と (0,0)/(0,1) の invalid は sample から除外される
    assert 0 < stats["count"] < full
    assert stats["median"] > 3.0


# --- 明瞭 fail（欠stage/欠key/shape不一致/空支持） ---------------------------------


def test_missing_valid_stage_fails_clearly(tmp_path):
    context = make_context(tmp_path)
    stage_ndr_and_upstream(context)
    stage_dir = context.root / "stages" / "lingbot"
    for path in stage_dir.glob("*.npz"):
        path.unlink()
    with pytest.raises(SystemExit, match="lingbot"):
        rdd.ndr_depth_bounds(context)


def test_missing_key_fails_clearly(tmp_path):
    context = make_context(tmp_path)
    stage_ndr_and_upstream(context)
    for index in range(FRAME_COUNT):
        models.save_stage_frame(
            context.root, "lingbot", index, {"lingbot_depth_raw": np.ones((HEIGHT, WIDTH))},
            {"stage": "lingbot", "fixture": True},
        )
    with pytest.raises(SystemExit, match="lingbot_valid"):
        rdd.ndr_depth_bounds(context)


def test_shape_mismatch_fails_clearly(tmp_path):
    context = make_context(tmp_path)
    stage_ndr_and_upstream(context)
    for index in range(FRAME_COUNT):
        models.save_stage_frame(
            context.root, "lingbot", index,
            {"lingbot_depth_raw": np.ones((HEIGHT, WIDTH)),
             "lingbot_valid": np.ones((HEIGHT - 1, WIDTH), dtype=bool)},
            {"stage": "lingbot", "fixture": True},
        )
    with pytest.raises(SystemExit, match="shape"):
        rdd.ndr_depth_bounds(context)


def test_empty_support_fails_before_computing(tmp_path):
    context = make_context(tmp_path)
    stage_ndr_and_upstream(context)
    for index in range(FRAME_COUNT):
        models.save_stage_frame(
            context.root, "lingbot", index,
            {"lingbot_depth_raw": np.ones((HEIGHT, WIDTH)),
             "lingbot_valid": np.zeros((HEIGHT, WIDTH), dtype=bool)},
            {"stage": "lingbot", "fixture": True},
        )
    with pytest.raises(SystemExit, match="有効値が空"):
        rdd.ndr_depth_bounds(context)


def test_stage_depth_percentiles_off_sampling_grid_fails(tmp_path):
    context = make_context(tmp_path)
    stage_ndr_and_upstream(context)
    valid = np.zeros((HEIGHT, WIDTH), dtype=bool)
    valid[1, 1] = True  # [::4,::4] サンプリングに乗らない支持のみ（sample関数の検査）
    for index in range(FRAME_COUNT):
        models.save_stage_frame(
            context.root, "lingbot", index,
            {"lingbot_depth_raw": np.ones((HEIGHT, WIDTH)),
             "lingbot_valid": valid},
            {"stage": "lingbot", "fixture": True},
        )
    with pytest.raises(SystemExit, match="統計"):
        rdd.stage_depth_percentiles(
            context, "lingbot_fix", "lingbot_depth", "lingbot_valid", valid_stage="lingbot"
        )


# --- sceneflow 設定の保存 schema 整合 ----------------------------------------------


def write_sceneflow_pairs(context: rdd.Context, ranges: dict[int, tuple[float, float]]) -> None:
    """pairs.json + pair入力npz（実prep保存schemaの最小形）を作る。ranges={pair_start: (min,max)}"""
    inputs_dir = context.root / "stages" / "sceneflow" / "pairs" / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    for start, (low, high) in sorted(ranges.items()):
        pair = [start, start + 1]
        depth = np.linspace(low, high, HEIGHT * WIDTH, dtype=np.float64).reshape(HEIGHT, WIDTH)
        name = f"pair_{pair[0]:06d}_{pair[1]:06d}.npz"
        np.savez_compressed(
            inputs_dir / name,
            rgb0_low=np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8),
            rgb1_low=np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8),
            rgb1_in_0=np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8),
            depth0_low=depth.astype(np.float32),
            depth1_low=depth.astype(np.float32),
            depth1_in_0=depth.astype(np.float32),
            valid0_low=np.ones((HEIGHT, WIDTH), dtype=bool),
            valid1_low=np.ones((HEIGHT, WIDTH), dtype=bool),
            valid1_in_0=np.ones((HEIGHT, WIDTH), dtype=bool),
            K_low=np.eye(3, dtype=np.float64),
            K_low_source=np.eye(3, dtype=np.float64),
            pair_index=np.asarray(pair, dtype=np.int64),
            clip_index=np.int32(0),
            low_size=np.asarray([HEIGHT, WIDTH], dtype=np.int32),
        )
        pairs.append({"pair_index": pair, "clip_index": 0, "file": str(inputs_dir / name)})
    models.atomic_write_json(
        context.root / "stages" / "sceneflow" / "pairs.json",
        {"dataset": context.dataset, "long_side": 32, "low_size": [HEIGHT, WIDTH],
         "pairs": pairs, "counts": {"written": len(pairs)}},
    )


def test_sceneflow_config_values_scans_all_prepared_inputs(tmp_path):
    context = make_context(tmp_path)
    write_sceneflow_pairs(context, {0: (1.0, 3.0)})
    min_depth, max_depth, max_motion, scan = rdd.sceneflow_config_values(context)
    assert scan["pair_count"] == 1 and scan["value_count"] > 0
    assert abs(scan["global_min"] - 1.0) < 1e-6 and abs(scan["global_max"] - 3.0) < 1e-6
    assert abs(min_depth - 0.5) < 1e-9
    assert abs(max_depth - 6.0) < 1e-9
    assert 0 < max_motion < min_depth  # C++ solver 制約


def test_sceneflow_bounds_cover_late_unsampled_outlier(tmp_path):
    """sample分位では取りこぼす下側外れ値(pair 83相当)も全域scanでboundsに含むこと。"""
    context = make_context(tmp_path)
    write_sceneflow_pairs(context, {0: (2.0, 4.0), 83: (0.55, 4.0)})
    min_depth, max_depth, _, scan = rdd.sceneflow_config_values(context)
    assert scan["global_min"] <= 0.55 + 1e-6  # float32保存の丸め許容
    assert min_depth <= 0.55 + 1e-6
    assert min_depth <= scan["global_min"] * 0.5 + 1e-9
    assert max_depth >= 4.0 * 2.0 - 1e-6


def test_sceneflow_bounds_missing_prepared_fails_clearly(tmp_path):
    context = make_context(tmp_path)
    with pytest.raises(SystemExit, match="prepared"):
        rdd.sceneflow_config_values(context)


def test_sceneflow_bounds_tiny_global_min_needs_design(tmp_path):
    context = make_context(tmp_path)
    write_sceneflow_pairs(context, {0: (5e-7, 2.0)})
    with pytest.raises(SystemExit, match="要設計"):
        rdd.sceneflow_config_values(context)


def test_sceneflow_signature_policy_without_prepared_inputs(tmp_path):
    """prep前に呼ばれる group_signature はデータを読まず policy を hash する。"""
    context = make_context(tmp_path)
    payload = rdd.group_signature(context, "sceneflow")
    assert len(payload["signature_hash"]) == 64
    assert "scan_all_prepared_inputs" in payload["params"]["bounds_policy"]
    argv = payload["argv"]
    assert "<runtime scan_all_prepared_inputs: global_min*0.5>" in argv
    assert "--workers" in argv


def test_ndr_bounds_full_scan_catches_unsampled_outlier(tmp_path):
    """26frame中の奇数frame(sample=12だと frame[::2] で除外される位置)の外れ値maxがboundsに入ること。"""
    context = make_context(tmp_path, frame_count=26)
    depth = np.full((HEIGHT, WIDTH), 3.0)
    valid = np.ones((HEIGHT, WIDTH), dtype=bool)
    for index in range(26):
        frame_depth = depth.copy()
        if index == 25:  # 旧sample (step=2 → 偶数frame) では見えない位置
            frame_depth[10, 10] = 50.0
        models.save_stage_frame(
            context.root, "lingbot", index,
            {"lingbot_depth_raw": frame_depth.astype(np.float32), "lingbot_valid": valid},
            {"stage": "lingbot", "fixture": True},
        )
        models.save_stage_frame(
            context.root, "lingbot_fix", index,
            {"lingbot_depth": frame_depth.astype(np.float32)},
            {"stage": "lingbot_fix", "fixture": True},
        )
    min_depth, max_depth, source = rdd.ndr_depth_bounds(context)
    assert "full-scan" in source
    assert max_depth >= 50.0 * 2.0 - 1e-9


# --- 全 STAGE_ORDER: group_signature と plan_argv（dry-run 代替ではない） ------------


def test_group_signature_and_plan_argv_all_stages(tmp_path):
    context = make_context(tmp_path)
    stage_ndr_and_upstream(context)
    write_signature_inputs(context)
    for group in rdd.STAGE_ORDER:
        payload = rdd.group_signature(context, group)
        assert len(payload["signature_hash"]) == 64, group
        assert payload["argv"] == [str(token) for token in rdd.plan_argv(context, group)], group
        assert payload["argv"], group


# --- 実 grass 完了データの read-only preflight -------------------------------------


def test_readonly_preflight_real_grass(tmp_path):
    if not (REAL_GRASS / "frames.json").is_file():
        pytest.skip("実 grass データなし")
    args = rdd.build_parser().parse_args(["--dataset", "grass", "--dry-run"])
    contexts = rdd.dataset_contexts(args, require_frames=True)
    context = contexts[0]

    def snapshot(path: Path):
        if not path.exists():
            return None  # missing は missing のままを検査（時刻依存にしない）
        return {"bytes": path.stat().st_size, "sha256": models.sha256_bytes(path.read_bytes())}

    watched = [context.signature_path("ndr"), context.root / "manifest.json"]
    before = {str(path): snapshot(path) for path in watched}
    # 読み取り専用 preflight: signatures/manifest/既存出力への書込なし（dry_run=True で cache 書込も抑止）
    min_depth, max_depth, _ = rdd.ndr_depth_bounds(context)
    assert min_depth > 0 and max_depth > min_depth
    payload = rdd.group_signature(context, "ndr")
    assert len(payload["signature_hash"]) == 64
    after = {str(path): snapshot(path) for path in watched}
    assert before == after, "read-only preflight が signatures/manifest を変更した"
