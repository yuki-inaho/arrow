"""scripts/infer_vace_inpaint.py (VACE masked completion adapter) の契約テスト。

実際のモデル重み・GPU は使わない。mask 規約（white=255=unknown/generate）は
インストール済み diffusers 0.34.0 pipeline_wan_vace.py の実ソースで確認済みの意味に合わせる。

実行（隔離環境）:
  temp/grass_static_review/env/venv/bin/python -m pytest tests/test_vace_inpaint.py -q
"""

from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = REPO_ROOT / "scripts" / "infer_vace_inpaint.py"


def _load_adapter():
    spec = importlib.util.spec_from_file_location("infer_vace_inpaint_under_test", ADAPTER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load adapter from {ADAPTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


adapter = _load_adapter()


def _controls_npz(
    tmp_path: Path,
    *,
    T: int = 9,
    H: int = 32,
    W: int = 32,
    draft: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    source: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    fps: float = 5.0,
    drop: str | None = None,
) -> Path:
    rng = np.random.default_rng(0)
    if source is None:
        source = rng.integers(0, 256, size=(T, H, W, 3), dtype=np.uint8)
    if mask is None:
        mask = np.zeros((T, H, W), dtype=bool)
        mask[:, H // 4 : H // 2, W // 4 : W // 2] = True  # 一部分のみ unknown
    mask_ok = np.asarray(mask).shape == (T, H, W)
    if draft is None:
        if source.shape[0] == T and mask_ok:
            draft = source.copy()
            draft[np.asarray(mask).astype(bool)] = 127  # unknown は flat gray 127
        else:
            draft = np.full((T, H, W, 3), 42, dtype=np.uint8)
            if mask_ok:
                draft[np.asarray(mask).astype(bool)] = 127
    if reference is None:
        reference = source[0].copy() if source.shape[0] == T else np.zeros((H, W, 3), dtype=np.uint8)
    payload = {
        "draft_rgb": draft,
        "edit_mask": mask,
        "source_rgb": source,
        "reference_rgb": reference,
        "fps": np.asarray(fps, dtype=np.float32),
    }
    if drop is not None:
        payload.pop(drop)
    path = tmp_path / "controls.npz"
    np.savez(path, **payload)
    return path


def _args(**overrides) -> Namespace:
    base = dict(
        model=None,
        controls=None,
        output_dir=None,
        steps=50,
        seed=42,
        guidance=5.0,
        conditioning_scale=1.0,
        flow_shift=None,
        quantization="none",
        model_id=adapter.DEFAULT_MODEL_ID,
        model_revision=adapter.DEFAULT_MODEL_REVISION,
        dry_run=False,
    )
    base.update(overrides)
    return Namespace(**base)


@pytest.mark.parametrize("option,value", [
    ("--steps", "0"), ("--steps", "-1"), ("--guidance", "nan"),
    ("--guidance", "inf"), ("--conditioning-scale", "-1"),
    ("--conditioning-scale", "nan"), ("--seed", str(2**64)), ("--seed", "-1"),
    ("--flow-shift", "nan"), ("--flow-shift", "0"),
])
def test_bad_options_rejected_before_loading_controls_or_model(monkeypatch, option, value):
    def forbidden(_args):
        pytest.fail("generation should not start with invalid options")
    monkeypatch.setattr(adapter, "run_generation", forbidden)
    assert adapter.main(["--controls", "/does-not-exist", option, value]) == 2


def test_zero_dimension_rejected(tmp_path):
    path = _controls_npz(tmp_path, H=0)
    with pytest.raises(adapter.ControlError, match="positive multiples"):
        adapter.load_controls(path)


def test_actual_unipc_shift_override_preserves_default_and_changes_schedule():
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from types import SimpleNamespace
    scheduler = UniPCMultistepScheduler(num_train_timesteps=1000, use_flow_sigmas=True,
                                       prediction_type="flow_prediction", flow_shift=3)
    pipe = SimpleNamespace(scheduler=scheduler)
    assert adapter.configure_scheduler(pipe, None)["flow_shift"] == 3
    assert pipe.scheduler is scheduler
    scheduler.set_timesteps(50)
    baseline = scheduler.sigmas.clone()
    receipt = adapter.configure_scheduler(pipe, 16)
    pipe.scheduler.set_timesteps(50)
    assert receipt["flow_shift"] == 16
    assert torch.isfinite(pipe.scheduler.sigmas).all()
    assert not torch.allclose(baseline, pipe.scheduler.sigmas)
    assert len(pipe.scheduler.sigmas) == len(baseline)


# ---------------------------------------------------------------------------
# mask 規約
# ---------------------------------------------------------------------------


def test_mask_convention_white_unknown_black_preserved():
    mask = np.zeros((2, 4, 4), dtype=bool)
    mask[0, 0, 0] = True
    mask[1, 3, 2] = True
    frames = adapter.mask_to_pil_frames(mask)
    assert len(frames) == 2
    for frame in frames:
        assert frame.mode == "L" and frame.size == (4, 4)
    first = np.asarray(frames[0])
    assert first[0, 0] == 255  # True(unknown) -> white = generate
    assert first[3, 3] == 0  # False -> black = preserve
    second = np.asarray(frames[1])
    assert second[3, 2] == 255 and second[0, 0] == 0


def test_draft_and_reference_pil_conversion():
    draft = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    draft[0, 1, 2] = [10, 20, 30]
    frames = adapter.draft_to_pil_frames(draft)
    assert len(frames) == 2 and frames[0].mode == "RGB"
    assert np.asarray(frames[0])[1, 2].tolist() == [10, 20, 30]
    reference = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    pil = adapter.reference_to_pil(reference)
    assert pil.size == (4, 4) and pil.mode == "RGB"
    assert np.array_equal(np.asarray(pil), reference)


def test_gray_mask_rejected(tmp_path):
    T, H, W = 9, 32, 32
    mask = np.full((T, H, W), 127, dtype=np.uint8)  # 灰色 = 曖昧値
    path = _controls_npz(tmp_path, mask=mask)
    with pytest.raises(adapter.ControlError, match="Gray/ambiguous"):
        adapter.load_controls(path)


def test_uint8_binary_mask_accepted_with_warning(tmp_path):
    T, H, W = 9, 32, 32
    mask = np.zeros((T, H, W), dtype=np.uint8)
    mask[0, :4, :4] = 255
    path = _controls_npz(tmp_path, mask=mask)
    controls = adapter.load_controls(path)
    assert controls.edit_mask.dtype == np.bool_
    assert any("uint8" in w for w in controls.warnings)


def test_draft_unknown_pixels_must_be_exact_gray_127(tmp_path):
    T, H, W = 9, 32, 32
    mask = np.zeros((T, H, W), dtype=bool)
    mask[0, 0, 0] = True
    draft = np.zeros((T, H, W, 3), dtype=np.uint8)
    draft[mask] = 126  # 127 でない
    path = _controls_npz(tmp_path, draft=draft, mask=mask)
    with pytest.raises(adapter.ControlError, match="exactly 127"):
        adapter.load_controls(path)

    draft[mask] = 127
    path_ok = _controls_npz(tmp_path, draft=draft, mask=mask)
    assert adapter.load_controls(path_ok).edit_mask[0, 0, 0]


def test_reference_must_equal_first_source_frame(tmp_path):
    H, W = 32, 32
    reference = np.zeros((H, W, 3), dtype=np.uint8)
    path = _controls_npz(tmp_path, reference=reference)
    with pytest.raises(adapter.ControlError, match="first actual source frame"):
        adapter.load_controls(path)


# ---------------------------------------------------------------------------
# 入力検証
# ---------------------------------------------------------------------------


def test_invalid_T_H_fps_and_shapes(tmp_path):
    with pytest.raises(adapter.ControlError, match=r"4k\+1"):
        adapter.load_controls(_controls_npz(tmp_path, T=50))
    with pytest.raises(adapter.ControlError, match="multiples of 16"):
        adapter.load_controls(_controls_npz(tmp_path, H=15))
    with pytest.raises(adapter.ControlError, match="fps"):
        adapter.load_controls(_controls_npz(tmp_path, fps=0.0))
    with pytest.raises(adapter.ControlError, match="missing keys"):
        adapter.load_controls(_controls_npz(tmp_path, drop="edit_mask"))
    # source shape 不一致
    T, H, W = 9, 32, 32
    bad_source = np.zeros((T, H, W, 3), dtype=np.uint8)[: T - 4]
    with pytest.raises(adapter.ControlError, match="source_rgb"):
        adapter.load_controls(_controls_npz(tmp_path, source=bad_source))
    # mask shape 不一致
    bad_mask = np.zeros((T, H, W - 4), dtype=bool)
    with pytest.raises(adapter.ControlError, match="edit_mask"):
        adapter.load_controls(_controls_npz(tmp_path, mask=bad_mask))


# ---------------------------------------------------------------------------
# composite（raw と区別。外部 mask 一致の厳密検査）
# ---------------------------------------------------------------------------


def test_composite_exact_outside_mask_with_known_fixture(tmp_path):
    T, H, W = 9, 32, 32
    controls = adapter.load_controls(_controls_npz(tmp_path))
    raw = np.full((T, H, W, 3), 200, dtype=np.uint8)  # draft と明確に異なる raw
    composite = adapter.composite_frames(raw, controls)
    mask = controls.edit_mask
    assert np.array_equal(composite[mask], raw[mask])  # 生成領域は raw
    assert np.array_equal(composite[~mask], controls.draft_rgb[~mask])  # 保持領域は draft
    report = adapter.check_outside_mask_equality(composite, controls.draft_rgb, mask)
    assert report["outside_mask_exact_equal_to_draft"] is True
    assert report["checked_preserved_pixels"] == int((~mask).sum())
    metadata = adapter.controls_metadata(controls)
    assert metadata["preserved_pixels_where_draft_equals_source"] == int((~mask).sum())


def test_composite_checker_detects_violation(tmp_path):
    T, H, W = 9, 32, 32
    controls = adapter.load_controls(_controls_npz(tmp_path))
    raw = np.full((T, H, W, 3), 200, dtype=np.uint8)
    composite = adapter.composite_frames(raw, controls)
    tampered = composite.copy()
    outside = ~controls.edit_mask
    first = np.argwhere(outside)[0]
    idx = tuple(int(x) for x in first)
    tampered[idx] = (tampered[idx].astype(int) + 1) % 256  # preserved 画素を改変
    with pytest.raises(adapter.AdapterError, match="outside-mask equality"):
        adapter.check_outside_mask_equality(tampered, controls.draft_rgb, controls.edit_mask)


def test_all_known_and_all_unknown_mask_handling(tmp_path):
    T, H, W = 9, 32, 32
    # all-False: 何も生成しない → composite == draft 全画素
    path_known = _controls_npz(tmp_path, mask=np.zeros((T, H, W), dtype=bool))
    controls_known = adapter.load_controls(path_known)
    assert any("all-False" in w for w in controls_known.warnings)
    raw = np.full((T, H, W, 3), 7, dtype=np.uint8)
    composite_known = adapter.composite_frames(raw, controls_known)
    assert np.array_equal(composite_known, controls_known.draft_rgb)
    adapter.check_outside_mask_equality(composite_known, controls_known.draft_rgb, controls_known.edit_mask)

    # all-True: 全画素生成 → composite == raw 全画素; preserved 0 なら検査は not_applicable
    path_unknown = _controls_npz(tmp_path, mask=np.ones((T, H, W), dtype=bool))
    controls_unknown = adapter.load_controls(path_unknown)
    assert any("all-True" in w for w in controls_unknown.warnings)
    composite_unknown = adapter.composite_frames(raw, controls_unknown)
    assert np.array_equal(composite_unknown, raw)
    report = adapter.check_outside_mask_equality(composite_unknown, controls_unknown.draft_rgb, controls_unknown.edit_mask)
    assert report["outside_mask_exact_equal_to_draft"] == "not_applicable"
    assert report["checked_preserved_pixels"] == 0


# ---------------------------------------------------------------------------
# pipeline 呼び出し（mock で引数・reference 次元を確認）
# ---------------------------------------------------------------------------


class _FakePipe:
    def __init__(self):
        self.kwargs: dict | None = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        # 実 postprocessor 契約: floating [0,1] np [1,F,H,W,C]
        return Namespace(frames=np.zeros((1, 9, 32, 32, 3), dtype=np.float32))


def test_mock_pipeline_args_and_reference_dimension(tmp_path):
    controls = adapter.load_controls(_controls_npz(tmp_path))
    fake = _FakePipe()
    result = adapter.call_vace_pipeline(fake, controls, _args(), "cpu")

    kwargs = fake.kwargs
    assert kwargs is not None, "pipeline arguments were not captured"
    assert len(kwargs["video"]) == controls.T  # video: T PIL frames
    assert all(frame.mode == "RGB" and frame.size == (controls.W, controls.H) for frame in kwargs["video"])
    assert len(kwargs["mask"]) == controls.T  # mask: T PIL frames
    assert all(frame.mode == "L" for frame in kwargs["mask"])
    mask_np = np.asarray(kwargs["mask"][0])
    assert set(np.unique(mask_np).tolist()) <= {0, 255}
    assert len(kwargs["reference_images"]) == 1  # reference 次元は 1
    assert kwargs["reference_images"][0].size == (controls.W, controls.H)
    assert np.array_equal(np.asarray(kwargs["reference_images"][0]), controls.source_rgb[0])
    assert kwargs["height"] == controls.H and kwargs["width"] == controls.W
    assert kwargs["num_frames"] == controls.T
    assert kwargs["num_inference_steps"] == 50
    assert kwargs["guidance_scale"] == 5.0
    assert kwargs["conditioning_scale"] == 1.0
    assert kwargs["output_type"] == "np"
    assert kwargs["negative_prompt"] is None  # still/static 系を入れない
    assert isinstance(kwargs["generator"], type(adapter.build_generator(0, "cpu")))
    assert hasattr(result, "frames") and result.frames.shape == (1, 9, 32, 32, 3)


def test_actual_postprocessor_known_values_and_extractor_quantization():
    """実 VideoProcessor.postprocess_video を known [-1,0,+1] tensor で実行する。

    receipt: evidence/engine/vace/postprocess_np_receipt.json と同一手順。
    """
    from diffusers.video_processor import VideoProcessor

    vp = VideoProcessor(vae_scale_factor=8)  # WanVACEPipeline L186 と同一構築
    video = torch.zeros(1, 3, 2, 4, 4)
    video[:, :, 0, 0, 0] = -1.0
    video[:, :, 0, 1, 1] = 0.0
    video[:, :, 0, 2, 2] = 1.0
    out = vp.postprocess_video(video, output_type="np")

    # 実測契約: floating [0,1], layout [B,F,H,W,C]
    assert isinstance(out, np.ndarray)
    assert out.shape == (1, 2, 4, 4, 3)
    assert np.issubdtype(out.dtype, np.floating)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out[0, 0, 0, 0, 0] == pytest.approx(0.0)
    assert out[0, 0, 1, 1, 0] == pytest.approx(0.5)
    assert out[0, 0, 2, 2, 0] == pytest.approx(1.0)

    quantized = adapter.extract_raw_frames(Namespace(frames=out), 2, 4, 4)
    assert quantized.dtype == np.uint8 and quantized.shape == (2, 4, 4, 3)
    assert quantized[0, 0, 0, 0] == 0
    assert quantized[0, 1, 1, 0] == 128  # round(0.5*255) half-to-even
    assert quantized[0, 2, 2, 0] == 255


def test_extract_raw_frames_rejects_nonfinite_out_of_range_and_wrong_shape():
    good = np.zeros((1, 9, 32, 32, 3), dtype=np.float32)
    assert adapter.extract_raw_frames(Namespace(frames=good), 9, 32, 32).shape == (9, 32, 32, 3)

    nan_frames = good.copy()
    nan_frames[0, 0, 0, 0, 0] = np.nan
    with pytest.raises(adapter.AdapterError, match="non-finite"):
        adapter.extract_raw_frames(Namespace(frames=nan_frames), 9, 32, 32)

    high = good.copy()
    high[0, 0, 0, 0, 0] = 1.01  # クリップで隠さない
    with pytest.raises(adapter.AdapterError, match=r"out of \[0,1\]"):
        adapter.extract_raw_frames(Namespace(frames=high), 9, 32, 32)

    low = good.copy()
    low[0, 0, 0, 0, 0] = -0.01
    with pytest.raises(adapter.AdapterError, match=r"out of \[0,1\]"):
        adapter.extract_raw_frames(Namespace(frames=low), 9, 32, 32)

    with pytest.raises(adapter.AdapterError, match="shape mismatch"):
        adapter.extract_raw_frames(Namespace(frames=np.zeros((1, 9, 30, 32, 3), dtype=np.float32)), 9, 32, 32)

    # uint8 を直接渡すのは実 postprocessor 契約違反（旧自己確認テストの除去）
    with pytest.raises(adapter.AdapterError, match="floating"):
        adapter.extract_raw_frames(Namespace(frames=np.zeros((1, 9, 32, 32, 3), dtype=np.uint8)), 9, 32, 32)


# ---------------------------------------------------------------------------
# MP4 + ffprobe 実frame数（実 ffprobe、CPU のみ）
# ---------------------------------------------------------------------------


def test_save_video_frames_writes_mp4_and_ffprobe_count(tmp_path):
    T, H, W = 5, 32, 32
    rng = np.random.default_rng(1)
    frames = rng.integers(0, 256, size=(T, H, W, 3), dtype=np.uint8)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    outputs = adapter.save_video_frames(frames, out_dir, 5.0, "raw")
    assert outputs["n_frames"] == T
    assert outputs["mp4"]["ffprobe_frame_count"] == T
    assert outputs["mp4"]["bytes"] > 0
    assert len(outputs["frames"]) == T and all(r["sha256"] for r in outputs["frames"])
    assert (out_dir / "raw_frames" / "frame_0000.png").is_file()


def test_save_masks_records_hashes_and_controls_path(tmp_path):
    controls = adapter.load_controls(_controls_npz(tmp_path))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    outputs = adapter.save_masks(controls, out_dir)
    assert outputs["controls_path"] == str(controls.path)
    assert outputs["controls_sha256"] == controls.sha256
    assert (out_dir / "masks" / "mask_0000.png").is_file()
    rendered = np.asarray(Image.open(out_dir / "masks" / "mask_0000.png"))
    assert set(np.unique(rendered).tolist()) <= {0, 255}


# ---------------------------------------------------------------------------
# quantization=nf4 と provenance（CPU mocks のみ。GPU round-trip は root 所有）
# ---------------------------------------------------------------------------


def test_cli_quantization_and_provenance_flags_defaults_and_overrides():
    parser = adapter.build_arg_parser()
    defaults = parser.parse_args(["--controls", "/x.npz"])
    assert defaults.quantization == "none"
    assert defaults.model_id == "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
    assert defaults.model_revision == adapter.DEFAULT_MODEL_REVISION

    explicit = parser.parse_args([
        "--controls", "/x.npz", "--quantization", "nf4",
        "--model-id", "Wan-AI/Wan2.1-VACE-14B-diffusers", "--model-revision", "deadbeef",
    ])
    assert explicit.quantization == "nf4"
    assert explicit.model_id == "Wan-AI/Wan2.1-VACE-14B-diffusers"
    assert explicit.model_revision == "deadbeef"
    with pytest.raises(SystemExit):
        parser.parse_args(["--controls", "/x.npz", "--quantization", "int8"])


def test_quantization_report_records_mode_compute_dtype_and_bnb_version(monkeypatch):
    monkeypatch.setattr(adapter, "_bitsandbytes_version", lambda: "0.46.1")
    report = adapter.quantization_report(_args(quantization="nf4"))
    assert report["mode"] == "nf4"
    assert report["requested_nf4_compute_dtype"] == "bfloat16"
    assert report["bitsandbytes_version"] == "0.46.1"
    assert report["bitsandbytes_min_version"] == "0.46.1"

    report_none = adapter.quantization_report(_args())
    assert report_none["mode"] == "none"
    assert report_none["requested_nf4_compute_dtype"] is None


def test_nf4_rejects_absent_or_old_bitsandbytes_before_any_load(monkeypatch, tmp_path):
    def forbidden(*_args, **_kwargs):
        pytest.fail("quantized transformer must not be loaded when bitsandbytes is unusable")

    monkeypatch.setattr(adapter, "load_quantized_transformer", forbidden)

    monkeypatch.setattr(adapter, "_bitsandbytes_version", lambda: None)
    with pytest.raises(adapter.AdapterError, match="not installed"):
        adapter.load_pipeline_nf4(tmp_path, "cuda")

    monkeypatch.setattr(adapter, "_bitsandbytes_version", lambda: "0.45.0")
    with pytest.raises(adapter.AdapterError, match=">=0.46.1"):
        adapter.load_pipeline_nf4(tmp_path, "cuda")


def test_nf4_load_order_transformer_reuse_cpu_move_and_no_pipe_to(monkeypatch, tmp_path):
    from diffusers.quantizers.quantization_config import BitsAndBytesConfig

    events: list = []

    class FakeVAEInstance:
        def enable_tiling(self):
            events.append("enable_tiling")

    vae_instance = FakeVAEInstance()
    state: dict = {}

    class FakeQuantizedTransformer:
        def to(self, *args, **kwargs):
            assert args == ("cpu",) and kwargs == {}  # dtype 引数なしの CPU 移動
            events.append("to_cpu")
            return self

    class FakeTransformer:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert kwargs["torch_dtype"] == torch.bfloat16
            assert kwargs["local_files_only"] is True
            assert kwargs["low_cpu_mem_usage"] is True
            assert kwargs["device_map"] == {"": 0}
            quant_config = kwargs["quantization_config"]
            assert isinstance(quant_config, BitsAndBytesConfig)
            assert quant_config.load_in_4bit is True
            assert quant_config.bnb_4bit_quant_type == "nf4"
            assert quant_config.bnb_4bit_compute_dtype == torch.bfloat16
            assert path.endswith("transformer")
            events.append("transformer_load")
            instance = FakeQuantizedTransformer()
            state["transformer"] = instance
            return instance

    class FakeVAE:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert kwargs == {"subfolder": "vae", "torch_dtype": torch.float32}
            events.append("vae_load")
            return vae_instance

    class FakePipe:
        def __init__(self):
            self.vae = vae_instance

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert kwargs["transformer"] is state["transformer"]  # 実オブジェクト再利用（同一性）
            assert kwargs["vae"] is vae_instance
            assert kwargs["torch_dtype"] == torch.bfloat16
            assert "quantization_config" not in kwargs
            events.append("pipeline_construct")
            return cls()

        def enable_model_cpu_offload(self, device):
            events.append(f"offload:{device}")

        def to(self, *args, **kwargs):  # pragma: no cover - 呼ばれたら fail
            events.append("pipe_to_called")
            raise AssertionError("pipe.to must never be called (no blanket .to('cuda'))")

    monkeypatch.setattr(adapter, "_bitsandbytes_version", lambda: "0.46.1")
    monkeypatch.setattr(adapter, "_get_diffusers_classes", lambda: (FakeTransformer, FakeVAE, FakePipe))

    pipe = adapter.load_pipeline_nf4(tmp_path, "cuda")
    assert pipe.vae is vae_instance
    assert events == [
        "transformer_load",
        "to_cpu",
        "vae_load",
        "pipeline_construct",
        "offload:cuda",
        "enable_tiling",
    ]
    assert "pipe_to_called" not in events


def test_none_path_unchanged_order_and_components(monkeypatch, tmp_path):
    original_import_module = importlib.import_module
    events: list = []

    class FakeVAEInstance:
        def enable_tiling(self):
            events.append("enable_tiling")

    vae_instance = FakeVAEInstance()

    class FakeVAE:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert kwargs == {"subfolder": "vae", "torch_dtype": torch.float32}
            events.append("vae_load")
            return vae_instance

    class FakePipe:
        def __init__(self):
            self.vae = None

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert kwargs == {"torch_dtype": torch.bfloat16}  # 既存どおり全体 load
            events.append("pipeline_load")
            return cls()

        def enable_model_cpu_offload(self, device):
            events.append(f"offload:{device}")

    def fake_import_module(name):
        if name == "diffusers":
            return SimpleNamespace(AutoencoderKLWan=FakeVAE, WanVACEPipeline=FakePipe)
        return original_import_module(name)

    monkeypatch.setattr(adapter.importlib, "import_module", fake_import_module)

    pipe = adapter.load_pipeline(tmp_path, "cuda")
    assert pipe.vae is vae_instance  # none path でも VAE は fp32 で差し替え
    assert events == ["pipeline_load", "vae_load", "offload:cuda", "enable_tiling"]


@pytest.mark.parametrize("cache_subdir", [None, ".cache", ".cache/huggingface/trees"])
def test_model_report_requested_14b_provenance_and_actual_revision_unknown(tmp_path, cache_subdir):
    model_dir = tmp_path / "model"
    (model_dir / "transformer").mkdir(parents=True)
    (model_dir / "model_index.json").write_text('{"_class_name": "WanVACEPipeline"}')
    (model_dir / "transformer" / "config.json").write_text(
        '{"in_channels": 16, "num_layers": 40, "patch_size": [1, 2, 2]}'
    )
    provenance = {"source": "converted", "bf16_derived": "sha256:abc"}
    (model_dir / "DERIVED_BF16_PROVENANCE.json").write_text(__import__("json").dumps(provenance))
    if cache_subdir is not None:
        (model_dir / cache_subdir).mkdir(parents=True)

    requested_model_id = "Wan-AI/Wan2.1-VACE-14B-diffusers"
    requested_revision = "db79b90c60bbb45ceec9e41b9d5a4df934538ac4"
    report = adapter.model_report(
        model_dir, _args(model_id=requested_model_id, model_revision=requested_revision)
    )
    assert report["requested_model_id"] == requested_model_id
    assert report["requested_revision"] == requested_revision
    assert "expected_revision" not in report  # legacy 1.3B identity must not leak into 14B receipts
    assert report["actual_revision"] is None  # 検証receiptが無い限り actual とは主張しない
    assert report["local_snapshot_receipt"] is None  # empty cache directories prove no weight identity
    assert "model_provenance" in report
    assert report["model_provenance"]["sha256"] == adapter.sha256_file(model_dir / "DERIVED_BF16_PROVENANCE.json")
    assert "derived local checkpoint" in report["model_provenance"]["note"]
    assert report["model_index_sha256"] == adapter.sha256_file(model_dir / "model_index.json")
    assert report["transformer_config_summary"]["num_layers"] == 40


def test_dry_run_nf4_validates_dependency_without_loading(monkeypatch, tmp_path):
    controls_path = _controls_npz(tmp_path)
    monkeypatch.setattr(adapter, "_bitsandbytes_version", lambda: None)
    with pytest.raises(adapter.AdapterError, match="not installed"):
        adapter.run_dry_run(_args(controls=str(controls_path), quantization="nf4"))


def test_dry_run_none_reports_quantization_metadata(monkeypatch, tmp_path, capsys):
    import json as _json

    controls_path = _controls_npz(tmp_path)
    monkeypatch.setattr(adapter, "_bitsandbytes_version", lambda: "0.46.1")
    assert adapter.run_dry_run(_args(controls=str(controls_path), quantization="none")) == 0
    report = _json.loads(capsys.readouterr().out)
    assert report["quantization"]["mode"] == "none"
    assert report["quantization"]["requested_nf4_compute_dtype"] is None
    assert report["quantization"]["bitsandbytes_version"] == "0.46.1"


def test_actual_quantization_config_is_json_serializable_before_generation():
    import json
    from types import SimpleNamespace
    from diffusers.quantizers.quantization_config import BitsAndBytesConfig

    config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16)
    transformer = SimpleNamespace(config=SimpleNamespace(quantization_config=config),
                                  dtype=torch.bfloat16, modules=lambda: iter(()))
    report = adapter.actual_transformer_report(transformer)
    restored = json.loads(json.dumps(report, allow_nan=False))
    assert restored["dtype"] == "torch.bfloat16"
    assert restored["quantization_config"]["bnb_4bit_quant_type"] == "nf4"
    assert restored["quantization_config"]["bnb_4bit_compute_dtype"] == "bfloat16"
    transformer.config = SimpleNamespace()
    assert adapter.actual_transformer_report(transformer)["quantization_config"] is None
