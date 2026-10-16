#!/usr/bin/env python3
"""VACE masked completion adapter with separately saved raw and composited RGB.

diffusers 0.34.0 の公式 WanVACEPipeline に接続する。
入力、研究比較、再現手順は docs/GUIDED_STATIC_WORLD.md を参照。

mask 規約（インストール済みソースで確認済み）:
  pipeline_wan_vace.py の docstring L701-702: "Black areas in the mask indicate
  conditioning regions, while white areas indicate regions for generation."
  → controls の edit_mask True(unknown) = 生成対象 = mask 画像の白(255)。
    False(preserved) = 黒(0)。preprocess で (x+1)/2 により 255→1.0, 0→0.0 になる
    ことも確認済み (preprocess_conditions L423-425)。

controls NPZ (pickle 禁止):
  draft_rgb     uint8[T,H,W,3]  unknown 画素は 127 (flat gray) でなければ fail
  edit_mask     bool[T,H,W] (または {0,255} uint8) True=unknown=生成
  source_rgb    uint8[T,H,W,3]  実測 source frames
  reference_rgb uint8[H,W,3]    = source_rgb[0] であることを厳密検査
  fps           positive scalar
  T=4k+1、H/W は 16 の倍数（Wan VAE temporal=4 / spatial*patch=16 と整合）

出力（raw と composite を分離保存。composite を raw 生成と呼称しない）:
  raw_frames/ raw.mp4       公式 pipeline の np[0,1] 出力を8bit RGBへ丸めたもの
  composite_frames/ composite.mp4  where(edit_mask, raw, draft_rgb)
  masks/                    白=unknown/生成 の PNG
  manifest.json             契約ハッシュ、外部mask一致の厳密検査、ffprobe 実frame数、
                            timings/peakmemory scope/model revision/package versions
  失敗時は manifest_failed.json + 非ゼロ終了（fallback/CPU/copy-source しない）

--dry-run は CPU で入力を検査する。本生成は CUDA を必要とする。
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

EXPECTED_MODEL_REVISION = "ec4d2cb062b548996b179d493fdd05340de702a1"
MASK_CONVENTION = "white(255)=generate/unknown; black(0)=preserve (diffusers 0.34.0 docstring L701-702)"
DRAFT_UNKNOWN_VALUE = 127
PROMPT = (
    "A close-up photographic view of real green weeds growing out of cracks in gray asphalt "
    "pavement, a white painted parking line crossing the ground, a pale concrete curb at the "
    "edge, natural daylight, realistic textures, same camera viewpoint."
)
NEGATIVE_PROMPT = None  # still/static 系の語は不使用（freeze と競合しないため空扱い）
CONTROL_KEYS = {"draft_rgb", "edit_mask", "source_rgb", "reference_rgb", "fps"}
DEFAULT_MODEL_ID = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
DEFAULT_MODEL_REVISION = EXPECTED_MODEL_REVISION  # 既存 1.3B 既定を維持
BITSANDBYTES_MIN_VERSION = "0.46.1"  # Params4bit.to が bnb_quantized を CPU/GPU 移動で保持する版
NF4_COMPUTE_DTYPE = "bfloat16"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AdapterError(RuntimeError):
    """contract 違反（fail-fast）。"""


class ControlError(AdapterError):
    """controls NPZ の契約違反。"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------


@dataclass
class Controls:
    path: Path
    sha256: str
    draft_rgb: np.ndarray
    edit_mask: np.ndarray  # bool, True=unknown=generate
    source_rgb: np.ndarray
    reference_rgb: np.ndarray
    fps: float
    warnings: list[str] = field(default_factory=list)

    @property
    def T(self) -> int:
        return int(self.draft_rgb.shape[0])

    @property
    def H(self) -> int:
        return int(self.draft_rgb.shape[1])

    @property
    def W(self) -> int:
        return int(self.draft_rgb.shape[2])


def load_controls(path: Path) -> Controls:
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

        draft = data["draft_rgb"]
        source = data["source_rgb"]
        reference = data["reference_rgb"]
        mask_raw = data["edit_mask"]

        for name, arr in (("draft_rgb", draft), ("source_rgb", source)):
            if arr.dtype != np.uint8:
                raise ControlError(f"{name} must be uint8, got {arr.dtype}")
        if draft.ndim != 4 or draft.shape[3] != 3:
            raise ControlError(f"draft_rgb must be [T,H,W,3], got {draft.shape}")
        T, H, W = (int(x) for x in draft.shape[:3])
        if source.shape != (T, H, W, 3):
            raise ControlError(f"source_rgb must match draft_rgb [{T},{H},{W},3], got {source.shape}")
        if T < 5 or (T - 1) % 4 != 0:
            raise ControlError(f"T must be 4k+1 (k>=1), got T={T}")
        if H <= 0 or W <= 0 or H % 16 != 0 or W % 16 != 0:
            raise ControlError(f"H/W must be positive multiples of 16 (Wan VAE/patch), got {H}x{W}")
        if reference.dtype != np.uint8 or reference.shape != (H, W, 3):
            raise ControlError(f"reference_rgb must be uint8 [{H},{W},3], got {reference.shape} {reference.dtype}")
        if not np.array_equal(reference, source[0]):
            raise ControlError("reference_rgb must equal the first actual source frame source_rgb[0]")

        # mask: bool を厳格に要求（{0,255} uint8 のみ許容。gray 等の曖昧値は拒否）
        if mask_raw.dtype == np.bool_:
            mask = mask_raw
        elif mask_raw.dtype == np.uint8 and set(np.unique(mask_raw).tolist()) <= {0, 255}:
            warning = "edit_mask was uint8 {0,255}; converted to bool"
            warnings.append(warning)
            mask = mask_raw.astype(bool)
        else:
            raise ControlError(
                "edit_mask must be bool (or uint8 with only {0,255}); got "
                f"dtype={mask_raw.dtype} values={np.unique(mask_raw)[:10].tolist()}. "
                "Gray/ambiguous mask values are rejected: white(255)=generate, black(0)=preserve."
            )
        if mask.shape != (T, H, W):
            raise ControlError(f"edit_mask must be [{T},{H},{W}], got {mask.shape}")

        # unknown=生成対象の draft は flat gray 127 でなければならない
        unknown = draft[mask]
        if unknown.size:
            bad = np.unique(unknown)
            if bad.size and not np.all(bad == DRAFT_UNKNOWN_VALUE):
                offenders = bad[bad != DRAFT_UNKNOWN_VALUE][:8].tolist()
                raise ControlError(
                    "draft_rgb under edit_mask=True (unknown/generate) must be exactly "
                    f"{DRAFT_UNKNOWN_VALUE}; found other values={offenders}"
                )

        unknown_count = int(mask.sum())
        known_count = int(mask.size - unknown_count)
        if unknown_count == 0:
            warnings.append("edit_mask is all-False: nothing to generate; composite equals draft")
        if known_count == 0:
            warnings.append("edit_mask is all-True: every pixel generated; composite equals raw model output")

        fps_arr = np.asarray(data["fps"]).astype(np.float64)
        if fps_arr.ndim != 0 or not np.isfinite(fps_arr) or float(fps_arr) <= 0:
            raise ControlError(f"fps must be a finite positive scalar, got {fps_arr!r}")
        fps = float(fps_arr)

    return Controls(
        path=path,
        sha256=sha256_file(path),
        draft_rgb=draft,
        edit_mask=mask,
        source_rgb=source,
        reference_rgb=reference,
        fps=fps,
        warnings=warnings,
    )


def controls_metadata(controls: Controls) -> dict:
    mask = controls.edit_mask
    preserved = ~mask
    draft_is_source = np.all(controls.draft_rgb[preserved] == controls.source_rgb[preserved], axis=-1)
    source_equal_pixels = int(draft_is_source.sum())
    return {
        "path": str(controls.path),
        "sha256": controls.sha256,
        "T": controls.T,
        "H": controls.H,
        "W": controls.W,
        "fps": controls.fps,
        "unknown_pixels": int(mask.sum()),
        "preserved_pixels": int(preserved.sum()),
        "unknown_fraction": float(mask.mean()),
        "preserved_pixels_where_draft_equals_source": source_equal_pixels,
        "preserved_pixels_note": (
            "pixel equality counts RGB triplets, not channels; equality is not provenance. "
            "Exact preservation applies to arrays/PNG; MP4 is lossy."
        ),
        "draft_rgb_sha256": sha256_array(controls.draft_rgb),
        "edit_mask_sha256": sha256_array(controls.edit_mask),
        "source_rgb_sha256": sha256_array(controls.source_rgb),
        "reference_rgb_sha256": sha256_array(controls.reference_rgb),
        "warnings": controls.warnings,
        "mask_convention": MASK_CONVENTION,
    }


# ---------------------------------------------------------------------------
# PIL 変換
# ---------------------------------------------------------------------------


def draft_to_pil_frames(draft_rgb: np.ndarray) -> list:
    from PIL import Image

    return [Image.fromarray(frame, mode="RGB") for frame in draft_rgb]


def mask_to_pil_frames(edit_mask: np.ndarray) -> list:
    """bool mask → white(255)=generate/unknown, black(0)=preserve の L-mode PIL。"""
    from PIL import Image

    return [Image.fromarray((frame.astype(np.uint8) * 255), mode="L") for frame in edit_mask]


def reference_to_pil(reference_rgb: np.ndarray):
    from PIL import Image

    return Image.fromarray(reference_rgb, mode="RGB")


# ---------------------------------------------------------------------------
# pipeline 構築（公式 WanVACEPipeline。VAE は fp32、他は bf16、CPU offload + tiling）
# ---------------------------------------------------------------------------


def load_pipeline(model_path: Path, device: str):
    """公式コンポーネントをローカルから構築する。失敗時の fallback は一切しない。

    quantization=none の既存パス（この関数は変更しない）。
    """
    diffusers_module = importlib.import_module("diffusers")
    AutoencoderKLWan = getattr(diffusers_module, "AutoencoderKLWan")
    WanVACEPipeline = getattr(diffusers_module, "WanVACEPipeline")

    model_path = Path(model_path)
    pipe = WanVACEPipeline.from_pretrained(str(model_path), torch_dtype=torch.bfloat16)
    # 公式慣例: VAE は float32（他は bf16）。offload hook 装着前に差し替える。
    pipe.vae = AutoencoderKLWan.from_pretrained(str(model_path), subfolder="vae", torch_dtype=torch.float32)
    pipe.enable_model_cpu_offload(device=device)
    pipe.vae.enable_tiling()
    return pipe


# ---------------------------------------------------------------------------
# quantization=nf4（14B）。実装順序は coordinator 指定に厳密に従う:
#   quantized transformer を cuda:0 でだけ load -> dtype 引数なしで CPU へ -> FP32 VAE
#   -> from_pretrained に実オブジェクトを供給 -> cpu offload -> vae tiling
# full BF16 transformer の確保・pipe.to('cuda')・quantized への dtype 変換は禁止。
# ---------------------------------------------------------------------------


def _version_tuple(value: str) -> tuple:
    parts = []
    for chunk in value.split("."):
        digits = ""
        for char in chunk:
            if char.isdigit():
                digits += char
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _bitsandbytes_version() -> str | None:
    from importlib import metadata

    try:
        return metadata.version("bitsandbytes")
    except metadata.PackageNotFoundError:
        return None


def require_bitsandbytes() -> str:
    """nf4 は bitsandbytes>=0.46.1 を要求する。無い/古い場合は silently fallback せず fail。"""
    version = _bitsandbytes_version()
    if version is None:
        raise AdapterError(
            f"--quantization nf4 requires bitsandbytes>={BITSANDBYTES_MIN_VERSION}; "
            "bitsandbytes is not installed (no fallback)"
        )
    if _version_tuple(version) < _version_tuple(BITSANDBYTES_MIN_VERSION):
        raise AdapterError(
            f"--quantization nf4 requires bitsandbytes>={BITSANDBYTES_MIN_VERSION}, found {version} "
            "(older Params4bit.to may not preserve bnb_quantized across CPU/GPU moves; no fallback)"
        )
    return version


def quantization_report(args) -> dict:
    mode = args.quantization
    if mode == "nf4":
        version = _bitsandbytes_version()
    else:
        version = _bitsandbytes_version()  # 記録のみ（none の挙動には使わない）
    return {
        "mode": mode,
        "requested_nf4_compute_dtype": NF4_COMPUTE_DTYPE if mode == "nf4" else None,
        "bitsandbytes_version": version,
        "bitsandbytes_min_version": BITSANDBYTES_MIN_VERSION,
        "note": (
            "nf4 loads only the quantized transformer; none path is the prior full-precision path"
        ),
    }


def _get_diffusers_classes() -> tuple:
    diffusers_module = importlib.import_module("diffusers")
    return (
        getattr(diffusers_module, "WanVACETransformer3DModel"),
        getattr(diffusers_module, "AutoencoderKLWan"),
        getattr(diffusers_module, "WanVACEPipeline"),
    )


def actual_transformer_report(transformer) -> dict:
    config = getattr(transformer.config, "quantization_config", None)
    if config is not None and hasattr(config, "to_dict"):
        config = config.to_dict()
    report = {
        "dtype": str(transformer.dtype),
        "nf4_layer_count": sum(type(m).__name__ == "Linear4bit" for m in transformer.modules()),
        "quantization_config": config,
    }
    # Validate before expensive generation, including real Diffusers config objects.
    json.dumps(report, allow_nan=False)
    return report


def load_quantized_transformer(model_path: Path, device: str):
    """NF4 transformer のみを cuda:0 (device_map={'':0}) で load し、dtype 引数なしで CPU へ移す。"""
    BitsAndBytesConfig = getattr(importlib.import_module("diffusers"), "BitsAndBytesConfig")

    transformer_cls, _, _ = _get_diffusers_classes()
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    transformer = transformer_cls.from_pretrained(
        str(Path(model_path) / "transformer"),
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,  # 非量子化の Conv/Norm 等も BF16 でロード
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map={"": 0},  # cuda:0 上でのみ量子化ロード（full BF16 は確保しない）
    )
    # dtype 引数なしの明示 CPU 移動（bitsandbytes>=0.46.1 が bnb_quantized を保持）
    transformer = transformer.to("cpu")
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    return transformer


def load_pipeline_nf4(model_path: Path, device: str):
    """nf4 経路の pipeline 構築。require_bitsandbytes は最初に、重み load より前に実行する。"""
    require_bitsandbytes()
    model_path = Path(model_path)
    transformer = load_quantized_transformer(model_path, device)
    _, AutoencoderKLWan, WanVACEPipeline = _get_diffusers_classes()
    # FP32 VAE（official 慣例）。offload hook 装着前に構築。
    vae = AutoencoderKLWan.from_pretrained(str(model_path), subfolder="vae", torch_dtype=torch.float32)
    # 実オブジェクト供給: diffusers は渡されたコンポーネントをそのまま使い dtype 変換しない。
    pipe = WanVACEPipeline.from_pretrained(
        str(model_path),
        transformer=transformer,
        vae=vae,
        torch_dtype=torch.bfloat16,  # 他コンポーネント（tokenizer/text encoder/scheduler）用
    )
    pipe.enable_model_cpu_offload(device=device)
    pipe.vae.enable_tiling()
    return pipe


def build_generator(seed: int, device: str) -> torch.Generator:
    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch.Generator(device=device).manual_seed(seed)


def configure_scheduler(pipe, flow_shift: float | None) -> dict:
    """Optional explicit author-recipe ablation; default preserves model config."""
    if flow_shift is not None:
        scheduler_class = getattr(importlib.import_module("diffusers"), "UniPCMultistepScheduler")
        if not isinstance(pipe.scheduler, scheduler_class) or not pipe.scheduler.config.use_flow_sigmas:
            raise AdapterError("--flow-shift requires UniPC with use_flow_sigmas=True")
        pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config, flow_shift=flow_shift)
    return {"class": type(pipe.scheduler).__name__,
            "flow_shift": pipe.scheduler.config.get("flow_shift"),
            "flow_shift_override": flow_shift}


def call_vace_pipeline(pipe, controls: Controls, args, device: str):
    """公式 pipeline を正確な kwargs で呼ぶ（テストが args を検証できるよう分離）。"""
    generator = build_generator(int(args.seed), device)
    return pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        video=draft_to_pil_frames(controls.draft_rgb),
        mask=mask_to_pil_frames(controls.edit_mask),
        reference_images=[reference_to_pil(controls.reference_rgb)],
        conditioning_scale=float(args.conditioning_scale),
        height=controls.H,
        width=controls.W,
        num_frames=controls.T,
        num_inference_steps=int(args.steps),
        guidance_scale=float(args.guidance),
        generator=generator,
        output_type="np",
    )


def extract_raw_frames(result, expected_T: int, H: int, W: int) -> np.ndarray:
    """pipeline 出力 np [1,T,H,W,C] float[0,1] を厳密に検証し uint8 へ量子化する。

    実測事実 (evidence/engine/vace/postprocess_np_receipt.json, 実 VideoProcessor):
      postprocess_video(output_type='np') は floating [0,1] (float32) を返す。
      uint8 化は明示的 quantize: round(x*255) (rint, half-to-even; 0->0, 0.5->128, 1->255)。
    量子化以外の補正はしない。out-of-range や non-finite はクリップで隠さず fail する。
    """
    frames = result.frames if hasattr(result, "frames") else result
    frames = np.asarray(frames)
    if frames.shape != (1, expected_T, H, W, 3):
        raise AdapterError(
            f"pipeline output shape mismatch: got {frames.shape}, expected (1,{expected_T},{H},{W},3); "
            "no silent resize/missing-frame correction is applied"
        )
    if not np.issubdtype(frames.dtype, np.floating):
        raise AdapterError(
            f"pipeline output dtype must be floating [0,1] (installed postprocessor contract), got {frames.dtype}"
        )
    if not np.isfinite(frames).all():
        raise AdapterError("pipeline output contains non-finite values")
    tol = 1e-6  # official processor は clamp 済み; 丸め誤差のみ許容し、外れ値は隠さない
    if float(frames.min()) < -tol or float(frames.max()) > 1.0 + tol:
        raise AdapterError(
            f"pipeline output out of [0,1]: min={float(frames.min()):.6g} max={float(frames.max()):.6g}"
        )
    return np.rint(frames[0] * 255.0).astype(np.uint8)  # documented quantization only


# ---------------------------------------------------------------------------
# composite 検査と出力
# ---------------------------------------------------------------------------


def composite_frames(raw_frames: np.ndarray, controls: Controls) -> np.ndarray:
    return np.where(controls.edit_mask[:, :, :, None], raw_frames, controls.draft_rgb)


def check_outside_mask_equality(composite: np.ndarray, draft_rgb: np.ndarray, edit_mask: np.ndarray) -> dict:
    """composite の preserved 画素が draft と厳密一致することを独立に検査する。"""
    outside = ~edit_mask
    if int(outside.sum()) == 0:
        return {
            "outside_mask_exact_equal_to_draft": "not_applicable",
            "checked_preserved_pixels": 0,
            "note": "no preserved pixels (all-True mask); nothing to verify outside mask",
        }
    equal_outside = np.array_equal(composite[outside], draft_rgb[outside])
    if not equal_outside:
        mismatched = int(np.count_nonzero(np.any(composite[outside] != draft_rgb[outside], axis=-1)))
        raise AdapterError(
            f"composite outside-mask equality to draft failed: {mismatched} mismatched preserved pixels"
        )
    return {
        "outside_mask_exact_equal_to_draft": True,
        "checked_preserved_pixels": int(outside.sum()),
        "note": "exact uint8 equality; independently recomputed, not assumed from construction",
    }


def ffprobe_frame_count(path: Path) -> int:
    """ffprobe で実 MP4 frame 数を数える（silent 補正の検出）。"""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise AdapterError(f"ffprobe failed for {path}: {result.stderr.strip()[:300]}")
    info = json.loads(result.stdout)
    streams = info.get("streams") or []
    if not streams or "nb_read_frames" not in streams[0]:
        raise AdapterError(f"ffprobe returned no frame count for {path}")
    return int(streams[0]["nb_read_frames"])


def save_video_frames(frames_u8: np.ndarray, output_dir: Path, fps: float, stem: str) -> dict:
    imageio = importlib.import_module("imageio")
    from PIL import Image

    frames_dir = output_dir / f"{stem}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(frames_u8.shape[0]):
        frame_path = frames_dir / f"frame_{index:04d}.png"
        Image.fromarray(frames_u8[index], mode="RGB").save(frame_path)
        records.append({"name": frame_path.name, "sha256": sha256_file(frame_path)})

    mp4_path = output_dir / f"{stem}.mp4"
    with imageio.get_writer(str(mp4_path), fps=float(fps), codec="libx264", pixelformat="yuv420p") as writer:
        for index in range(frames_u8.shape[0]):
            writer.append_data(frames_u8[index])

    combined = hashlib.sha256()
    for record in records:
        combined.update(f"{record['name']}:{record['sha256']}\n".encode())
    probed = ffprobe_frame_count(mp4_path)
    return {
        "frames_dir": str(frames_dir),
        "n_frames": int(frames_u8.shape[0]),
        "frames_sha256_combined": combined.hexdigest(),
        "frames": records,
        "mp4": {
            "path": str(mp4_path),
            "bytes": mp4_path.stat().st_size,
            "sha256": sha256_file(mp4_path),
            "fps": float(fps),
            "ffprobe_frame_count": probed,
        },
    }


def save_masks(controls: Controls, output_dir: Path) -> dict:
    from PIL import Image

    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, frame in enumerate(controls.edit_mask):
        path = masks_dir / f"mask_{index:04d}.png"
        Image.fromarray(frame.astype(np.uint8) * 255, mode="L").save(path)
        records.append({"name": path.name, "sha256": sha256_file(path)})
    combined = hashlib.sha256()
    for record in records:
        combined.update(f"{record['name']}:{record['sha256']}\n".encode())
    return {
        "dir": str(masks_dir),
        "masks_sha256_combined": combined.hexdigest(),
        "controls_path": str(controls.path),
        "controls_sha256": controls.sha256,
        "convention": MASK_CONVENTION,
    }


# ---------------------------------------------------------------------------
# 実行
# ---------------------------------------------------------------------------


def package_versions() -> dict:
    from importlib import metadata

    names = (
        "diffusers",
        "transformers",
        "torch",
        "accelerate",
        "huggingface-hub",
        "numpy",
        "safetensors",
        "pillow",
        "imageio",
        "imageio-ffmpeg",
        "bitsandbytes",
        "ftfy",
    )
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def model_report(model_path: Path, args=None) -> dict:
    report: dict = {
        "path": str(model_path),
        "present": model_path.is_dir(),
        "requested_model_id": getattr(args, "model_id", None) if args is not None else None,
        "requested_revision": getattr(args, "model_revision", None) if args is not None else None,
        "actual_revision": None,
        "actual_revision_note": (
            "no real local receipt verifies the actual revision; requested identity/revision are "
            "reported separately and never claimed as actual"
        ),
    }
    index_path = model_path / "model_index.json"
    if index_path.is_file():
        report["model_index_sha256"] = sha256_file(index_path)
        report["model_index"] = json.loads(index_path.read_text())
    transformer_config = model_path / "transformer" / "config.json"
    if transformer_config.is_file():
        report["transformer_config_sha256"] = sha256_file(transformer_config)
        config = json.loads(transformer_config.read_text())
        report["transformer_config_summary"] = {
            key: config.get(key)
            for key in (
                "in_channels",
                "num_layers",
                "patch_size",
                "vace_layers",
                "vace_in_channels",
                "rope_max_seq_len",
            )
        }
    provenance_path = model_path / "DERIVED_BF16_PROVENANCE.json"
    if provenance_path.is_file():
        report["model_provenance"] = {
            "path": str(provenance_path),
            "sha256": sha256_file(provenance_path),
            "note": (
                "derived local checkpoint (converted); this is not untouched upstream bytes. "
                "Official raw hashes and BF16 derived hashes are retained by root's converter."
            ),
        }
    report["local_snapshot_receipt"] = None  # cache directory existence does not verify weights
    report["note"] = "metadata only; weights are not loaded in dry-run"
    return report


def require_cuda() -> str:
    if not torch.cuda.is_available():
        raise AdapterError("CUDA is required for actual generation (use --dry-run for CPU checks)")
    return "cuda"


def run_dry_run(args) -> int:
    controls = load_controls(Path(args.controls))
    if args.quantization == "nf4":
        require_bitsandbytes()  # dependency version check; no weights are loaded
    diffusers_module = importlib.import_module("diffusers")
    AutoencoderKLWan = getattr(diffusers_module, "AutoencoderKLWan")
    WanVACEPipeline = getattr(diffusers_module, "WanVACEPipeline")


    draft_frames = draft_to_pil_frames(controls.draft_rgb)
    mask_frames = mask_to_pil_frames(controls.edit_mask)
    reference = reference_to_pil(controls.reference_rgb)
    report = {
        "status": "dry_run_ok",
        "at": utc_now(),
        "argv": sys.argv,
        "controls": controls_metadata(controls),
        "mask_convention_confirmed": MASK_CONVENTION,
        "flow_shift_override": args.flow_shift,
        "planned_pipeline_call": {
            "video": f"list of {len(draft_frames)} RGB PIL frames (draft, unknown=gray127)",
            "mask": f"list of {len(mask_frames)} L PIL frames (255=generate, 0=preserve)",
            "reference_images": [f"1 RGB PIL frame {reference.size} (== source_rgb[0])"],
            "conditioning_scale": float(args.conditioning_scale),
            "height": controls.H,
            "width": controls.W,
            "num_frames": controls.T,
            "num_inference_steps": int(args.steps),
            "guidance_scale": float(args.guidance),
            "output_type": "np",
            "expected_output_shape": [1, controls.T, controls.H, controls.W, 3],
        },
        "pipeline": {
            "diffusers_version": diffusers_module.__version__,
            "pipeline_cls": f"{WanVACEPipeline.__module__}.{WanVACEPipeline.__qualname__}",
            "vae_cls": f"{AutoencoderKLWan.__module__}.{AutoencoderKLWan.__qualname__}",
            "dtype_plan": "pipeline torch_dtype=bfloat16; VAE replaced with float32; enable_model_cpu_offload; vae tiling",
        },
        "quantization": quantization_report(args),
        "model": model_report(Path(args.model), args) if args.model else None,
        "device": {
            "cuda_available": torch.cuda.is_available(),
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "versions": package_versions(),
        "note": "weights not loaded; generation is root-owned",
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def run_generation(args) -> int:
    controls = load_controls(Path(args.controls))
    if not args.model or not args.output_dir:
        raise AdapterError("--model and --output-dir are required for generation")

    model_path = Path(args.model)
    if not (model_path / "model_index.json").is_file():
        raise AdapterError(f"model dir incomplete (no model_index.json): {model_path}")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise AdapterError(f"output dir is occupied: {output_dir} (refusing to overwrite)")
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    result: dict = {
        "status": "running",
        "argv": sys.argv,
        "at": utc_now(),
        "steps": int(args.steps),
        "seed": int(args.seed),
        "guidance": float(args.guidance),
        "conditioning_scale": float(args.conditioning_scale),
        "flow_shift_override": args.flow_shift,
        "quantization": quantization_report(args),
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "controls": controls_metadata(controls),
        "model": model_report(model_path, args),
        "mask_convention_confirmed": MASK_CONVENTION,
        "output_meaning": (
            "raw_frames/raw.mp4 = raw model output; composite_frames/composite.mp4 = "
            "where(edit_mask, raw_model, draft); not metric RGB-D"
        ),
        "versions": package_versions(),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    try:
        device = require_cuda()
        load_start = time.monotonic()
        if args.quantization == "nf4":
            pipe = load_pipeline_nf4(model_path, device)
        else:
            pipe = load_pipeline(model_path, device)
        result["scheduler"] = configure_scheduler(pipe, args.flow_shift)
        result["actual_transformer"] = actual_transformer_report(pipe.transformer)
        result["timing_load_seconds"] = time.monotonic() - load_start
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        infer_start = time.monotonic()
        with torch.no_grad():
            pipeline_result = call_vace_pipeline(pipe, controls, args, device)
        result["timing_infer_seconds"] = time.monotonic() - infer_start

        raw_frames = extract_raw_frames(pipeline_result, controls.T, controls.H, controls.W)
        if not np.isfinite(raw_frames).all():
            raise AdapterError("raw frames contain non-finite values")
        composite = composite_frames(raw_frames, controls)
        equality = check_outside_mask_equality(composite, controls.draft_rgb, controls.edit_mask)

        save_start = time.monotonic()
        raw_outputs = save_video_frames(raw_frames, output_dir, controls.fps, "raw")
        composite_outputs = save_video_frames(composite, output_dir, controls.fps, "composite")
        mask_outputs = save_masks(controls, output_dir)
        result["timing_save_seconds"] = time.monotonic() - save_start

        for name, outputs in (("raw", raw_outputs), ("composite", composite_outputs)):
            if outputs["mp4"]["ffprobe_frame_count"] != controls.T:
                raise AdapterError(
                    f"{name}.mp4 ffprobe frame count {outputs['mp4']['ffprobe_frame_count']} != T={controls.T}"
                )
        result["outputs"] = {
            "raw": raw_outputs,
            "composite": composite_outputs,
            "masks": mask_outputs,
            "outside_mask_equality_check": equality,
        }
        result["peak_memory"] = {
            "scope": "torch.cuda.max_memory_allocated since reset before inference (generation scope)",
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
        }
        result["status"] = "ok"
        result["timing_total_seconds"] = time.monotonic() - t0
        result["at_end"] = utc_now()
        manifest_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"status": "ok", "outputs": result["outputs"]}, indent=2, ensure_ascii=False))
        return 0
    except Exception as error:  # noqa: BLE001 - 失敗は manifest_failed.json と非ゼロ終了で明示
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
        result["timing_total_seconds"] = time.monotonic() - t0
        result["at_end"] = utc_now()
        (output_dir / "manifest_failed.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )
        print(result["error"], file=sys.stderr)
        return 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VACE masked completion adapter (official diffusers WanVACEPipeline)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=None, help="local Wan2.1-VACE-1.3B-diffusers dir")
    parser.add_argument("--controls", required=True, help="controls NPZ (no pickles)")
    parser.add_argument("--output-dir", default=None, help="output dir (must be unoccupied)")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--conditioning-scale", type=float, default=1.0)
    parser.add_argument("--flow-shift", type=float, default=None,
                        help="explicit UniPC flow shift override; default keeps model config")
    parser.add_argument("--quantization", choices=["none", "nf4"], default="none",
                        help="none = prior full-precision path; nf4 = 4-bit quantized transformer")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                        help="requested model identity (provenance; not claimed as actual)")
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION,
                        help="requested revision (provenance; not claimed as actual)")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_options(args) -> None:
    if args.steps <= 0:
        raise AdapterError("--steps must be positive")
    for name in ("guidance", "conditioning_scale"):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            raise AdapterError(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if not 0 <= args.seed < 2**64:
        raise AdapterError("--seed must be in [0, 2**64)")
    if args.flow_shift is not None and (not np.isfinite(args.flow_shift) or args.flow_shift <= 0):
        raise AdapterError("--flow-shift must be finite and positive")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        validate_options(args)
        if args.dry_run:
            return run_dry_run(args)
        return run_generation(args)
    except AdapterError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
