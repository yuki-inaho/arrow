#!/usr/bin/env python3
"""ARROW 深度詳細化パイプラインの実行CLI（pipeline担当 scope）。

正本: temp/workdoc_Oct16-2026_depth_workflow.md（2026-10-16 最新ユーザー方針）。
主経路: ARROW clip (低解像度depth, 暫定尺度) -> clip Sim3 尺度統一 -> fullres anchor + pixel K
-> LingBot-Depth -> ARROW anchor へ robust log ratio 再固定 -> depth由来normal + gray edge NDR
-> camera補償 scene flow -> edge/depth guided scene flow SR -> 時間融合 -> 共通range の比較映像。

単位は arrow_provisional (metric_calibrated=false)。全 depth / camera translation / 3D flow は
後段で単一 scalar 乗算により一括校正できる構造を保つ。

使い方::

    # 実行計画の確認（推論しない）
    uv run --no-sync python scripts/run_depth_detail.py --dataset all --dry-run

    # 全段階（GPUジョブは統括が lease 後に実行する。本runnerはGPUを勝手に起動しない）
    uv run --no-sync python scripts/run_depth_detail.py --dataset all --stage all \
        --output-dir results/depth_detail_oct16

    # 個別 stage（resume: 入力/設定hash一致と出力存在を確認して skip）
    uv run --no-sync python scripts/run_depth_detail.py --dataset grass --stage ndr

すべての段階は専用 venv の subprocess を literal argv で起動し、
attempts/<stage>_<utc>/exit.json に実測終了コードを保存する。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
MODELS_SCRIPT = SCRIPTS / "depth_detail_models.py"
sys.path.insert(0, str(SCRIPTS))

import depth_detail_models as models  # noqa: E402

NONCE = "82e63c25084c"
UNITS = models.UNITS
DATASET_ROOT = "results/depth_detail_oct16"
GRASS_DIR = Path("/home/kasm-user/Downloads/grass/douga_2026-09-15_14-16-51_frames_5fps")
RAILWAY_VIDEO = REPO / "assets" / "videos" / "railway.mp4"

ARROW_VENV_PYTHON = REPO / ".venv" / "bin" / "python"
MOGE_VENV_PYTHON = Path("/home/kasm-user/Desktop/MoGe/.venv/bin/python")
LINGBOT_VENV_PYTHON = Path("/home/kasm-user/Desktop/lingbot-depth/.venv/bin/python")
NDR_VENV_PYTHON = Path("/home/kasm-user/Desktop/normal_depth_refiner_sandbox/.venv/bin/python")
SCENEFLOW_VENV_PYTHON = Path("/home/kasm-user/Desktop/rgbd_sceneflow_sandbox/.venv/bin/python")
SCENEFLOW_PYTHONPATH = models.SCENEFLOW_PYTHONPATH

# LingBot venv は torch2.6+cu124 wheel 内蔵 cuDNN を前置する（ARROW の cu130 lib は流用しない）
LINGBOT_CUDNN_LIB = Path(
    "/home/kasm-user/Desktop/lingbot-depth/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib"
)

CUDNN_LIB = REPO / ".venv" / "lib" / "python3.13" / "site-packages" / "nvidia" / "cudnn" / "lib"

DATASETS = {
    "grass": {"source": GRASS_DIR, "fps": 5.0, "expected_frames": 107},
    "railway": {"source": RAILWAY_VIDEO, "fps": 24.0, "expected_frames": 80},
}

STAGE_ORDER = [
    "ingest", "arrow", "align", "anchor", "geometry", "lingbot", "prep",
    "ndr", "sceneflow", "sr", "temporal", "final", "render",
]
OPTIONAL_STAGES = ["moge"]
MANIFEST_STAGES = STAGE_ORDER
STAGE_CHAIN = models.STAGE_CHAIN
STAGE_CHAIN_FULL = models.STAGE_CHAIN_FULL


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def code_hash() -> str:
    files = {
        "run_depth_detail": SCRIPTS / "run_depth_detail.py",
        "depth_detail_models": MODELS_SCRIPT,
        "core_depth_detail": REPO / "src" / "arrow" / "depth_detail.py",
    }
    digests = {name: models.sha256_file(path) for name, path in files.items()}
    return models.sha256_bytes(models.canonical_json(digests).encode("utf-8"))


@dataclass
class Context:
    args: argparse.Namespace
    dataset: str
    source: Path
    root: Path  # <output-dir>/<dataset>
    frames_json: Path
    expected_frames: int
    video_fps: float
    clips_dir: Path = field(init=False)
    alignments: Path = field(init=False)

    def __post_init__(self):
        self.clips_dir = self.root / "stages" / "arrow_clips"
        self.alignments = self.root / "stages" / "align" / "alignments.json"

    @property
    def frames(self) -> list[dict]:
        if not self.frames_json.is_file():
            raise SystemExit(
                f"frames.json がありません: {self.frames_json}（先に --stage ingest を実行）"
            )
        payload = models.load_json(self.frames_json)
        return payload["frames"]

    @property
    def original_size(self) -> list[int]:
        if not self.frames_json.is_file():
            raise SystemExit(
                f"frames.json がありません: {self.frames_json}（先に --stage ingest を実行）"
            )
        payload = models.load_json(self.frames_json)
        return list(payload["original_size"])

    def stage_dir(self, stage: str) -> Path:
        return self.root / "stages" / stage

    def config_dir(self) -> Path:
        return self.root / "configs" / "stage_configs"

    def attempt_dir(self, stage: str) -> Path:
        return self.root / "attempts"

    def signature_path(self, stage: str) -> Path:
        return self.root / "signatures" / f"{stage}.json"


# --- CLI ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ARROW 深度詳細化パイプライン（段階実行 + resume + 証跡記録）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", choices=["grass", "railway", "all"], required=True,
                        help="対象データセット（grass=107枚 / railway=80フレーム）")
    parser.add_argument("--output-dir", type=Path, default=REPO / DATASET_ROOT,
                        help=f"出力先（既定: {DATASET_ROOT}）")
    parser.add_argument("--stage", default="all",
                        choices=["all", *STAGE_ORDER, *OPTIONAL_STAGES],
                        help="実行する段階（既定: all。mogeは--with-moge時のみ）")
    parser.add_argument("--dry-run", action="store_true", help="実行計画とargvを表示する（何も実行しない）")
    parser.add_argument("--force", action="store_true", help="resume判定を無視して再実行する")
    parser.add_argument("--with-moge", action="store_true",
                        help="MoGe v3 を OPTIONAL 比較として追加実行する（主経路外）")
    parser.add_argument("--grass-dir", type=Path, default=GRASS_DIR, help="grass 入力（テスト用上書き）")
    parser.add_argument("--railway-video", type=Path, default=RAILWAY_VIDEO,
                        help="railway 入力（テスト用上書き）")
    parser.add_argument("--device", default="cuda", help="モデル実行デバイス（黙って変更しない）")
    parser.add_argument(
        "--arrow-device", default=None,
        help="ARROW stage の --device 実解決値の上書き。None(既定)=--device を使用、"
             "auto=process_sequence の --device を省略（既存 prefill signature device=null と同条件）、"
             "cuda/cpu 等は明示指定。LingBot の --device には影響しない",
    )
    parser.add_argument("--lingbot-model", default="robbyant/lingbot-depth-pretrain-vitl-14-v0.5")
    parser.add_argument("--lingbot-resolution-level", type=int, default=9)
    parser.add_argument("--moge-model", default="Ruicheng/moge-3-vitl")
    parser.add_argument("--moge-num-tokens", type=int, default=3600)
    parser.add_argument("--moge-refine-steps", type=int, default=3)
    parser.add_argument("--ndr-backend", default="rust", help="NDR backend（既定 rust, fallback禁止）")
    parser.add_argument("--ndr-iterations", type=int, default=40)
    parser.add_argument("--clip-frames", type=int, default=24)
    parser.add_argument("--clip-overlap", type=int, default=4)
    parser.add_argument("--max-side", type=int, default=504)
    parser.add_argument("--track-stride", type=int, default=2)
    parser.add_argument("--reconstruction-stride", type=int, default=2)
    parser.add_argument("--depth-stride", type=int, default=2)
    parser.add_argument("--long-side", type=int, default=160, help="scene flow 低解像度の長辺")
    parser.add_argument("--trust-abs", type=float, default=0.05,
                        help="trust 絶対幅（arrow_provisional 単位の開始値）")
    parser.add_argument("--trust-rel", type=float, default=0.05)
    parser.add_argument("--fusion-weight", type=float, default=0.25)
    parser.add_argument("--sceneflow-max-motion-frac", type=float, default=0.05,
                        help="max_motion = frac * median depth（全prepared入力scanのmedian; 暫定単位）")
    parser.add_argument("--sceneflow-workers", type=int, default=1,
                        help="sceneflow_refine のCPU並列数（default 1。C++ ctypes はGIL解放・errorはthread_local）")
    parser.add_argument("--cell-width", type=int, default=256)
    parser.add_argument("--point-stride", type=int, default=4)
    parser.add_argument("--video-fps", type=float, default=None, help="比較MP4のfps（既定: データ由来）")
    return parser


def dataset_contexts(args: argparse.Namespace, *, require_frames: bool) -> list[Context]:
    names = ["grass", "railway"] if args.dataset == "all" else [args.dataset]
    contexts = []
    for name in names:
        spec = DATASETS[name]
        source = args.grass_dir if name == "grass" else args.railway_video
        if not Path(source).exists():
            raise SystemExit(f"入力が見つかりません: {source}")
        root = args.output_dir / name
        contexts.append(
            Context(
                args=args,
                dataset=name,
                source=Path(source),
                root=root,
                frames_json=root / "frames.json",
                expected_frames=int(spec["expected_frames"]),
                video_fps=float(spec["fps"]),
            )
        )
    if require_frames:
        for context in contexts:
            if not context.frames_json.is_file():
                raise SystemExit(
                    f"frames.json がありません: {context.frames_json}（先に --stage ingest を実行）"
                )
    return contexts


# --- signature / resume ------------------------------------------------------


def source_fingerprint(context: Context) -> dict:
    source = context.source
    cache = context.root / "configs" / "source_fingerprint.json"
    if source.is_dir():
        entries = []
        for path in models.list_image_files(source):
            entries.append({"name": path.name, "bytes": path.stat().st_size,
                            "sha256": models.sha256_file(path)})
        fingerprint = {"kind": "folder", "count": len(entries),
                       "digest": models.sha256_bytes(models.canonical_json(entries).encode("utf-8"))}
    else:
        fingerprint = {"kind": "video", "count": None,
                       "digest": models.sha256_file(source)}
    try:
        cached = models.load_json(cache)
        if cached == fingerprint:
            return fingerprint
    except (OSError, ValueError):
        pass
    if not context.args.dry_run:
        models.atomic_write_json(cache, fingerprint)
    return fingerprint


def compute_signature(context: Context, stage: str, *, inputs: list[dict], params: dict,
                      model: str, argv: list,
                      chain: dict[str, str] | None = None) -> dict:
    combined_inputs = list(inputs)
    for name, value in (chain or {}).items():
        combined_inputs.append({"path": f"chain:{name}", "sha256": value})
    code = code_hash()
    payload = {
        "stage": stage,
        "dataset": context.dataset,
        "inputs": combined_inputs,
        "params": params,
        "model": model,
        "argv": [str(item) for item in argv],
        "code_sha256": code,
    }
    payload["signature_hash"] = models.sha256_bytes(
        models.canonical_json(payload).encode("utf-8")
    )
    return payload


def chain_hashes(context: Context) -> dict[str, str]:
    signatures = {}
    for name in STAGE_ORDER + (["moge"] if (context.args.with_moge or context.args.stage == "moge") else []):
        path = context.signature_path(name)
        if path.is_file():
            try:
                signatures[name] = models.load_json(path).get("signature_hash", "")
            except (OSError, ValueError):
                continue
    return signatures


def previous_chain(context: Context, stage: str) -> dict[str, str]:
    hashes = chain_hashes(context)
    ordered = [name for name in STAGE_ORDER if name != stage]
    result = {}
    for name in ordered:
        if name in hashes:
            result[name] = hashes[name]
        if name == stage:
            break
    return result


def write_signature(context: Context, payload: dict) -> Path:
    path = context.signature_path(payload["stage"])
    models.atomic_write_json(path, payload)
    return path


def stage_outputs_present(context: Context, stage: str) -> tuple[bool, str]:
    root = context.root
    total = context.expected_frames
    if stage == "ingest":
        if not context.frames_json.is_file():
            return False, "frames.json なし"
        count = len(context.frames)
        if count != total:
            return False, f"frames.json の枚数 {count} != {total}"
        pngs = list((root / "frames_rgb").glob("*.png"))
        return (len(pngs) == total, f"frames_rgb {len(pngs)}/{total}")
    if stage == "arrow":
        manifest_path = context.clips_dir / "manifest.json"
        if not manifest_path.is_file():
            return False, "clips manifest なし"
        manifest = models.load_json(manifest_path)
        done = [entry for entry in manifest["clips"] if entry.get("status") == "done"]
        if not done:
            return False, "done クリップなし"
        return True, f"{len(done)} clips"
    if stage == "align":
        return (context.alignments.is_file(), str(context.alignments))
    dirs = {
        "anchor": (root / "stages" / "anchor", total),
        "geometry": (root / "stages" / "geometry", total),
        "lingbot": (root / "stages" / "lingbot_fix", total),
        "prep": (root / "stages" / "prep", total),
        "ndr": (root / "stages" / "ndr", total),
        "sceneflow": (root / "stages" / "sceneflow_post", total),
        "sr": (root / "stages" / "sr", total),
        "temporal": (root / "stages" / "temporal", total),
        "final": (root / "stages" / "final", total),
    }
    if stage in dirs:
        directory, expected = dirs[stage]
        files = sorted(directory.glob("*.npz")) if directory.is_dir() else []
        ok = len(files) == expected
        extra_ok = True
        if stage == "sceneflow":
            extra_ok = (root / "stages" / "sceneflow" / "pairs.json").is_file()
        if stage == "sr":
            extra_ok = (root / "stages" / "sr.npz").is_file()
        return (ok and extra_ok, f"{len(files)}/{expected} npz")
    if stage == "render":
        report = root / "visuals" / "render_report.json"
        if not report.is_file():
            return False, "render_report.json なし"
        payload = models.load_json(report)
        mp4 = Path(payload.get("mp4", {}).get("path", ""))
        rrd = Path(payload.get("rrd", {}).get("path", ""))
        ok = mp4.is_file() and rrd.is_file()
        return (ok, f"mp4={mp4.is_file()} rrd={rrd.is_file()}")
    if stage == "moge":
        directory = root / "stages" / "moge"
        files = sorted(directory.glob("*.npz")) if directory.is_dir() else []
        return (len(files) == total, f"{len(files)}/{total} npz")
    return False, "unknown stage"


def resume_status(context: Context, stage: str, payload: dict) -> tuple[bool, str]:
    if context.args.force:
        return False, "force"
    path = context.signature_path(stage)
    if not path.is_file():
        return False, "signature なし"
    try:
        stored = models.load_json(path)
    except (OSError, ValueError):
        return False, "signature 読めない"
    if stored.get("signature_hash") != payload.get("signature_hash"):
        return False, "signature 不一致（入力/設定/コード変更）"
    present, detail = stage_outputs_present(context, stage)
    if not present:
        return False, f"出力不完全: {detail}"
    return True, f"resume skip ({detail})"


# --- attempts / subprocess ----------------------------------------------------


def run_attempt(context: Context, stage: str, argv: list, *, env_extra: dict | None = None,
                cwd: Path = REPO, timeout: float | None = None) -> tuple[Path, int]:
    attempt = context.attempt_dir(stage) / f"{stage}_{utc_stamp()}"
    attempt.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    additions = {}
    if env_extra:
        for key, value in env_extra.items():
            if key == "LD_LIBRARY_PATH_PREPEND":
                env["LD_LIBRARY_PATH"] = f"{value}:{env.get('LD_LIBRARY_PATH', '')}"
                additions["LD_LIBRARY_PATH_PREPEND"] = str(value)
            elif key == "PYTHONPATH":
                env["PYTHONPATH"] = f"{value}:{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else str(value)
                additions["PYTHONPATH"] = str(value)
            else:
                env[key] = str(value)
                additions[key] = str(value)
    models.atomic_write_json(
        attempt / "cmd.json",
        {
            "stage": stage,
            "dataset": context.dataset,
            "argv": [str(item) for item in argv],
            "cwd": str(cwd),
            "env_additions": additions,
            "started_at": utc_now(),
        },
    )
    started = time.perf_counter()
    timed_out = False
    signals_sent: list = []
    try:
        completed = subprocess.run(
            [str(item) for item in argv],
            cwd=str(cwd),
            env=env,
            capture_output=True,
            timeout=timeout,
        )
        code = int(completed.returncode)
        stdout, stderr = completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as error:
        timed_out = True
        code = 124
        stdout = error.stdout or b""
        stderr = error.stderr or b""
        signals_sent.append("SIGKILL(timeout)")
    wall = time.perf_counter() - started
    (attempt / "stdout.log").write_bytes(stdout if isinstance(stdout, bytes) else stdout.encode())
    (attempt / "stderr.log").write_bytes(stderr if isinstance(stderr, bytes) else stderr.encode())
    exit_payload = {
        "actual_exit_code": code,
        "ended_at": utc_now(),
        "wall_seconds": round(wall, 3),
        "measured_by": "run_depth_detail.py subprocess.run",
        "timed_out": timed_out,
        "signals_sent": signals_sent,
        "argv": [str(item) for item in argv],
    }
    models.atomic_write_json(attempt / "exit.json", exit_payload)
    tail = (stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else str(stderr))[-600:]
    print(f"  [{stage}] exit={code} wall={wall:.1f}s attempt={attempt.name}")
    if code != 0 and tail.strip():
        print(f"  [{stage}] stderr tail: {tail}")
    return attempt, code


# --- depth statistics for explicit configs ------------------------------------


def stage_depth_percentiles(context: Context, stage: str, key: str, valid_key: str,
                            *, valid_stage: str | None = None, sample: int = 12) -> dict:
    """depth は `stage[key]`、mask は `valid_stage[valid_key]`（既定は同一stage）から読む。

    欠stage/欠key/shape不一致/有効sample空は明瞭に SystemExit で拒否する。
    mask を捏造せず、正しい stage から読むこと（誤ったstageから読むのは設定ミス）。
    """
    valid_stage = valid_stage or stage
    frames = context.frames
    count = len(frames)
    step = max(1, count // sample)
    values = []
    checked = 0
    for record in frames[::step]:
        frame_index = int(record["frame_index"])
        depth_payload = models.load_frame_npz(context.root, stage, frame_index)
        if depth_payload is None:
            raise SystemExit(
                f"stage_depth_percentiles: depth stage '{stage}' が未生成です (frame {frame_index})"
            )
        if key not in depth_payload:
            raise SystemExit(
                f"stage_depth_percentiles: '{stage}' にキー '{key}' がありません "
                f"(keys={sorted(depth_payload)})"
            )
        depth = np.asarray(depth_payload[key], dtype=np.float64)
        if valid_stage == stage:
            valid_payload = depth_payload
        else:
            valid_payload = models.load_frame_npz(context.root, valid_stage, frame_index)
            if valid_payload is None:
                raise SystemExit(
                    f"stage_depth_percentiles: valid stage '{valid_stage}' が未生成です "
                    f"(frame {frame_index})"
                )
        if valid_key not in valid_payload:
            raise SystemExit(
                f"stage_depth_percentiles: '{valid_stage}' にキー '{valid_key}' がありません "
                f"(keys={sorted(valid_payload)})"
            )
        valid = np.asarray(valid_payload[valid_key], dtype=bool)
        if valid.shape != depth.shape:
            raise SystemExit(
                f"stage_depth_percentiles: shape不一致 depth {depth.shape} != valid {valid.shape} "
                f"({stage}.{key} vs {valid_stage}.{valid_key}, frame {frame_index})"
            )
        support = valid & np.isfinite(depth) & (depth > 0)
        if support.any():
            values.append(depth[::4, ::4][support[::4, ::4]])
        checked += 1
    if not values:
        raise SystemExit(
            f"depth 統計が取れません: {stage}.{key} (support={valid_stage}.{valid_key}, "
            f"frames_checked={checked})"
        )
    merged = np.concatenate(values)
    if merged.size == 0:
        raise SystemExit(
            f"depth 統計の有効sampleが空です: {stage}.{key} "
            f"(support={valid_stage}.{valid_key}; 支持が[::4,::4]サンプル位置に無い)"
        )
    return {
        "count": int(merged.size),
        "p1": float(np.percentile(merged, 1)),
        "median": float(np.median(merged)),
        "p99": float(np.percentile(merged, 99)),
        "p99_9": float(np.percentile(merged, 99.9)),
        "max": float(merged.max()),
    }


# --- stage runners ------------------------------------------------------------


def require_venv(path: Path, name: str) -> str:
    if not Path(path).is_file():
        raise SystemExit(f"{name} venv python が見つかりません: {path}")
    return str(path)


def common_worker_argv(context: Context) -> list:
    return [
        "--dataset-dir", str(context.root),
        "--dataset", context.dataset,
        "--frames", str(context.frames_json),
    ]


def step_ingest(context: Context) -> tuple[Path, int]:
    argv = [
        require_venv(ARROW_VENV_PYTHON, "arrow"),
        str(MODELS_SCRIPT), "ingest",
        "--dataset-dir", str(context.root),
        "--dataset", context.dataset,
        "--source", str(context.source),
        "--frames-json", str(context.frames_json),
    ]
    return run_attempt(context, "ingest", argv)


def arrow_env() -> dict:
    if CUDNN_LIB.is_dir():
        return {"LD_LIBRARY_PATH_PREPEND": str(CUDNN_LIB)}
    return {}


def resolve_arrow_device(args: argparse.Namespace):
    """ARROW stage の --device 実解決値（None/auto は省略 = process_sequence 既定の auto）。"""
    if args.arrow_device is not None:
        return args.arrow_device
    return args.device


def arrow_argv(context: Context) -> list:
    """ARROW stage の argv（step_arrow と plan_argv の共有生成器）。"""
    args = context.args
    device = resolve_arrow_device(args)
    argv = [
        require_venv(ARROW_VENV_PYTHON, "arrow"),
        str(SCRIPTS / "process_sequence.py"), str(context.source),
        "--output-dir", str(context.clips_dir),
        "--clip-frames", str(args.clip_frames),
        "--frame-stride", "1",
        "--clip-overlap", str(args.clip_overlap),
        "--max-side", str(args.max_side),
        "--tasks", "track", "reconstruct", "depth", "pose",
        "--track-stride", str(args.track_stride),
        "--reconstruction-stride", str(args.reconstruction_stride),
        "--depth-stride", str(args.depth_stride),
    ]
    if device is not None and str(device) != "auto":
        argv += ["--device", str(device)]
    return argv


def step_arrow(context: Context) -> tuple[Path, int]:
    argv = arrow_argv(context)
    return run_attempt(context, "arrow", argv, env_extra=arrow_env())


def step_align(context: Context) -> tuple[Path, int]:
    argv = [
        require_venv(ARROW_VENV_PYTHON, "arrow"),
        str(MODELS_SCRIPT), "align",
        *common_worker_argv(context),
        "--clips-dir", str(context.clips_dir),
    ]
    return run_attempt(context, "align", argv)


def step_anchor(context: Context) -> tuple[Path, int]:
    argv = [
        require_venv(ARROW_VENV_PYTHON, "arrow"),
        str(MODELS_SCRIPT), "anchor",
        *common_worker_argv(context),
        "--clips-dir", str(context.clips_dir),
        "--alignments", str(context.alignments),
    ]
    return run_attempt(context, "anchor", argv)


def step_geometry(context: Context) -> tuple[Path, int]:
    argv = [
        require_venv(ARROW_VENV_PYTHON, "arrow"),
        str(MODELS_SCRIPT), "geometry",
        *common_worker_argv(context),
        "--clips-dir", str(context.clips_dir),
        "--alignments", str(context.alignments),
    ]
    return run_attempt(context, "geometry", argv)


def step_lingbot(context: Context) -> tuple[Path, int]:
    args = context.args
    argv = [
        require_venv(LINGBOT_VENV_PYTHON, "lingbot-depth"),
        str(MODELS_SCRIPT), "lingbot",
        *common_worker_argv(context),
        "--model", args.lingbot_model,
        "--device", str(args.device),
        "--resolution-level", str(args.lingbot_resolution_level),
    ]
    env: dict = {}
    if LINGBOT_CUDNN_LIB.is_dir():
        env["LD_LIBRARY_PATH_PREPEND"] = str(LINGBOT_CUDNN_LIB)
    return run_attempt(context, "lingbot", argv, env_extra=env)


def step_lingbot_fix(context: Context) -> tuple[Path, int]:
    argv = [
        require_venv(ARROW_VENV_PYTHON, "arrow"),
        str(MODELS_SCRIPT), "lingbot_fix",
        *common_worker_argv(context),
        "--strength", "2.0",
        "--min-points", "32",
    ]
    return run_attempt(context, "lingbot_fix", argv)


def step_prep(context: Context) -> tuple[Path, int]:
    argv = [
        require_venv(ARROW_VENV_PYTHON, "arrow"),
        str(MODELS_SCRIPT), "prep",
        *common_worker_argv(context),
    ]
    return run_attempt(context, "prep", argv)


def stage_depth_global_stats(context: Context, stage: str, key: str, valid_key: str,
                             *, valid_stage: str | None = None) -> dict:
    """全frame走査（間引きなし）で depth の global min/max/median を返す。

    NDR の max bounds は sample 分位だけに頼らず本関数の global_max を使う（未sample外れ値対策）。
    欠stage/欠key/shape不一致/有効値空は明瞭に SystemExit。
    """
    valid_stage = valid_stage or stage
    values = []
    frames_checked = 0
    for record in context.frames:
        frame_index = int(record["frame_index"])
        depth_payload = models.load_frame_npz(context.root, stage, frame_index)
        if depth_payload is None:
            raise SystemExit(
                f"stage_depth_global_stats: depth stage '{stage}' が未生成です (frame {frame_index})"
            )
        if key not in depth_payload:
            raise SystemExit(
                f"stage_depth_global_stats: '{stage}' にキー '{key}' がありません "
                f"(keys={sorted(depth_payload)})"
            )
        depth = np.asarray(depth_payload[key], dtype=np.float64)
        if valid_stage == stage:
            valid_payload = depth_payload
        else:
            valid_payload = models.load_frame_npz(context.root, valid_stage, frame_index)
            if valid_payload is None:
                raise SystemExit(
                    f"stage_depth_global_stats: valid stage '{valid_stage}' が未生成です "
                    f"(frame {frame_index})"
                )
        if valid_key not in valid_payload:
            raise SystemExit(
                f"stage_depth_global_stats: '{valid_stage}' にキー '{valid_key}' がありません "
                f"(keys={sorted(valid_payload)})"
            )
        valid = np.asarray(valid_payload[valid_key], dtype=bool)
        if valid.shape != depth.shape:
            raise SystemExit(
                f"stage_depth_global_stats: shape不一致 depth {depth.shape} != valid {valid.shape} "
                f"({stage}.{key} vs {valid_stage}.{valid_key}, frame {frame_index})"
            )
        support = valid & np.isfinite(depth) & (depth > 0)
        if support.any():
            values.append(depth[support])
        frames_checked += 1
    if not values:
        raise SystemExit(
            f"stage_depth_global_stats: 有効値が空です: {stage}.{key} "
            f"(support={valid_stage}.{valid_key}, frames_checked={frames_checked})"
        )
    merged = np.concatenate(values)
    return {
        "count": int(merged.size),
        "min": float(merged.min()),
        "median": float(np.median(merged)),
        "max": float(merged.max()),
        "frames_checked": frames_checked,
    }


def ndr_depth_bounds(context: Context) -> tuple[float, float, str]:
    # valid mask は stage 'lingbot' 側、depth は stage 'lingbot_fix' 側に保存される。
    # max bounds は全frame走査の global_max から十分marginを取る（railway等の未sample外れ値対策）
    stats = stage_depth_global_stats(
        context, "lingbot_fix", "lingbot_depth", "lingbot_valid", valid_stage="lingbot"
    )
    max_depth = max(1e-3, stats["max"] * 2.0)
    return (
        1e-6,
        max_depth,
        f"full-scan max={stats['max']:.6g} * 2 (n={stats['count']}, frames={stats['frames_checked']})",
    )


def step_ndr(context: Context) -> tuple[Path, int]:
    args = context.args
    min_depth, max_depth, source = ndr_depth_bounds(context)
    argv = [
        require_venv(NDR_VENV_PYTHON, "normal_depth_refiner"),
        str(MODELS_SCRIPT), "ndr",
        *common_worker_argv(context),
        "--backend", str(args.ndr_backend),
        "--iterations", str(args.ndr_iterations),
        "--trust-abs", str(args.trust_abs),
        "--trust-rel", str(args.trust_rel),
        "--min-depth", str(min_depth),
        "--max-depth", str(max_depth),
        "--max-depth-source", source,
    ]
    if str(args.ndr_backend) == "numpy":
        argv.append("--allow-numpy")
    return run_attempt(context, "ndr", argv)


def sceneflow_input_scan(context: Context) -> dict:
    """sceneflow_prep が生成した全pairの実solver入力 (depth0_low/depth1_in_0) を全走査する。

    - 範囲決定の正本は prep 後の実入力（clamp/除外で値を変えない）。
    - 欠落/空は明瞭に SystemExit。median は motion 導出用（bounds とは別扱い）。
    """
    pairs_path = context.root / "stages" / "sceneflow" / "pairs.json"
    if not pairs_path.is_file():
        raise SystemExit(
            f"sceneflow prepared inputs がありません: {pairs_path}"
            "（先に --stage sceneflow の prep/refine を実行）"
        )
    pair_list = models.load_json(pairs_path)
    pairs = pair_list.get("pairs", [])
    if not pairs:
        raise SystemExit(f"sceneflow pair が空です: {pairs_path}")
    values = []
    for item in pairs:
        path = Path(item["file"])
        if not path.is_file():
            raise SystemExit(f"sceneflow pair input がありません: {path}")
        with np.load(path, allow_pickle=False) as data:
            for key in ("depth0_low", "depth1_in_0"):
                arr = np.asarray(data[key], dtype=np.float64)
                finite_pos = arr[np.isfinite(arr) & (arr > 0)]
                if finite_pos.size:
                    values.append(finite_pos)
    if not values:
        raise SystemExit(f"sceneflow: finite positive な solver 入力がありません: {pairs_path}")
    merged = np.concatenate(values)
    if merged.size == 0:
        raise SystemExit(f"sceneflow: solver 入力 scan が空です: {pairs_path}")
    return {
        "pair_count": len(pairs),
        "value_count": int(merged.size),
        "global_min": float(merged.min()),
        "global_max": float(merged.max()),
        "median": float(np.median(merged)),
    }


def sceneflow_config_values(context: Context, *, scan: dict | None = None
                            ) -> tuple[float, float, float, dict]:
    """全prepared inputs (実solver args) から bounds を導出する（sample分位は使わない）。"""
    scan = scan or sceneflow_input_scan(context)
    global_min = scan["global_min"]
    if global_min <= 1e-6:
        raise SystemExit(
            f"sceneflow: global_min={global_min:.6g} <= 1e-6 のため min_depth を設計できません（要設計）"
        )
    min_depth = max(1e-6, global_min * 0.5)
    max_depth = scan["global_max"] * 2.0
    if max_depth <= min_depth:
        raise SystemExit(
            f"sceneflow: global range が縮退しています (min={global_min:.6g}, max={scan['global_max']:.6g})"
        )
    max_motion = max(1e-6, context.args.sceneflow_max_motion_frac * scan["median"])
    # C++ solver 制約: max_motion < min_depth
    max_motion = min(max_motion, 0.5 * min_depth)
    return min_depth, max_depth, max_motion, scan


def step_sceneflow_prep(context: Context) -> tuple[Path, int]:
    argv = [
        require_venv(ARROW_VENV_PYTHON, "arrow"),
        str(MODELS_SCRIPT), "sceneflow_prep",
        *common_worker_argv(context),
        "--clips-dir", str(context.clips_dir),
        "--long-side", str(context.args.long_side),
    ]
    return run_attempt(context, "sceneflow_prep", argv)


def step_sceneflow_refine(context: Context) -> tuple[Path, int]:
    min_depth, max_depth, max_motion, scan = sceneflow_config_values(context)
    workers = max(1, int(context.args.sceneflow_workers))
    argv = [
        require_venv(SCENEFLOW_VENV_PYTHON, "rgbd_sceneflow"),
        str(MODELS_SCRIPT), "sceneflow_refine",
        *common_worker_argv(context),
        "--min-depth", repr(min_depth),
        "--max-depth", repr(max_depth),
        "--max-motion", repr(max_motion),
        "--trust-sigma", "4.0",
        "--workers", str(workers),
    ]
    # resolved 値と全scan結果を config と receipt(cmd.json) に保存
    models.atomic_write_json(
        context.config_dir() / "sceneflow_bounds.json",
        {
            "dataset": context.dataset,
            "policy": "scan_all_prepared_inputs (全pair実solver入力全域; sample分位不使用)",
            "min_depth": min_depth, "max_depth": max_depth, "max_motion": max_motion,
            "motion_frac": context.args.sceneflow_max_motion_frac,
            "scan": scan,
            "workers": workers,
            "argv": argv,
            "resolved_at": utc_now(),
        },
    )
    return run_attempt(
        context, "sceneflow_refine", argv,
        env_extra={"PYTHONPATH": str(SCENEFLOW_PYTHONPATH)},
    )


def step_sceneflow_post(context: Context) -> tuple[Path, int]:
    argv = [
        require_venv(ARROW_VENV_PYTHON, "arrow"),
        str(MODELS_SCRIPT), "sceneflow_post",
        *common_worker_argv(context),
        "--trust-abs", str(context.args.trust_abs),
        "--trust-rel", str(context.args.trust_rel),
    ]
    return run_attempt(context, "sceneflow_post", argv)


def step_sr(context: Context) -> tuple[Path, int]:
    argv = [
        require_venv(ARROW_VENV_PYTHON, "arrow"),
        str(MODELS_SCRIPT), "sr",
        *common_worker_argv(context),
    ]
    return run_attempt(context, "sr", argv)


def step_temporal(context: Context) -> tuple[Path, int]:
    args = context.args
    argv = [
        require_venv(ARROW_VENV_PYTHON, "arrow"),
        str(MODELS_SCRIPT), "temporal",
        *common_worker_argv(context),
        "--weight", str(args.fusion_weight),
        "--trust-abs", str(args.trust_abs),
        "--trust-rel", str(args.trust_rel),
        "--gate-tolerance", "0.05",
        "--edge-threshold", "0.5",
    ]
    return run_attempt(context, "temporal", argv)


def step_final(context: Context) -> tuple[Path, int]:
    argv = [
        require_venv(ARROW_VENV_PYTHON, "arrow"),
        str(MODELS_SCRIPT), "final",
        *common_worker_argv(context),
        "--range-low", "2.0",
        "--range-high", "98.0",
    ]
    return run_attempt(context, "final", argv)


def step_render(context: Context) -> tuple[Path, int]:
    args = context.args
    fps = args.video_fps if args.video_fps is not None else context.video_fps
    argv = [
        require_venv(ARROW_VENV_PYTHON, "arrow"),
        str(MODELS_SCRIPT), "render",
        *common_worker_argv(context),
        "--fps", repr(float(fps)),
        "--cell-width", str(args.cell_width),
        "--point-stride", str(args.point_stride),
    ]
    return run_attempt(context, "render", argv)


def step_moge(context: Context) -> tuple[Path, int]:
    args = context.args
    argv = [
        require_venv(MOGE_VENV_PYTHON, "MoGe"),
        str(MODELS_SCRIPT), "moge",
        *common_worker_argv(context),
        "--model", args.moge_model,
        "--device", str(args.device),
        "--num-tokens", str(args.moge_num_tokens),
        "--resolution-level", "9",
        "--refine-steps", str(args.moge_refine_steps),
    ]
    return run_attempt(context, "moge", argv)


GROUPS: dict[str, list[tuple[str, Callable[[Context], tuple[Path, int]]]]] = {
    "ingest": [("ingest", step_ingest)],
    "arrow": [("arrow", step_arrow)],
    "align": [("align", step_align)],
    "anchor": [("anchor", step_anchor)],
    "geometry": [("geometry", step_geometry)],
    "lingbot": [("lingbot", step_lingbot), ("lingbot_fix", step_lingbot_fix)],
    "prep": [("prep", step_prep)],
    "ndr": [("ndr", step_ndr)],
    "sceneflow": [
        ("sceneflow_prep", step_sceneflow_prep),
        ("sceneflow_refine", step_sceneflow_refine),
        ("sceneflow_post", step_sceneflow_post),
    ],
    "sr": [("sr", step_sr)],
    "temporal": [("temporal", step_temporal)],
    "final": [("final", step_final)],
    "render": [("render", step_render)],
    "moge": [("moge", step_moge)],
}

PRIMARY_STEP = {
    "lingbot": "lingbot_fix",
    "sceneflow": "sceneflow_refine",
}

PREREQUISITES = {
    "ingest": [],
    "arrow": ["ingest"],
    "align": ["arrow"],
    "anchor": ["arrow", "align"],
    "geometry": ["arrow", "align"],
    "lingbot": ["anchor", "geometry"],
    "prep": ["lingbot", "geometry"],
    "ndr": ["lingbot", "prep", "geometry"],
    "sceneflow": ["ndr"],
    "sr": ["sceneflow"],
    "temporal": ["sceneflow", "sr", "prep", "anchor"],
    "final": ["temporal"],
    "render": ["final"],
    "moge": ["ingest"],
}


# --- signatures per group ------------------------------------------------------


def group_signature(context: Context, group: str) -> dict:
    args = context.args
    chain = previous_chain(context, group)
    frames_sha = models.sha256_file(context.frames_json) if context.frames_json.is_file() else ""
    clips_manifest = context.clips_dir / "manifest.json"
    clips_sha = models.sha256_file(clips_manifest) if clips_manifest.is_file() else ""
    alignments_sha = models.sha256_file(context.alignments) if context.alignments.is_file() else ""
    if group == "ingest":
        fingerprint = source_fingerprint(context)
        return compute_signature(
            context, group,
            inputs=[{"path": str(context.source), "sha256": fingerprint["digest"]}],
            params={"dataset": context.dataset, "source_kind": fingerprint["kind"]},
            model="arrow ingest (PNG materialization)", argv=plan_argv(context, "ingest"),
        )
    if group == "arrow":
        fingerprint = source_fingerprint(context)
        arrow_device = resolve_arrow_device(args)
        return compute_signature(
            context, group,
            inputs=[{"path": str(context.source), "sha256": fingerprint["digest"]}],
            params={
                "clip_frames": args.clip_frames, "clip_overlap": args.clip_overlap,
                "max_side": args.max_side, "tasks": ["track", "reconstruct", "depth", "pose"],
                "track_stride": args.track_stride,
                "reconstruction_stride": args.reconstruction_stride,
                "depth_stride": args.depth_stride, "device": args.device,
                "arrow_device_resolved": arrow_device,
                "arrow_device_flag": (
                    None if (arrow_device is None or str(arrow_device) == "auto")
                    else str(arrow_device)
                ),
                "arrow_argv": plan_argv(context, "arrow"),
            },
            model="ifradlin/arrow", argv=plan_argv(context, "arrow"),
        )
    if group == "align":
        return compute_signature(
            context, group,
            inputs=[{"path": str(clips_manifest), "sha256": clips_sha}],
            params={"adapter": "inv(P0) ref0 cameras + estimate_chain"},
            model="align_clip_scenes adapter", argv=plan_argv(context, "align"),
        )
    if group in ("anchor", "geometry"):
        return compute_signature(
            context, group,
            inputs=[{"path": str(clips_manifest), "sha256": clips_sha},
                    {"path": str(context.alignments), "sha256": alignments_sha}],
            params={"depth_reference": "source", "pose_reference": "middle",
                    "reconstruct_reference": 0},
            model="ifradlin/arrow", argv=plan_argv(context, group),
        )
    if group == "lingbot":
        return compute_signature(
            context, group,
            inputs=[{"path": str(context.frames_json), "sha256": frames_sha}],
            params={"model": args.lingbot_model, "resolution_level": args.lingbot_resolution_level,
                    "device": args.device, "reanchor": "fit_scale + regularize_scales(strength=2.0)"},
            model=args.lingbot_model, argv=plan_argv(context, "lingbot"), chain=chain,
        )
    if group == "prep":
        return compute_signature(
            context, group,
            inputs=[{"path": str(context.frames_json), "sha256": frames_sha}],
            params={"edges": "core.gray_edges", "normals": "depth-derived 3D local differences"},
            model="arrow.depth_detail.gray_edges", argv=plan_argv(context, "prep"), chain=chain,
        )
    if group == "ndr":
        try:
            min_depth, max_depth, source = ndr_depth_bounds(context)
        except SystemExit:
            min_depth, max_depth, source = 1e-6, 1000.0, "fallback (data unavailable)"
        return compute_signature(
            context, group,
            inputs=[],
            params={"backend": args.ndr_backend, "iterations": args.ndr_iterations,
                    "trust_abs": args.trust_abs, "trust_rel": args.trust_rel,
                    "min_depth": min_depth, "max_depth": max_depth, "max_depth_source": source},
            model="normal_depth_refiner", argv=plan_argv(context, "ndr"), chain=chain,
        )
    if group == "sceneflow":
        # 署名は「prep後の全実入力scanで解く」という policy を hash する（resolved値はruntimeでconfigs/receiptへ）。
        # prep前でも plan/policy が通るよう、ここでデータを読まない。
        return compute_signature(
            context, group,
            inputs=[{"path": str(clips_manifest), "sha256": clips_sha},
                    {"path": str(context.frames_json), "sha256": frames_sha}],
            params={
                "long_side": args.long_side,
                "bounds_policy": "scan_all_prepared_inputs (全pair実solver入力全域; sample分位不使用)",
                "min_depth_policy": "max(1e-6, global_min*0.5)",
                "max_depth_policy": "global_max*2.0",
                "max_motion_policy": "min(motion_frac*median, 0.5*min_depth)",
                "motion_frac": args.sceneflow_max_motion_frac,
                "trust_sigma": 4.0,
                "trust_abs": args.trust_abs,
                "trust_rel": args.trust_rel,
                "workers": max(1, int(args.sceneflow_workers)),
            },
            model="scene_flow_refinement + core gates", argv=plan_argv(context, "sceneflow"), chain=chain,
        )
    if group == "sr":
        return compute_signature(
            context, group,
            inputs=[{"path": str(context.frames_json), "sha256": frames_sha}],
            params={"api": "superresolve_scene_flow", "no_resolution_scaling": True},
            model="arrow.depth_detail.superresolve_scene_flow", argv=plan_argv(context, "sr"), chain=chain,
        )
    if group == "temporal":
        return compute_signature(
            context, group,
            inputs=[],
            params={"weight": args.fusion_weight, "trust_abs": args.trust_abs,
                    "trust_rel": args.trust_rel, "gate_tolerance": 0.05,
                    "edge_threshold": 0.5},
            model="arrow.depth_detail.temporal_fuse", argv=plan_argv(context, "temporal"), chain=chain,
        )
    if group == "final":
        return compute_signature(
            context, group,
            inputs=[{"path": str(context.frames_json), "sha256": frames_sha}],
            params={"range_low": 2.0, "range_high": 98.0},
            model="final aggregation", argv=plan_argv(context, "final"), chain=chain,
        )
    if group == "render":
        fps = args.video_fps if args.video_fps is not None else context.video_fps
        return compute_signature(
            context, group,
            inputs=[{"path": str(context.frames_json), "sha256": frames_sha}],
            params={"fps": float(fps), "cell_width": args.cell_width,
                    "point_stride": args.point_stride,
                    "common_range": True},
            model="render", argv=plan_argv(context, "render"), chain=chain,
        )
    if group == "moge":
        return compute_signature(
            context, group,
            inputs=[{"path": str(context.frames_json), "sha256": frames_sha}],
            params={"model": args.moge_model, "num_tokens": args.moge_num_tokens,
                    "refine_steps": args.moge_refine_steps, "role": "optional_comparison_only"},
            model=args.moge_model, argv=plan_argv(context, "moge"), chain=chain,
        )
    raise SystemExit(f"unknown group: {group}")


def plan_argv(context: Context, group: str) -> list:
    """dry-run/署名用に argv を組み立てる（実行しない）。"""
    label_steps = GROUPS[group]
    argv: list = []
    for label, _ in label_steps:
        if label == "ingest":
            argv += [str(ARROW_VENV_PYTHON), str(MODELS_SCRIPT), "ingest",
                     "--dataset-dir", str(context.root), "--dataset", context.dataset,
                     "--source", str(context.source), "--frames-json", str(context.frames_json)]
        elif label == "arrow":
            argv += arrow_argv(context)
        elif label == "align":
            argv += [str(ARROW_VENV_PYTHON), str(MODELS_SCRIPT), "align",
                     *common_worker_argv(context), "--clips-dir", str(context.clips_dir)]
        elif label in ("anchor", "geometry"):
            argv += [str(ARROW_VENV_PYTHON), str(MODELS_SCRIPT), label,
                     *common_worker_argv(context), "--clips-dir", str(context.clips_dir),
                     "--alignments", str(context.alignments)]
        elif label == "lingbot":
            argv += [str(LINGBOT_VENV_PYTHON), str(MODELS_SCRIPT), "lingbot",
                     *common_worker_argv(context), "--model", context.args.lingbot_model,
                     "--device", str(context.args.device),
                     "--resolution-level", str(context.args.lingbot_resolution_level)]
        elif label == "lingbot_fix":
            argv += [str(ARROW_VENV_PYTHON), str(MODELS_SCRIPT), "lingbot_fix",
                     *common_worker_argv(context), "--strength", "2.0", "--min-points", "32"]
        elif label == "prep":
            argv += [str(ARROW_VENV_PYTHON), str(MODELS_SCRIPT), "prep", *common_worker_argv(context)]
        elif label == "ndr":
            argv += [str(NDR_VENV_PYTHON), str(MODELS_SCRIPT), "ndr", *common_worker_argv(context),
                     "--backend", str(context.args.ndr_backend),
                     "--iterations", str(context.args.ndr_iterations),
                     "--trust-abs", str(context.args.trust_abs),
                     "--trust-rel", str(context.args.trust_rel),
                     "--min-depth", "<runtime p1*0.5>", "--max-depth", "<runtime p99.9*2>"]
        elif label == "sceneflow_prep":
            argv += [str(ARROW_VENV_PYTHON), str(MODELS_SCRIPT), "sceneflow_prep",
                     *common_worker_argv(context), "--clips-dir", str(context.clips_dir),
                     "--long-side", str(context.args.long_side)]
        elif label == "sceneflow_refine":
            argv += [str(SCENEFLOW_VENV_PYTHON), str(MODELS_SCRIPT), "sceneflow_refine",
                     *common_worker_argv(context),
                     "--min-depth", "<runtime scan_all_prepared_inputs: global_min*0.5>",
                     "--max-depth", "<runtime scan_all_prepared_inputs: global_max*2>",
                     "--max-motion", "<runtime motion_frac*median clamp(0.5*min_depth)>",
                     "--trust-sigma", "4.0",
                     "--workers", str(max(1, int(context.args.sceneflow_workers)))]
        elif label == "sceneflow_post":
            argv += [str(ARROW_VENV_PYTHON), str(MODELS_SCRIPT), "sceneflow_post",
                     *common_worker_argv(context), "--trust-abs", str(context.args.trust_abs),
                     "--trust-rel", str(context.args.trust_rel)]
        elif label == "sr":
            argv += [str(ARROW_VENV_PYTHON), str(MODELS_SCRIPT), "sr", *common_worker_argv(context)]
        elif label == "temporal":
            argv += [str(ARROW_VENV_PYTHON), str(MODELS_SCRIPT), "temporal", *common_worker_argv(context),
                     "--weight", str(context.args.fusion_weight),
                     "--trust-abs", str(context.args.trust_abs),
                     "--trust-rel", str(context.args.trust_rel),
                     "--gate-tolerance", "0.05", "--edge-threshold", "0.5"]
        elif label == "final":
            argv += [str(ARROW_VENV_PYTHON), str(MODELS_SCRIPT), "final", *common_worker_argv(context),
                     "--range-low", "2.0", "--range-high", "98.0"]
        elif label == "render":
            fps = context.args.video_fps if context.args.video_fps is not None else context.video_fps
            argv += [str(ARROW_VENV_PYTHON), str(MODELS_SCRIPT), "render", *common_worker_argv(context),
                     "--fps", repr(float(fps)), "--cell-width", str(context.args.cell_width),
                     "--point-stride", str(context.args.point_stride)]
        elif label == "moge":
            argv += [str(MOGE_VENV_PYTHON), str(MODELS_SCRIPT), "moge", *common_worker_argv(context),
                     "--model", context.args.moge_model, "--device", str(context.args.device),
                     "--num-tokens", str(context.args.moge_num_tokens),
                     "--resolution-level", "9",
                     "--refine-steps", str(context.args.moge_refine_steps)]
    return argv


# --- manifest / events / main --------------------------------------------------


def record_event(context: Context, stage: str, action: str, detail: dict | None = None) -> None:
    path = context.root / "logs" / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"utc": utc_now(), "dataset": context.dataset, "stage": stage, "action": action}
    if detail:
        payload.update(detail)
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_receipts(context: Context) -> dict:
    path = context.root / "configs" / "receipts.json"
    try:
        return models.load_json(path)
    except (OSError, ValueError):
        return {}


def save_receipts(context: Context, receipts: dict) -> None:
    models.atomic_write_json(context.root / "configs" / "receipts.json", receipts)


def cli_config_snapshot(args: argparse.Namespace) -> dict:
    snapshot = {}
    for key, value in sorted(vars(args).items()):
        if key in ("dry_run", "force"):
            continue
        snapshot[key] = str(value) if isinstance(value, Path) else value
    return snapshot


def update_manifest(context: Context) -> dict | None:
    root = context.root
    if not context.frames_json.is_file():
        return None
    frames_payload = models.load_json(context.frames_json)
    frames = frames_payload["frames"]
    configs = {}
    config_dir = context.config_dir()
    if config_dir.is_dir():
        for path in sorted(config_dir.glob("*.json")):
            try:
                configs[path.stem] = models.load_json(path)
            except (OSError, ValueError):
                continue
    receipts = load_receipts(context)
    stage_receipts = {
        group: record.get("primary") for group, record in receipts.items() if record.get("primary")
    }
    stage_signatures = {}
    for stage in list(STAGE_ORDER) + OPTIONAL_STAGES:
        path = context.signature_path(stage)
        if path.is_file():
            stage_signatures[stage] = str(path)
    stages = [stage for stage in STAGE_ORDER if stage_outputs_present(context, stage)[0]]
    for optional in OPTIONAL_STAGES:
        if stage_outputs_present(context, optional)[0]:
            stages.append(optional)
    config_hash = models.sha256_bytes(
        models.canonical_json(
            {
                "cli": cli_config_snapshot(context.args),
                "stage_configs": {
                    name: models.sha256_bytes(models.canonical_json(value).encode("utf-8"))
                    for name, value in configs.items()
                },
            }
        ).encode("utf-8")
    )
    temporal_config = configs.get("temporal", {})
    sr_config = configs.get("sr", {})
    render_config = configs.get("render", {})
    render_report = {}
    report_path = root / "visuals" / "render_report.json"
    if report_path.is_file():
        try:
            render_report = models.load_json(report_path)
        except (OSError, ValueError):
            render_report = {}
    visuals = {}
    if render_config:
        for key in ("mp4", "rrd", "png"):
            value = render_config.get(key)
            if isinstance(value, str) and value:
                try:
                    visuals[key] = str(Path(value).relative_to(root))
                except ValueError:
                    visuals[key] = value
    manifest = {
        "nonce": NONCE,
        "dataset": context.dataset,
        "expected_frame_count": context.expected_frames,
        "original_size": frames_payload.get("original_size"),
        "units": UNITS,
        "metric_calibrated": False,
        "calibration": {
            "single_scalar": 1.0,
            "applied": False,
            "targets": ["depth", "camera_translation", "scene_flow_3d"],
            "note": "後段で単一 scalar 乗算により全 depth/translation/3D flow を一括校正できる",
        },
        "stages": stages,
        "optional_stages": OPTIONAL_STAGES,
        "models": {
            "arrow": "ifradlin/arrow",
            "lingbot": context.args.lingbot_model,
            "ndr": "normal_depth_refiner",
            "sceneflow": "rgbd_sceneflow/scene_flow_refinement",
            "sr": "arrow.depth_detail.superresolve_scene_flow",
            "temporal": "arrow.depth_detail.temporal_fuse",
            "moge": context.args.moge_model if "moge" in stages else None,
        },
        "config_hash": config_hash,
        "frames": [
            {
                "frame_index": int(record["frame_index"]),
                "source_index": int(record["source_index"]),
                "relative_path": record["relative_path"],
                "rgb_path": record.get("rgb_path"),
                "rgb_sha256": record["rgb_sha256"],
                "rgb_sha256_source": record.get("rgb_sha256_source"),
                "height": int(record["height"]),
                "width": int(record["width"]),
                "timestamp": record.get("timestamp"),
            }
            for record in frames
        ],
        "stage_receipts": stage_receipts,
        "stage_signatures": stage_signatures,
        "stage_configs": configs,
        "temporal": {
            key: temporal_config.get(key)
            for key in (
                "pairs_compared", "denom_total", "error_before", "error_after",
                "worsened_frames", "rejected_frames", "failed_frames",
                "edge_leakage_median", "anchor_deviation_median",
                "anchor_deviation_before_median",
                "candidate_source", "sr_pairs_expected", "sr_pairs_consumed",
                "sr_pairs_unconsumed",
            )
        },
        "sr": {
            key: sr_config.get(key)
            for key in ("pairs", "bilinear_vs_sr", "low_consistency_abs_diff", "valid_fraction", "gt_gate")
        },
        "visuals": visuals,
        "render_report": render_report if render_report else None,
        "stage_chain": STAGE_CHAIN,
        "stage_chain_full": STAGE_CHAIN_FULL,
        "updated_at": utc_now(),
    }
    models.atomic_write_json(root / "manifest.json", manifest)
    return manifest


def print_plan(contexts: list[Context], args: argparse.Namespace, selected: list[str]) -> None:
    print(f"dry-run: dataset={args.dataset} stage={args.stage} output-dir={args.output_dir}")
    print(f"  選択stage: {', '.join(selected)}")
    print(f"  単位: {UNITS} (metric_calibrated=false, 後段scalar校正用)")
    print(f"  GPU: 本計画は実行しない（実行argvのみ表示）。device={args.device}")
    print(f"  venv: arrow={ARROW_VENV_PYTHON}")
    print(f"        lingbot={LINGBOT_VENV_PYTHON}")
    resolved_arrow_device = resolve_arrow_device(args)
    if resolved_arrow_device is None or str(resolved_arrow_device) == "auto":
        print("  arrow device: auto（--device を省略; process_sequence 既定 = prefill signature device=null と一致）")
    else:
        print(f"  arrow device: {resolved_arrow_device}（--device を明示）")
    print(f"        ndr={NDR_VENV_PYTHON} (backend={args.ndr_backend}, iterations={args.ndr_iterations})")
    print(f"        sceneflow={SCENEFLOW_VENV_PYTHON} (PYTHONPATH={SCENEFLOW_PYTHONPATH}, "
          f"workers={max(1, int(args.sceneflow_workers))})")
    if CUDNN_LIB.is_dir():
        print(f"  arrow GPU env: LD_LIBRARY_PATH_PREPEND={CUDNN_LIB}")
    if LINGBOT_CUDNN_LIB.is_dir():
        print(f"  lingbot GPU env: LD_LIBRARY_PATH_PREPEND={LINGBOT_CUDNN_LIB} "
              f"(ARROWのcu130 libは流用しない)")
    for context in contexts:
        try:
            enumerated = models.enumerate_source(context.source, decode_frames=False)
            count = len(enumerated)
        except SystemExit:
            count = None
        print(f"\n[{context.dataset}] source={context.source}")
        print(f"  expected_frames={context.expected_frames} enumerated={count} "
              f"video_fps={context.video_fps}")
        print(f"  root={context.root}")
        for group in selected:
            print(f"  stage {group}: prereq={PREREQUISITES.get(group, [])}")
            for line in _format_argv(plan_argv(context, group)):
                print(f"    {line}")
        print("  resume: signatures/<stage>.json の signature_hash と出力存在を確認し、"
              "一致時は再実行しない（--force で無視）")
    print("\ndry-run: ファイル生成・推論は行っていません")


def _format_argv(argv: list) -> list[str]:
    lines = []
    current = ""
    for token in argv:
        token = str(token)
        if current and len(current) + len(token) + 1 > 110:
            lines.append(current)
            current = token
        else:
            current = f"{current} {token}".strip()
    if current:
        lines.append(current)
    return lines


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir = args.output_dir.expanduser().resolve()
    contexts = dataset_contexts(args, require_frames=False)
    selected = list(STAGE_ORDER) + (["moge"] if args.with_moge else [])
    if args.stage not in ("all",):
        selected = [args.stage]
    if args.dry_run:
        print_plan(contexts, args, selected)
        return 0

    for context in contexts:
        context.root.mkdir(parents=True, exist_ok=True)
        models.atomic_write_json(
            context.root / "configs" / "run_invocation.json",
            {
                "nonce": NONCE,
                "dataset": context.dataset,
                "argv": [str(token) for token in sys.argv],
                "cwd": str(Path.cwd()),
                "env_additions": {
                    "arrow": arrow_env(),
                    "sceneflow_pythonpath": str(SCENEFLOW_PYTHONPATH),
                },
                "created_at": utc_now(),
            },
        )
    for context in contexts:
        if "ingest" not in selected and not context.frames_json.is_file():
            raise SystemExit(
                f"frames.json がありません: {context.frames_json}（先に --stage ingest か --stage all を実行）"
            )
        for group in selected:
            missing = [
                prerequisite for prerequisite in PREREQUISITES.get(group, [])
                if not stage_outputs_present(context, prerequisite)[0]
            ]
            if missing:
                raise SystemExit(
                    f"[{context.dataset}] stage {group} には先に {missing} が必要です"
                )
            payload = group_signature(context, group)
            skip, reason = resume_status(context, group, payload)
            if skip:
                print(f"[{context.dataset}] stage {group}: {reason}")
                record_event(context, group, "skip", {"reason": reason})
                continue
            print(f"[{context.dataset}] stage {group}: 実行（{reason}）")
            receipts = load_receipts(context)
            record = {"primary": None, "steps": {}, "started_at": utc_now()}
            failed = False
            for label, step in GROUPS[group]:
                attempt, code = step(context)
                record["steps"][label] = str(attempt)
                if label == PRIMARY_STEP.get(group, label):
                    record["primary"] = str(attempt)
                record_event(context, group, "step", {"step": label, "exit_code": code,
                                                      "attempt": attempt.name})
                if code != 0:
                    failed = True
                    break
            if record["primary"] is None and record["steps"]:
                record["primary"] = list(record["steps"].values())[-1]
            record["finished_at"] = utc_now()
            receipts[group] = record
            save_receipts(context, receipts)
            if failed:
                record_event(context, group, "fail")
                update_manifest(context)
                print(f"[{context.dataset}] stage {group}: 失敗（attempts と manifest を確認）")
                return 1
            write_signature(context, payload)
            record_event(context, group, "done", {"signature_hash": payload["signature_hash"]})
            update_manifest(context)
    for context in contexts:
        update_manifest(context)
    print("完了: " + ", ".join(f"{context.dataset}({context.root})" for context in contexts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
