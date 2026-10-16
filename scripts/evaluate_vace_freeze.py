#!/usr/bin/env python3
"""Compare source/draft/raw VACE/composite RGB and independently audit provenance.

Numeric checks require NumPy; media export reuses evaluate_grass_freeze.py and
its PIL/cv2/ffmpeg dependencies. No quality acceptance follows from an exact
compositing check. Pixel equality applies to raw PNGs, not the lossy MP4.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mask_array(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape:
        raise ValueError(f"{name} shape {value.shape} != {shape}")
    if value.dtype == np.bool_:
        return value.copy()
    if value.dtype == np.uint8 and np.isin(value, [0, 255]).all():
        return value == 255
    raise ValueError(f"{name} must be bool or uint8 containing only 0/255")


def region_masks(edit, optional):
    """Use explicit provenance only; identical colors never establish provenance."""
    regions = {"unknown_generation": edit.copy()}
    assigned = edit.copy()
    for name, key in (("source_preserved", "source_preserved_mask"),
                      ("warped_known", "warped_known_mask"),
                      ("reference_projected", "reference_projected_mask")):
        if key not in optional:
            regions[name] = None
            continue
        mask = mask_array(optional[key], edit.shape, key)
        if np.any(mask & assigned):
            raise ValueError(f"{key} overlaps unknown or another provenance region")
        regions[name] = mask
        assigned |= mask
    if all(key in optional for key in ("source_preserved_mask", "warped_known_mask",
                                      "reference_projected_mask")) and not assigned.all():
        raise ValueError("Explicit source/warped/reference/edit masks do not form a full partition")
    regions["known_unclassified"] = ~assigned
    if "known_mask" in optional:
        known = mask_array(optional["known_mask"], edit.shape, "known_mask")
        if not np.array_equal(known, ~edit):
            raise ValueError("known_mask != ~edit_mask")
    return regions


def equality(a, b, mask):
    count = int(mask.sum())
    if not count:
        return {"status": "not_applicable", "checked_pixels": 0, "mismatched_pixels": 0}
    mismatched = int(np.any(a[mask] != b[mask], axis=-1).sum())
    return {"status": "verified" if not mismatched else "failed",
            "checked_pixels": count, "mismatched_pixels": mismatched}


def masked_mae(a, b, mask):
    if not mask.any():
        return None
    return float(np.mean(np.abs(a[mask].astype(np.float64) - b[mask].astype(np.float64))))


def analyze_pixels(source, draft, raw, composite, edit, optional=None, reference=None):
    """Return evidence, including failed checks; callers decide the process exit."""
    arrays = {"source": source, "draft": draft, "raw": raw, "composite": composite}
    for name, array in arrays.items():
        if array.dtype != np.uint8 or array.shape != source.shape:
            raise ValueError(f"{name} must be uint8 with source shape")
    if source.ndim != 4 or source.shape[-1] != 3 or min(source.shape[:3]) < 1:
        raise ValueError("RGB arrays must be nonempty [T,H,W,3]")
    edit = mask_array(edit, source.shape[:3], "edit_mask")
    regions = region_masks(edit, optional or {})
    integrity = {
        "composite_known_equals_draft": equality(composite, draft, ~edit),
        "composite_unknown_equals_raw": equality(composite, raw, edit),
        "draft_unknown_equals_gray127": equality(draft, np.full_like(draft, 127), edit),
    }
    source_mask = regions["source_preserved"]
    integrity["explicit_source_preserved_equals_source"] = (
        equality(composite, source, source_mask) if source_mask is not None
        else {"status": "not_available", "reason": "No explicit source_preserved_mask supplied"})
    integrity["explicit_source_draft_equals_source"] = (
        equality(draft, source, source_mask) if source_mask is not None
        else {"status": "not_available", "reason": "No explicit source_preserved_mask supplied"})
    if not edit[0].any():
        integrity["frame0_no_edit_equals_source"] = equality(
            composite[0], source[0], np.ones(edit.shape[1:], bool))
    else:
        integrity["frame0_no_edit_equals_source"] = {
            "status": "not_applicable", "reason": "frame0 has edit pixels"}
    if reference is not None:
        if reference.dtype != np.uint8 or reference.shape != source.shape[1:]:
            raise ValueError("reference_rgb must have source frame shape and uint8 dtype")
        integrity["reference_equals_source_frame0"] = equality(
            reference, source[0], np.ones(edit.shape[1:], bool))
    region_scores = {}
    for name, mask in regions.items():
        if mask is None:
            region_scores[name] = {"status": "not_available", "pixel_count": None,
                                   "reason": "Explicit provenance mask not supplied"}
            continue
        scores = {"status": "diagnostic_only", "pixel_count": int(mask.sum()),
                  "fraction": float(mask.mean()), "per_frame": []}
        for candidate in ("draft", "raw", "composite"):
            scores[f"source_to_{candidate}_mae_255"] = masked_mae(source, arrays[candidate], mask)
        scores["draft_to_composite_mae_255"] = masked_mae(draft, composite, mask)
        scores["raw_to_composite_mae_255"] = masked_mae(raw, composite, mask)
        for index in range(len(source)):
            frame_mask = mask[index]
            scores["per_frame"].append({"frame": index, "pixel_count": int(frame_mask.sum()),
                "source_to_raw_mae_255": masked_mae(source[index], raw[index], frame_mask),
                "source_to_composite_mae_255": masked_mae(source[index], composite[index], frame_mask)})
        region_scores[name] = scores
    return {"integrity": integrity,
            "integrity_failures": [name for name, value in integrity.items() if value["status"] == "failed"],
            "regions": region_scores}, regions


def media_module():
    path = Path(__file__).with_name("evaluate_grass_freeze.py")
    spec = importlib.util.spec_from_file_location("grass_freeze_media", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import media helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def load_controls(path):
    with np.load(path, allow_pickle=False) as data:
        required = ("source_rgb", "draft_rgb", "reference_rgb", "edit_mask", "fps")
        for key in required:
            if key not in data:
                raise ValueError(f"Missing control {key}")
        arrays = {key: data[key].copy() for key in required}
        optional = {key: data[key].copy() for key in
                    ("source_preserved_mask", "warped_known_mask", "reference_projected_mask",
                     "known_mask") if key in data}
        available_keys = list(data.files)
    fps_array = arrays.pop("fps")
    if fps_array.shape != () or not np.isfinite(fps_array) or float(fps_array) <= 0:
        raise ValueError("fps must be a positive finite scalar")
    return arrays, optional, float(fps_array), available_keys


def run(controls_path, generated_dir, output_dir):
    controls_path, generated_dir, output_dir = map(Path, (controls_path, generated_dir, output_dir))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory must be unoccupied: {output_dir}")
    input_hash = sha256(controls_path)
    arrays, optional, fps, keys = load_controls(controls_path)
    source, draft = arrays["source_rgb"], arrays["draft_rgb"]
    if source.ndim != 4 or source.shape[-1] != 3:
        raise ValueError("source_rgb must be [T,H,W,3]")
    count, height, width, _ = source.shape
    media, media_path = media_module()
    raw, raw_receipts = media.load_frames(generated_dir / "raw_frames", count)
    composite, composite_receipts = media.load_frames(generated_dir / "composite_frames", count)
    report, regions = analyze_pixels(source, draft, raw, composite, arrays["edit_mask"],
                                     optional, arrays["reference_rgb"])
    edit = mask_array(arrays["edit_mask"], source.shape[:3], "edit_mask")
    report.update({"created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "integrity_failed" if report["integrity_failures"] else "diagnostic_only_no_quality_acceptance",
        "controls": {"path": str(controls_path.resolve()), "sha256": input_hash,
                     "available_keys": keys, "shape": list(source.shape), "fps": fps},
        "source_code": {"evaluator": sha256(__file__), "media_helper": sha256(media_path)},
        "raw_png_receipts": raw_receipts, "composite_png_receipts": composite_receipts,
        "array_receipts": {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                                  "sha256_raw_bytes": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
                           for key, value in {**arrays, **optional}.items()},
        "limitations": ["Exact preservation describes raw PNG pixels, not lossy MP4 values.",
                        "Known pixels are only as correct as their draft and explicit provenance masks.",
                        "No source-preservation claim is inferred from RGB equality.",
                        "Reference-projected pixels use reference-time appearance, not current observed texture.",
                        "The source-preserved footprint assumes that those background pixels are static.",
                        "RGB similarity is diagnostic; it does not establish freeze correctness or geometric accuracy.",
                        "The generated regions do not have validated metric depth."]})
    generation_manifest = generated_dir / "manifest.json"
    if generation_manifest.is_file():
        generation_metadata = json.loads(generation_manifest.read_text())
        declared_controls_hash = generation_metadata.get("controls", {}).get("sha256")
        if declared_controls_hash != input_hash:
            raise ValueError("Generated manifest controls SHA256 differs from evaluation controls")
        report["generation_manifest"] = {
            "path": str(generation_manifest.resolve()), "sha256": sha256(generation_manifest),
            "declared_status": generation_metadata.get("status"),
            "controls_hash_matches": True}
    else:
        report["generation_manifest"] = {"status": "not_available"}
    sequences = {"source": source, "draft": draft, "raw": raw, "composite": composite}
    titles = {"source": "SOURCE / exact input", "draft": "DRAFT / gray = unknown",
              "raw": "VACE / raw generated RGB",
              "composite": "COMPOSITE / known pixels preserved"}
    # Failure labels cannot imply that a failed integrity check succeeded.
    if report["integrity_failures"]:
        titles["composite"] = "COMPOSITE / integrity check FAILED"
    images = [media.board(sequences, titles, i, fps, height, width) for i in range(count)]
    # Reuse panel layout, replace its generic footer with this comparison's meaning.
    for board in images:
        draw = ImageDraw.Draw(board)
        draw.rectangle((0, height + 60, board.width, board.height), fill=(24, 24, 24))
        draw.text((8, height + 67), "Raw PNG equality checked separately | MP4 is lossy | generated regions have no validated metric depth",
                  font=media.font(12), fill="white")
    chosen = sorted(set(np.linspace(0, count - 1, min(5, count), dtype=int).tolist()))
    output_dir.mkdir(parents=True)
    scale = min(1.0, 1472 / images[0].width)
    contact_size = (round(images[0].width * scale), round(images[0].height * scale))
    contact = Image.new("RGB", (contact_size[0], contact_size[1] * len(chosen)))
    for row, index in enumerate(chosen):
        images[index].save(output_dir / f"comparison_{index:04d}.png")
        contact.paste(images[index].resize(contact_size, Image.Resampling.LANCZOS), (0, row * contact_size[1]))
    contact.save(output_dir / "comparison_contact.png")
    report["comparison_mp4_probe"] = media.video_encode(output_dir / "comparison.mp4", images, fps)
    mask_frames = np.repeat(edit[..., None], 3, axis=-1).astype(np.uint8) * 255
    mask_images = [media.board({"mask": mask_frames}, {"mask": "UNKNOWN / white = generated"},
                              i, fps, height, width) for i in range(count)]
    for board in mask_images:
        draw = ImageDraw.Draw(board)
        draw.rectangle((0, height + 60, board.width, board.height), fill=(24, 24, 24))
        draw.text((8, height + 67), "White: generated | black: kept from draft",
                  font=media.font(12), fill="white")
    report["unknown_mask_mp4_probe"] = media.video_encode(output_dir / "unknown_mask.mp4", mask_images, fps)
    Image.fromarray(np.concatenate([mask_frames[i] for i in chosen], axis=1)).save(output_dir / "unknown_mask_contact.png")
    if sha256(controls_path) != input_hash:
        raise RuntimeError("Controls changed during evaluation")
    for receipt in raw_receipts + composite_receipts:
        if sha256(receipt["path"]) != receipt["file_sha256"]:
            raise RuntimeError(f"Input PNG changed during evaluation: {receipt['path']}")
    report["inputs_verified_after_export"] = True
    report["contact_indices"] = chosen
    report["artifacts"] = {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                           for path in output_dir.iterdir() if path.is_file()}
    (output_dir / "diagnostics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output_dir": str(output_dir.resolve()),
                      "integrity_failures": report["integrity_failures"],
                      "region_pixel_counts": {name: score["pixel_count"] for name, score in report["regions"].items()}}))
    return 1 if report["integrity_failures"] else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return run(args.controls, args.generated_dir, args.output_dir)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
