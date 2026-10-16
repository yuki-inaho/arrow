#!/usr/bin/env python3
"""Render saved total / camera-only / residual flow without running inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
from matplotlib.colors import hsv_to_rgb
from PIL import Image, ImageDraw, ImageFont


def flow_rgb(flow, valid, maximum):
    magnitude = np.linalg.norm(np.nan_to_num(flow), axis=-1)
    hue = (np.arctan2(np.nan_to_num(flow[..., 1]), np.nan_to_num(flow[..., 0])) / (2 * np.pi)) % 1
    hsv = np.stack([hue, np.clip(magnitude / maximum, 0, 1), np.ones_like(hue)], axis=-1)
    rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
    rgb[~valid] = [24, 27, 33]
    return rgb


def render(input_root, output_root, dataset):
    source = input_root / dataset
    target = output_root / dataset / "static_world"
    manifest = json.loads((target / "manifest.json").read_text())
    frames = json.loads((source / "frames.json").read_text())["frames"]
    if manifest["render_status"] != "complete" or len(frames) != manifest["frame_count"]:
        raise ValueError("Static export must be complete before diagnostic rendering")
    sample = []
    for i in range(len(frames) - 1):
        with np.load(target / "motion" / f"{i:06d}.npz") as data:
            mask = data["valid"][::16, ::16]
            for key in ("rigid_flow", "nonrigid_flow", "total_flow"):
                sample.append(np.linalg.norm(data[key][::16, ::16][mask], axis=-1))
    maximum = max(0.01, float(np.percentile(np.concatenate(sample), 98)))
    h, w = manifest["original_size"]
    cw = 420
    ch = round(h * cw / w)
    canvas_w, canvas_h = 2 * cw, 2 * (ch + 32) + 40
    canvas_h += canvas_h % 2
    path = target / f"{dataset}_motion_decomposition.mp4"
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=manifest["fps"])
    stream.width, stream.height = canvas_w, canvas_h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "fast"}
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    selected = []
    try:
        for i, record in enumerate(frames):
            original = Image.open(record["rgb_path"]).convert("RGB")
            panels = [original]
            if i + 1 < len(frames):
                with np.load(target / "motion" / f"{i:06d}.npz") as data:
                    valid = data["valid"]
                    for key in ("total_flow", "rigid_flow", "nonrigid_flow"):
                        panels.append(Image.fromarray(flow_rgb(data[key], valid, maximum)))
            else:
                panels.extend([Image.new("RGB", (w, h), (24, 27, 33))] * 3)
            canvas = Image.new("RGB", (canvas_w, canvas_h), (15, 17, 22))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 8), f"Frame {i + 1}/{len(frames)} | white=zero | hue=direction | saturation={maximum:.2f} px", font=font, fill="white")
            titles = ("Observed RGB", "Total apparent flow", "Rigid: camera only", "Non-rigid: non-camera residual")
            for j, (panel, title) in enumerate(zip(panels, titles)):
                x, y = (j % 2) * cw, 40 + (j // 2) * (ch + 32)
                draw.text((x + 8, y + 6), title, font=font, fill="white")
                canvas.paste(panel.resize((cw, ch)), (x, y + 32))
            if i + 1 == len(frames):
                draw.text((cw + 15, ch // 2), "No next frame / no flow", font=font, fill="white")
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")):
                container.mux(packet)
            if i in {0, len(frames) // 2, len(frames) - 2}:
                selected.append(np.asarray(canvas))
            print(f"{dataset} flow display {i + 1}/{len(frames)}", flush=True)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    Image.fromarray(np.concatenate(selected, axis=0)).save(target / f"{dataset}_motion_decomposition.png")
    (target / "motion_display.json").write_text(json.dumps({"frame_count": len(frames), "fps": manifest["fps"],
        "flow_range_pixel": [0, maximum], "flow_hue": "atan2(v,u)", "zero_flow": "white",
        "invalid": "dark gray", "last_frame": "no next frame", "native_flow_units": "pixel",
        "dimensions": [canvas_w, canvas_h], "video": path.name}, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results/depth_detail_oct16"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/sr_static_oct16"))
    parser.add_argument("--dataset", choices=("grass", "railway", "all"), default="all")
    args = parser.parse_args(argv)
    for name in (("grass", "railway") if args.dataset == "all" else (args.dataset,)):
        render(args.input_dir, args.output_dir, name)


if __name__ == "__main__":
    main()
