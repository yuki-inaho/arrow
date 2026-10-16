#!/usr/bin/env python3
"""Prepare explicit Rerun views for saved depth-detail recordings."""

from __future__ import annotations

import argparse
from pathlib import Path

import rerun.blueprint as rrb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("grass", "railway", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("results/depth_detail_oct16"))
    args = parser.parse_args()
    datasets = ("grass", "railway") if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        visuals = args.output_dir / dataset / "visuals"
        recording = visuals / f"{dataset}_final_pointcloud.rrd"
        if not recording.is_file():
            parser.error(f"recording missing: {recording}")
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="Final depth / SR (arrow_provisional)",
                    origin="/world",
                    contents=["/world/final_points", "/world/sr_arrows"],
                ),
                rrb.Vertical(
                    rrb.Spatial2DView(name="RGB", origin="/world/camera/image"),
                    rrb.Spatial2DView(name="Final depth", origin="/world/camera/depth"),
                ),
                column_shares=[2, 1],
            ),
            auto_layout=False,
            auto_views=False,
            collapse_panels=True,
        )
        destination = visuals / f"{dataset}_view.rbl"
        blueprint.save(f"depth_detail_{dataset}", destination)
        print(f"{recording}\n{destination}")


if __name__ == "__main__":
    main()
