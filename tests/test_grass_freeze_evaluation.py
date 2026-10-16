"""Independent mathematical/IO checks for the diagnostic helper (no GPU)."""
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np
from PIL import Image

import importlib.util

_spec = importlib.util.spec_from_file_location("evaluate_grass_freeze", Path(__file__).resolve().parents[1] / "scripts/evaluate_grass_freeze.py")
q = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q)


def test_cpu_diagnostics():
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, (160, 160, 3), dtype=np.uint8)
    assert abs(q.ssim_rgb(image, image) - 1) < 1e-12
    inverted = 255 - image
    assert q.ssim_rgb(image, inverted) < 0
    frame = q.board({"source": image[None]}, {"source": "test"}, 0, 5, 160, 160)
    assert np.array_equal(np.asarray(frame)[60:220], image)
    small = q.fitted_tile(image[:120], 160, 160)
    assert np.array_equal(np.asarray(small)[20:140], image[:120])
    blank = np.zeros((160, 160), np.uint8)
    mask = np.ones_like(blank, bool)
    homography, report = q.fit_background_homography(blank, blank, mask, mask)
    assert homography is None and report["status"] == "insufficient_matches"
    texture = cv2.GaussianBlur(rng.integers(0, 256, (400, 400), dtype=np.uint8), (3, 3), .7)
    expected = np.float64([[1, 0, 7], [0, 1, -5], [0, 0, 1]])
    translated = cv2.warpPerspective(texture, expected, (400, 400))
    mask0 = np.zeros_like(texture, bool)
    mask0[20:-20, 20:-20] = True
    homography, report = q.fit_background_homography(texture, translated, mask0, mask0)
    assert report["status"] == "diagnostic_available", report
    corners = np.float32([[[50, 50], [350, 50], [50, 350], [350, 350]]])
    assert np.max(np.abs(cv2.perspectiveTransform(corners, homography) - cv2.perspectiveTransform(corners, expected))) < .1
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
        folder = Path(folder)
        Image.fromarray(image).save(folder / "frame_0000.png")
        frames, receipts = q.load_frames(folder, 1)
        assert np.array_equal(frames[0], image) and len(receipts) == 1
        try:
            q.load_frames(folder, 2)
        except ValueError:
            pass
        else:
            raise AssertionError("Missing PNG frame accepted")
        video = folder / "probe.mp4"
        probe = q.video_encode(video, [Image.fromarray(image)] * 3, 5)
        assert probe["nb_read_frames"] == "3"
    result = {"status": "pass", "checks": ["SSIM identity/inversion", "raw board pixels preserved",
                "aspect fit preserves unchanged image", "blank registration unavailable", "known homography translation",
                "PNG exact sequence", "missing frame rejection", "MP4 decode frame count"]}
    print(json.dumps(result))


if __name__ == "__main__":
    test_cpu_diagnostics()
