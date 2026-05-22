"""Generate a synthetic demo image and run the pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "demo"
OUT = ROOT / "outputs" / "demo_run"


def make_demo_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (512, 384), (240, 240, 245))
    draw = ImageDraw.Draw(img)
    draw.ellipse([80, 120, 180, 280], fill=(200, 40, 40), outline=(120, 20, 20))
    draw.rectangle([280, 140, 400, 300], fill=(60, 120, 200), outline=(30, 60, 120))
    draw.rectangle([200, 200, 260, 260], fill=(80, 180, 80), outline=(40, 100, 40))
    img.save(path)


def main() -> None:
    img_path = DATA / "test_scene.png"
    make_demo_image(img_path)
    cmd = [
        sys.executable,
        str(ROOT / "main.py"),
        "--image_path",
        str(img_path),
        "--question",
        "What is the color of the object behind the red cylinder?",
        "--output_dir",
        str(OUT),
        "--use_mock_models",
        "--no_resume",
    ]
    subprocess.check_call(cmd, cwd=str(ROOT))
    print(f"Demo finished: {OUT}")


if __name__ == "__main__":
    main()
