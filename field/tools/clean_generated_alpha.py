"""Convert a generated checkerboard-backed cutout into a real RGBA asset.

The generator's checker field is neutral gray and connected to the canvas edge.
Warm limestone and painted details are deliberately preserved, even when bright.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

from PIL import Image, ImageFilter


def neutral_background(pixel: tuple[int, int, int]) -> bool:
    high = max(pixel)
    low = min(pixel)
    return high - low <= 7 and high >= 72


def extract(input_path: Path, output_path: Path, padding: int = 28) -> dict[str, object]:
    source = Image.open(input_path).convert("RGB")
    width, height = source.size
    pixels = source.load()
    background = bytearray(width * height)
    queue: deque[tuple[int, int]] = deque()

    def seed(x: int, y: int) -> None:
        index = y * width + x
        if not background[index] and neutral_background(pixels[x, y]):
            background[index] = 1
            queue.append((x, y))

    for x in range(width):
        seed(x, 0)
        seed(x, height - 1)
    for y in range(height):
        seed(0, y)
        seed(width - 1, y)

    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            index = ny * width + nx
            if background[index] or not neutral_background(pixels[nx, ny]):
                continue
            background[index] = 1
            queue.append((nx, ny))

    mask = Image.new("L", (width, height), 255)
    mask.putdata([0 if value else 255 for value in background])
    mask = mask.filter(ImageFilter.GaussianBlur(0.55))
    rgba = source.convert("RGBA")
    rgba.putalpha(mask)

    bounds = mask.getbbox()
    if bounds is None:
        raise RuntimeError("cleanup removed the entire image")
    left = max(0, bounds[0] - padding)
    top = max(0, bounds[1] - padding)
    right = min(width, bounds[2] + padding)
    bottom = min(height, bounds[3] + padding)
    cropped = rgba.crop((left, top, right, bottom))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_options: dict[str, object] = {"optimize": True}
    if output_path.suffix.lower() == ".webp":
        save_options.update({"quality": 90, "method": 6})
    cropped.save(output_path, **save_options)
    alpha = cropped.getchannel("A")
    histogram = alpha.histogram()
    return {
        "source_size": source.size,
        "output_size": cropped.size,
        "bounds": (left, top, right, bottom),
        "transparent_pixels": histogram[0],
        "opaque_pixels": histogram[255],
        "edge_pixels": sum(histogram[1:255]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--padding", type=int, default=28)
    args = parser.parse_args()
    print(extract(args.input, args.output, args.padding))


if __name__ == "__main__":
    main()
