"""Predict COCO-format poses for one image, with an optional overlay."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from runtime import VFR

EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
         (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 1),
         (0, 2), (1, 3), (2, 4)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["s", "m"], required=True)
    parser.add_argument("--weights-dir")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/image.json"))
    parser.add_argument("--visualize", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Visualization only; JSON keeps all 60 predictions.")
    args = parser.parse_args()
    image = Image.open(args.input).convert("RGB")
    runner = VFR(args.model, args.weights_dir)
    predictions = runner.predict(image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(predictions), encoding="utf-8")
    if args.visualize:
        draw = ImageDraw.Draw(image)
        for prediction in predictions:
            if prediction["category_id"] != 1 or prediction["score"] < args.threshold:
                continue
            points = list(zip(prediction["keypoints"][::3], prediction["keypoints"][1::3]))
            for a, b in EDGES:
                draw.line([points[a], points[b]], fill=(30, 200, 120), width=3)
            for x, y in points:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(235, 70, 70))
        args.visualize.parent.mkdir(parents=True, exist_ok=True)
        image.save(args.visualize)
    print(f"Saved {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
