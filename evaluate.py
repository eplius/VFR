"""COCO val2017 evaluation with predictions and reproducible settings."""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from runtime import VFR, sha256

METRICS = ["AP", "AP50", "AP75", "APM", "APL", "AR", "AR50", "AR75", "ARM", "ARL"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["s", "m"], required=True)
    parser.add_argument("--weights-dir")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, help="Defaults to the checkpoint's reference evaluation batch size.")
    parser.add_argument("--limit", type=int, default=0, help="0: all images; nonzero is NOT a full benchmark.")
    args = parser.parse_args()
    if args.limit < 0 or (args.batch_size is not None and args.batch_size < 1):
        parser.error("Limit must be nonnegative and batch size must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty to avoid overwriting an evaluation")
    args.output.mkdir(parents=True, exist_ok=True)
    coco = COCO(str(args.annotations))
    all_ids = sorted(coco.getImgIds())
    ids = all_ids[:args.limit] if args.limit else all_ids
    if not ids:
        parser.error("No images found")
    runner = VFR(args.model, args.weights_dir)
    batch_size = args.batch_size or runner.metadata["evaluation_batch_size"]
    print(json.dumps({"model": args.model, "images": len(ids), "precision": runner.metadata["precision"],
                      "position_embedding": runner.metadata["position_embedding"],
                      "batch_size": batch_size}), flush=True)
    started = time.monotonic()
    predictions = []
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start:start + batch_size]
        images = []
        for image_id in batch_ids:
            with Image.open(args.images / coco.imgs[image_id]["file_name"]) as image:
                images.append(image.convert("RGB"))
        results = runner.predict_batch(images)
        for image_id, result in zip(batch_ids, results):
            predictions.extend(dict(image_id=image_id, **item) for item in result)
        step = start + len(batch_ids)
        if start == 0 or start // 100 != step // 100 or step == len(ids):
            print(f"VFR-{args.model.upper()} {step}/{len(ids)}", flush=True)
    prediction_file = args.output / "predictions.json"
    prediction_file.write_text(json.dumps(predictions), encoding="utf-8")
    evaluator = COCOeval(coco, coco.loadRes(predictions), "keypoints")
    evaluator.params.imgIds = ids
    evaluator.params.catIds = [1]
    evaluator.params.maxDets = [20]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    report = {
        "model": f"VFR-{args.model.upper()}", "dataset": "COCO val2017",
        "images": len(ids), "full_dataset": len(ids) == len(all_ids),
        "standard_val2017_count": len(ids) == 5000,
        "metrics": dict(zip(METRICS, map(float, evaluator.stats))),
        "metric_scale": "0-1; multiply by 100 for AP points",
        "checkpoint_sha256": runner.metadata["sha256"],
        "annotations_sha256": sha256(args.annotations),
        "precision": runner.metadata["precision"],
        "position_embedding": runner.metadata["position_embedding"],
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "batch_size": batch_size, "input_size": [640, 640],
        "max_dets": 20, "nms": False, "score_threshold": None,
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved predictions and metrics to {args.output}", flush=True)


if __name__ == "__main__":
    main()
