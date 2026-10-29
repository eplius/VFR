# Release validation

Validation date: 2026-10-29. Tests ran on one NVIDIA GeForce RTX 2080 Ti;
both models evaluated all 5,000 COCO val2017 images with no score threshold
or NMS. VFR-S uses its AMP deployment profile at batch size 1; VFR-M uses
FP32 at the original evaluation batch size of 12.

| Model | AP | AP50 | AP75 | APM | APL |
| --- | ---: | ---: | ---: | ---: | ---: |
| VFR-S | 67.5522 | 87.5132 | 73.3685 | 61.1190 | 77.7405 |
| VFR-M | 70.5719 | 89.2922 | 76.6742 | 64.3341 | 80.2916 |

The table is in AP points (0-100). The JSON reports retain full precision
on the 0-1 scale, annotation and model SHA-256 hashes, and evaluation settings.

## Checks performed

- Python syntax checks for the runtime, inference, evaluation, and private exporter.
- Strict loading of each source checkpoint's EMA state before exporting.
- Three-image source-forward versus reloaded TorchScript output comparison
  in each deployment profile's precision:
  VFR-S was exact on the tested images; VFR-M maximum absolute differences were
  3.58e-6 for logits and 8.20e-7 for normalized keypoint coordinates.
- Standalone image inference, prediction JSON, and skeleton visualization for S/M.
- Full COCO evaluation through the standalone package for both models.
- Runtime code has no imports from the private project's model/config/training tree.

Three-image parity is not proof of equality on every possible input. The full
COCO measurements above are the release's actual benchmark results.

## Evaluation alignment

The original checkpoint training logs recorded S=67.5761 and M=70.5719 AP.
The release results round to 67.6 and 70.6, respectively. Full-precision release
measurements remain in the JSON reports; they are not replaced by paper values.

Re-running the original M evaluation command confirmed 70.6. With the same exported
M asset, restoring
the original contiguous batched evaluation (batch size 12) measured
70.5718777295 AP, versus the historical 70.5718526554. No M weights, temperature,
score threshold, NMS, or annotations were changed. Twelve paired images had
identical preprocessed pixel values; batched and single-image forward outputs
differed at floating-point precision. The release now defaults to the reference
batch size instead of forcing all models to batch size 1.

The original evaluator does not enable autocast merely because `--amp` is passed
to `train.py`. S is explicitly released as a separately validated AMP deployment
profile, with measured AP 67.5522.

Use the position-embedding, precision, and batch settings documented in the main
README and manifests. The two profiles are not an identical-forward-protocol
comparison of S/M capacity alone.

## Files

- `coco_s.json`, `coco_m.json`: full-dataset evaluation metrics.
- `parity_s.json`, `parity_m.json`: three-image graph comparison results.
- `environment.json`: versions of the tested runtime dependencies.

Full prediction files stay outside Git under `results/`. For the validation run,
they are `results/coco_release_validation/s/predictions.json` and
`results/coco_release_v2/m/predictions.json`. They may be
published separately as release assets if desired. No training losses, optimizer
states, or dataset images are part of this release.
