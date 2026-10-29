# Paper figures

- `framework.png`: Figure 1, overview of Visibility-aware Fine-grained Refinement.
- `comparison_metrics.png`: Figure 4, COCO accuracy versus latency, parameter count,
  and GFLOPs for lightweight pose estimators.

The PNGs are web-optimized copies of the manuscript figures. They are resized
with Lanczos resampling, palette-optimized, and saved with PNG compression;
metadata is omitted. Content, labels, plotted values, and layouts are preserved.
The original manuscript images are unchanged. Figure numbers follow the
manuscript, not the order of images in this repository.

| Asset | Web dimensions | File size |
| --- | --- | ---: |
| `framework.png` | 1800 x 851 | 180,564 bytes |
| `comparison_metrics.png` | 2400 x 776 | 84,223 bytes |

The combined download is approximately 259 KiB instead of 1.95 MiB, an 87.0%
reduction. Full numeric comparisons are also available as text tables in the
main README, so they remain accessible without loading the figures.

Figure 4 reports the paper's TensorRT v10.12 FP16, batch-size-1 benchmark on an
NVIDIA GeForce RTX2080Ti. It is not a TorchScript runtime benchmark for this release.
